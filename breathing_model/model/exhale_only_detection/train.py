import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing import Tuple, Optional, Dict

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from torch import nn, optim
from torch.utils.data import DataLoader

import sys
# Add project root to path to allow relative imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from breathing_model.model.exhale_only_detection.dataset import BreathDataset, collate_fn, calculate_class_weights
from breathing_model.model.exhale_only_detection.model import BreathPhaseTransformerSeq
from breathing_model.model.exhale_only_detection.utils import split_dataset, load_yaml

IGNORE_INDEX = -100    # value to give to CrossEntropyLoss for ignored positions

def run_train_epoch(model: nn.Module,
                    data_loader: DataLoader,
                    loss_function: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    scheduler):
    model.train()

    total_loss_weighted = 0.0
    total_valid_frames = 0
    total_correct_predictions = 0

    for batch_index, (spectrograms_batch, labels_batch, padding_mask_batch) in enumerate(data_loader):
        spectrograms_batch = spectrograms_batch.to(device)
        labels_batch = labels_batch.to(device)
        padding_mask_batch = padding_mask_batch.to(device)

        labels_for_loss = labels_batch.clone()
        labels_for_loss[padding_mask_batch] = IGNORE_INDEX  # ignored by CrossEntropyLoss

        optimizer.zero_grad()

        # Forward pass: pass src_key_padding_mask (True = padded positions)
        outputs = model(spectrograms_batch, src_key_padding_mask=padding_mask_batch) # [batch_size,
                                                                                    # time_frames,
                                                                                    # num_classes]

        # Compute loss: CrossEntropyLoss expects [N, C] logits and [N] targets
        logits_flat = outputs.view(-1, outputs.size(-1))  # [B*T, num_classes]
        targets_flat = labels_for_loss.view(-1)  # [B*T]

        batch_loss = loss_function(logits_flat, targets_flat)

        # Count valid frames in this batch to weight loss averaging correctly
        valid_frames_in_batch = (~padding_mask_batch).sum().item()  # number of frames with real labels

        # If there are no valid frames in this batch (all padding) skip metric counting but still propagate loss
        # Note: loss_function will return 0 if there are no valid elements; guard divide by zero below.
        batch_loss_value = batch_loss.item()

        # Backprop and optimizer step
        batch_loss.backward()
        optimizer.step()
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        # Accumulate weighted loss
        total_loss_weighted += batch_loss_value * valid_frames_in_batch
        total_valid_frames += valid_frames_in_batch

        # Compute predictions and accumulate correct count for valid frames
        predicted_labels = torch.argmax(outputs, dim=-1)  # [B, T]
        # valid_frame_mask: True where we HAVE real labels (not pad)
        valid_frame_mask = ~padding_mask_batch
        correct_predictions_in_batch = ((predicted_labels == labels_batch) & valid_frame_mask).sum().item()
        total_correct_predictions += correct_predictions_in_batch

    average_loss = (total_loss_weighted / total_valid_frames) if total_valid_frames > 0 else 0.0
    accuracy = (total_correct_predictions / total_valid_frames) if total_valid_frames > 0 else 0.0

    return average_loss, accuracy

def run_validation_epoch(model: nn.Module,
                         data_loader: DataLoader,
                         loss_function: nn.Module,
                         device: torch.device) -> Dict[str, float]:
    """
    Runs validation epoch. Returns a dictionary of metrics.
    """
    model.eval()

    total_loss_weighted = 0.0
    total_valid_frames = 0
    total_correct_predictions = 0

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for spectrograms_batch, labels_batch, padding_mask_batch in data_loader:
            spectrograms_batch = spectrograms_batch.to(device)
            labels_batch = labels_batch.to(device)
            padding_mask_batch = padding_mask_batch.to(device)

            labels_for_loss = labels_batch.clone()
            labels_for_loss[padding_mask_batch] = IGNORE_INDEX

            outputs = model(spectrograms_batch, src_key_padding_mask=padding_mask_batch)  # [B, T, num_classes]

            logits_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = labels_for_loss.view(-1)

            batch_loss = loss_function(logits_flat, targets_flat)
            valid_frames_in_batch = (~padding_mask_batch).sum().item()
            total_loss_weighted += batch_loss.item() * valid_frames_in_batch
            total_valid_frames += valid_frames_in_batch

            predicted_labels = torch.argmax(outputs, dim=-1)
            valid_frame_mask = ~padding_mask_batch

            # Collect predictions and labels for metrics calculation
            all_predictions.append(predicted_labels[valid_frame_mask].cpu().numpy())
            all_true_labels.append(labels_batch[valid_frame_mask].cpu().numpy())

    avg_loss = (total_loss_weighted / total_valid_frames) if total_valid_frames > 0 else 0.0

    if not all_predictions:
        return {"loss": avg_loss, "accuracy": 0, "precision_exhale": 0, "recall_exhale": 0, "f1_exhale": 0}

    # Concatenate all batches
    all_predictions_np = np.concatenate(all_predictions)
    all_true_labels_np = np.concatenate(all_true_labels)

    # Calculate metrics using sklearn for robustness
    accuracy = accuracy_score(all_true_labels_np, all_predictions_np)

    # Calculate precision, recall, f1 for each class.
    # labels=[0, 1] ensures we get scores for both classes even if one isn't predicted.
    # zero_division=0 prevents warnings if a class is never predicted.
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels_np, all_predictions_np, labels=[0, 1], average=None, zero_division=0
    )

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision_exhale": precision[0], "recall_exhale": recall[0], "f1_exhale": f1[0],
        "precision_other": precision[1], "recall_other": recall[1], "f1_other": f1[1],
    }
    return metrics

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                num_epochs: int,
                optimizer: optim.Optimizer,
                scheduler,
                save_directory: str,
                patience: int = 6,
                class_weights: Optional[torch.Tensor] = None) -> None:
    """
    Full training loop with early stopping based on validation loss.
    """
    os.makedirs(save_directory, exist_ok=True)

    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=class_weights)
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = run_train_epoch(model, train_loader, loss_function, optimizer, device, scheduler)
        print(f"Epoch {epoch} / {num_epochs} - Train Loss: {train_loss:.6f} | Train Acc: {train_accuracy:.4f}")

        val_metrics = run_validation_epoch(model, val_loader, loss_function, device)
        val_loss = val_metrics['loss']
        print(f"Epoch {epoch} / {num_epochs} - Val   Loss: {val_loss:.6f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"                 └─ Exhale: P={val_metrics['precision_exhale']:.4f} | R={val_metrics['recall_exhale']:.4f} | F1={val_metrics['f1_exhale']:.4f}")

        # Step scheduler (if provided)
        # OneCycleLR is stepped per batch, StepLR per epoch
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        # Checkpointing based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            save_path = os.path.join(save_directory, f"best_model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"Saved best model to {save_path}")
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s).")

        if epochs_since_improvement >= patience:
            print("Early stopping triggered.")
            break

    print("Training finished.")


def main():
    # === Load config ===
    config = load_yaml("./config.yaml")

    # === Device ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Datasets ===
    # Create training dataset with augmentation
    print("Loading training dataset...")
    train_dataset = BreathDataset(
        data_dir=config['data']['data_dir'],
        label_dir=config['data']['label_dir'],
        sample_rate=config['data']['sample_rate'],
        n_mels=config['data']['n_mels'],
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        augment=config['augment']['enabled'],
        p_noise=config['augment']['p_noise'],
        p_volume=config['augment']['p_volume'],
        p_shift=config['augment']['p_shift'],
        volume_range=tuple(config['augment']['volume_range']),
        noise_factor_range=tuple(config['augment']['noise_factor_range']),
        max_shift_seconds=config['augment']['max_shift_seconds'],
        seed=config['augment']['seed']
    )

    # Create validation dataset from a separate directory for unseen people.
    # Augmentation is disabled for the validation set.
    print("Loading validation dataset for unseen people...")
    val_dataset = BreathDataset(
        data_dir='../../data/eval_unseen_people/raw',
        label_dir='../../data/eval_unseen_people/label',
        sample_rate=config['data']['sample_rate'],
        n_mels=config['data']['n_mels'],
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        augment=False,  # No augmentation for validation
    )

    # === Calculate Class Weights for Loss Function ===
    # Use the training dataset to get the class distribution for weighting the loss.
    # These weights will be passed to the loss function to penalize errors on the
    # minority class ('exhale') more heavily.
    class_weights = calculate_class_weights(train_dataset).to(device)

    # === DataLoaders ===
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=device.type == 'cuda'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=device.type == 'cuda'
    )

    # === Model ===
    model = BreathPhaseTransformerSeq(
        n_mels=config['model']['n_mels'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes']
    ).to(device)

    # === Optimizer ===
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )

    # === Scheduler ===
    scheduler_cfg = config['scheduler']
    scheduler_type = scheduler_cfg['type'].lower()

    if scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_cfg['max_lr'],
            epochs=config['train']['num_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=scheduler_cfg['pct_start'],
            anneal_strategy=scheduler_cfg['anneal_strategy'],
            div_factor=scheduler_cfg['div_factor'],
            final_div_factor=scheduler_cfg['final_div_factor'],
        )
    elif scheduler_type == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg['step_size'],
            gamma=scheduler_cfg['gamma']
        )
    else:
        scheduler = None

    # === Training ===
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=config['train']['num_epochs'],
        optimizer=optimizer,
        scheduler=scheduler,
        save_directory=config['train']['save_dir'],
        patience=config['train']['patience'],
        class_weights=class_weights
    )


if __name__ == "__main__":
    main()