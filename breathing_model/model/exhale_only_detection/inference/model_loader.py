from typing import Optional, Tuple

import torch
import numpy as np
from breathing_model.model.exhale_only_detection.model import BreathPhaseTransformerSeq
from breathing_model.model.exhale_only_detection.utils import ModelConfig, DataConfig


class BreathPhaseClassifier:
    def __init__(self,
                 model_path: str,
                 model_config: ModelConfig,
                 data_config: DataConfig,
                 device: Optional[torch.device] = None,
                 strict: bool = True):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config
        self.model = self._load_model(model_path, model_config, strict)

        self.last_mel_frames = int(0.2 * data_config.sample_rate / data_config.hop_length)

    def _load_model(self,
                    model_path: str,
                    config: ModelConfig,
                    strict: bool) -> BreathPhaseTransformerSeq:
        model = BreathPhaseTransformerSeq(
            n_mels=config.n_mels,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            num_classes=config.num_classes
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # If state_dict shapes mismatch (e.g., positional encoding length), try to adapt
        model_state = model.state_dict()
        adapted = False
        new_state = {}
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape != model_state[k].shape:
                    # Handle positional encoding resizing by copying overlapping prefix
                    if 'pos_encoder.pe' in k:
                        target = model_state[k]
                        # create a tensor with target shape and copy min prefix
                        new_v = target.clone()
                        min_len = min(v.shape[1], target.shape[1]) if v.ndim >= 3 else min(v.shape[0], target.shape[0])
                        try:
                            # assume shape [1, seq_len, d_model]
                            new_v[:, :min_len, ...] = v[:, :min_len, ...]
                        except Exception:
                            # fallback: flatten-copy as much as possible
                            flat_src = v.flatten()
                            flat_dst = new_v.flatten()
                            flat_dst[:min(flat_src.numel(), flat_dst.numel())] = flat_src[:flat_dst.numel()]
                            new_v = flat_dst.view_as(new_v)
                        new_state[k] = new_v
                        adapted = True
                        continue
                    else:
                        # Skip incompatible parameter and warn
                        print(f"Warning: skipping incompatible parameter '{k}' with shape {v.shape} (expected {model_state[k].shape})")
                        adapted = True
                        continue
                else:
                    new_state[k] = v
            else:
                # param not present in model -> skip
                adapted = True
                continue

        if adapted:
            # Merge new_state into model_state for loading (non-strict)
            merged = model_state.copy()
            merged.update(new_state)
            model.load_state_dict(merged, strict=False)
        else:
            model.load_state_dict(state_dict, strict=strict)
        model.eval()
        return model

    @torch.no_grad()
    def predict_frames(self,
                       mel_tensor: torch.Tensor,
                       src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (preds_frame, probs_frame_mean).
        Args:
            mel_tensor: Tensor of shape [1, 1, n_mels, T]
            src_key_padding_mask: optional [1, T] (bool), True = position to ignore
        """
        mel_tensor = mel_tensor.to(self.device)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(self.device)

        logits = self.model(mel_tensor, src_key_padding_mask=src_key_padding_mask)  # [1, T, C]
        probs = torch.softmax(logits, dim=-1)  # [1, T, C]

        probs_np = probs.squeeze(0).cpu().numpy()  # [T, C]
        if probs_np.shape[0] == 0:
            # Empty sequence: return a uniform distribution
            num_classes = self.model_config.num_classes
            return np.array([], dtype=int), np.full((num_classes,), 1.0 / num_classes, dtype=float)

        recent_probs = probs_np[-self.last_mel_frames:]

        frame_preds = recent_probs.argmax(axis=1)  # [T]
        mean_probs = recent_probs.mean(axis=0)     # [C]
        return frame_preds, mean_probs

    @torch.no_grad()
    def predict(self,
                mel_tensor: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[int, np.ndarray]:
        """
        Classification of the entire sequence (majority of frames).
        Returns (predicted_class, mean_class_probs).
        """
        # predict_frames returns (frame_predictions, mean_class_probabilities)
        _, mean_probs = self.predict_frames(mel_tensor, src_key_padding_mask)

        # Select the class with the highest mean probability across all frames
        predicted_class = int(np.argmax(mean_probs))
        return predicted_class, mean_probs
