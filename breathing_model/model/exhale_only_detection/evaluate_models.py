import os
import sys
import argparse
import json
from collections import deque
from typing import List, Tuple

import numpy as np
import torchaudio
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Ensure project root in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from breathing_model.model.exhale_only_detection.utils import Config, BreathType
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform
from breathing_model.model.exhale_only_detection.inference.model_loader import BreathPhaseClassifier
from breathing_model.model.exhale_only_detection.inference.main_onnx import OnnxBreathPhaseClassifier


def load_audio(wav_path: str, target_sr: int) -> np.ndarray:
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy().astype(np.float32)


def parse_label_csv(csv_path: str) -> List[dict]:
    import csv
    labels = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            cls, start_s, end_s = row[0], row[1], row[2]
            try:
                start_i = int(start_s)
                end_i = int(end_s)
            except ValueError:
                continue
            labels.append({'class': cls.strip().lower(), 'start': start_i, 'end': end_i})
    return labels


def chunk_majority_gt(labels: List[dict], start: int, end: int, invert: bool = False) -> int:
    # Binary mapping: exhale -> 0, other -> 1
    counts = {0: 0, 1: 0}
    counts[1] = end - start
    for lab in labels:
        cls_id = 0 if lab['class'] == 'exhale' else 1
        overlap_start = max(lab['start'], start)
        overlap_end = min(lab['end'], end)
        overlap = overlap_end - overlap_start
        if overlap > 0:
            counts[cls_id] += overlap
            counts[1] -= overlap
    res = 0 if counts[0] >= counts[1] else 1
    return 1 - res if invert else res


def evaluate_single_model(model_path: str, config: Config, config_dir: str, limit_wav: int = None,
                          wav_dir_override: str = None, label_dir_override: str = None,
                          thresholds: List[float] = None, invert_labels: bool = False) -> dict:
    print(f"Evaluating model: {model_path}")
    is_onnx = model_path.lower().endswith('.onnx')

    # Prefer paths specified in the config file
    wav_dir = getattr(config.data, 'data_dir', None)
    label_dir = getattr(config.data, 'label_dir', None)
    # Apply explicit overrides if provided
    if wav_dir_override:
        wav_dir = wav_dir_override
    if label_dir_override:
        label_dir = label_dir_override
    if not wav_dir:
        wav_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'eval', 'raw'))
    if not label_dir:
        label_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'eval', 'label'))

    # Resolve relative paths: CLI overrides -> repo root, config paths -> config dir
    if wav_dir and not os.path.isabs(wav_dir):
        if wav_dir_override:
            wav_dir = os.path.normpath(os.path.join(PROJECT_ROOT, wav_dir))
        else:
            wav_dir = os.path.normpath(os.path.join(config_dir, wav_dir))
    if label_dir and not os.path.isabs(label_dir):
        if label_dir_override:
            label_dir = os.path.normpath(os.path.join(PROJECT_ROOT, label_dir))
        else:
            label_dir = os.path.normpath(os.path.join(config_dir, label_dir))

    # If configured paths don't exist, try common evaluation folders
    if not os.path.exists(wav_dir):
        candidate = os.path.normpath(os.path.join(config_dir, '..', '..', 'data', 'eval', 'raw'))
        if os.path.exists(candidate):
            wav_dir = candidate
    if not os.path.exists(label_dir):
        candidate = os.path.normpath(os.path.join(config_dir, '..', '..', 'data', 'eval', 'label'))
        if os.path.exists(candidate):
            label_dir = candidate

    if not os.path.exists(wav_dir) or not os.path.exists(label_dir):
        raise FileNotFoundError(f"Data directories not found. wav_dir={wav_dir}, label_dir={label_dir}")

    if is_onnx:
        classifier = OnnxBreathPhaseClassifier(model_path, config)
        use_mel = False
    else:
        # Try loading PTH with strict=False to tolerate e.g. positional-encoding size mismatches
        try:
            classifier = BreathPhaseClassifier(model_path, config.model, config.data, strict=False)
        except Exception as e:
            print(f"Warning: non-strict load failed ({e}), retrying strict load...")
            classifier = BreathPhaseClassifier(model_path, config.model, config.data, strict=True)
        mel_transform = MelSpectrogramTransform(config.data)
        use_mel = True

    wav_files = sorted([f for f in os.listdir(wav_dir) if f.lower().endswith('.wav')])
    if limit_wav:
        wav_files = wav_files[:limit_wav]

    all_y_true = []
    all_y_pred = []
    all_probs = []

    chunk_size = int(config.audio.chunk_length * config.data.sample_rate)
    buffer_seconds = 3.5
    max_buffer = int(buffer_seconds * config.data.sample_rate)

    for wav_name in wav_files:
        base = os.path.splitext(wav_name)[0]
        wav_path = os.path.join(wav_dir, wav_name)
        csv_path = os.path.join(label_dir, f"{base}.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {wav_name} (no label)")
            continue

        audio = load_audio(wav_path, config.data.sample_rate)
        n_samples = audio.shape[0]
        labels = parse_label_csv(csv_path)

        audio_buffer = deque(maxlen=max_buffer)

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = audio[start:end]
            if chunk.size == 0:
                continue
            audio_buffer.extend(chunk)
            buf_np = np.array(audio_buffer, dtype=np.float32)
            if buf_np.size == 0:
                continue

            if use_mel:
                mel = mel_transform(buf_np)
                pred_cls, probs = classifier.predict(mel)
            else:
                pred_cls, probs = classifier.predict(buf_np)

            gt = chunk_majority_gt(labels, start, end, invert=invert_labels)
            all_y_true.append(gt)
            all_y_pred.append(int(pred_cls))
            try:
                all_probs.append(float(probs[0]) if len(probs.shape) == 1 else float(probs[0]))
            except Exception:
                all_probs.append(None)

    results = {}
    if len(all_y_true) == 0:
        print("No chunks processed for this model.")
        return results
    # Prepare arrays
    probs_arr = np.array([p if p is not None else np.nan for p in all_probs])
    y_true = np.array(all_y_true)

    out_dir = os.path.join(os.path.dirname(__file__), 'evaluation_output')
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(model_path).replace('.', '_')

    per_thr_results = {}
    if thresholds is None:
        thresholds = [0.5]

    for thr in thresholds:
        thr_key = f"thr_{str(thr).replace('.', '_')}"
        # Compute predictions from probs where available; fallback to original preds
        y_pred_thr = []
        for i, p in enumerate(probs_arr):
            if not np.isnan(p):
                # treat p as probability of class 1 (Other)
                y_pred_thr.append(1 if p >= thr else 0)
            else:
                y_pred_thr.append(all_y_pred[i])
        y_pred_thr = np.array(y_pred_thr)

        acc = accuracy_score(y_true, y_pred_thr)
        clf_report = classification_report(y_true, y_pred_thr, target_names=['Exhale', 'Other'], digits=4, zero_division=0)
        cm = confusion_matrix(y_true, y_pred_thr)

        thr_results = {
            'threshold': thr,
            'num_chunks': int(len(y_true)),
            'accuracy': float(acc),
            'classification_report': clf_report,
            'confusion_matrix': cm.tolist(),
        }

        # ROC AUC if possible
        try:
            valid = ~np.isnan(probs_arr)
            if valid.sum() > 0:
                roc = roc_auc_score(y_true[valid], probs_arr[valid])
                thr_results['roc_auc'] = float(roc)
        except Exception:
            thr_results['roc_auc'] = None

        # Save confusion matrix plot per threshold
        cm_path = os.path.join(out_dir, f"confusion_{base_name}_{thr_key}.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', xticklabels=['Exhale', 'Other'], yticklabels=['Exhale', 'Other'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion matrix: {os.path.basename(model_path)} (thr={thr})')
        plt.savefig(cm_path)
        plt.close()
        thr_results['confusion_matrix_plot'] = cm_path

        # Save textual report per threshold
        report_path = os.path.join(out_dir, f"report_{base_name}_{thr_key}.txt")
        with open(report_path, 'w', encoding='utf-8') as fh:
            fh.write(f"Model: {model_path}\n")
            fh.write(f"Threshold: {thr}\n")
            fh.write(f"Num chunks: {thr_results['num_chunks']}\n")
            fh.write(f"Accuracy: {thr_results['accuracy']:.4f}\n\n")
            fh.write("Classification report:\n")
            fh.write(thr_results['classification_report'])
        thr_results['report_path'] = report_path

        # Save machine-readable JSON per threshold
        json_path = os.path.join(out_dir, f"metrics_{base_name}_{thr_key}.json")
        with open(json_path, 'w') as fh:
            json.dump(thr_results, fh, indent=2)
        thr_results['metrics_json'] = json_path

        print(f"Saved confusion matrix to {cm_path}")
        print(f"Saved text report to {report_path}")
        print(f"Saved JSON metrics to {json_path}")

        per_thr_results[str(thr)] = thr_results

    results['model_path'] = model_path
    results['per_threshold'] = per_thr_results

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='List of model paths (relative to this script) to evaluate', required=False)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of wav files to process')
    parser.add_argument('--wav_dir', type=str, default=None, help='Override wav directory to evaluate (absolute or relative to repo root)')
    parser.add_argument('--label_dir', type=str, default=None, help='Override label directory to evaluate (absolute or relative to repo root)')
    parser.add_argument('--thresholds', type=str, default=None, help='Comma-separated thresholds (e.g. 0.3,0.5,0.6)')
    parser.add_argument('--flip-labels', action='store_true', help='Invert label mapping (swap Exhale/Other)')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = Config.from_yaml(config_path)

    # Default models (look for best_models folder)
    model_folder = os.path.join(os.path.dirname(__file__), 'best_models')
    default_models = [
        os.path.join(model_folder, 'best_model_epoch_18_new.pth'),
        os.path.join(model_folder, 'best_model_epoch_18_new.onnx'),
        os.path.join(model_folder, 'best_model_epoch_21.pth'),
        os.path.join(model_folder, 'best_model_epoch_21.onnx'),
    ]

    models = args.models if args.models else default_models

    all_results = {}
    for m in models:
        if not os.path.isabs(m):
            m_path = os.path.normpath(os.path.join(os.path.dirname(__file__), m))
        else:
            m_path = m
        if not os.path.exists(m_path):
            print(f"Model not found: {m_path}, skipping.")
            continue
        # parse thresholds
        if args.thresholds:
            try:
                thr_list = [float(x) for x in args.thresholds.split(',') if x.strip() != '']
            except Exception:
                print(f"Invalid thresholds: {args.thresholds}, using default 0.5")
                thr_list = None
        else:
            thr_list = None

        res = evaluate_single_model(m_path, config, os.path.dirname(config_path), limit_wav=args.limit,
                wav_dir_override=args.wav_dir, label_dir_override=args.label_dir, thresholds=thr_list,
                invert_labels=args.flip_labels)
        all_results[os.path.basename(m_path)] = res

    summary_path = os.path.join(os.path.dirname(__file__), 'evaluation_output', 'summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(all_results, fh, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == '__main__':
    main()
