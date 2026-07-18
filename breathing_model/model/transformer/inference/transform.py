import torch
import torchaudio
import numpy as np
from breathing_model.model.transformer.utils import DataConfig
from torch import nn


class MelSpectrogramTransform(nn.Module):
    def __init__(self, config: DataConfig):
        super().__init__()

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power')

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if isinstance(signal, np.ndarray):
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            waveform = torch.tensor(signal).unsqueeze(0)
        else:  # It's a tensor
            waveform = signal

        mel = self.transform(waveform)
        mel = self.db_transform(mel)
        mel = mel.unsqueeze(1)  # Add channel dimension for CNN
        return mel
