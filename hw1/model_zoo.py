"""Model definitions for HW1 piano transcription."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as transforms

from config import F_MAX, F_MIN, HOP_SIZE, N_FFT, N_MELS, N_PIANO_KEYS, SAMPLE_RATE


class LogMelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.melspectrogram = transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_SIZE,
            f_min=F_MIN,
            f_max=F_MAX,
            n_mels=N_MELS,
            normalized=False,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # pretty_midi.get_piano_roll uses ceil frame indexing while MelSpectrogram
        # uses round-like indexing, so we prepend half-hop for better alignment.
        padded_audio = nn.functional.pad(audio, (HOP_SIZE // 2, 0), "constant")
        mel = self.melspectrogram(padded_audio)[:, :, 1:]
        mel = mel.transpose(-1, -2)
        # mel: (batch_size, time_frames, n_mels)
        mel = torch.log(torch.clamp(mel, min=1e-9))
        return mel


class ConvStack(nn.Module):
    def __init__(self, n_mels: int, cnn_unit: int, fc_unit: int):
        super().__init__()
        self.cnn = nn.Sequential(
            # x: (batch_size, 1, time_frames, n_mels)
            nn.Conv2d(1, cnn_unit, (3, 3), padding=1),
            # x: (batch_size, cnn_unit, time_frames, n_mels)
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            nn.Conv2d(cnn_unit, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            # x: (batch_size, cnn_unit, time_frames, n_mels // 2)
            nn.Dropout(0.25),
            nn.Conv2d(cnn_unit, cnn_unit * 2, (3, 3), padding=1),
            # x: (batch_size, cnn_unit * 2, time_frames, n_mels // 2)
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            # x: (batch_size, cnn_unit * 2, time_frames, n_mels // 4)
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((cnn_unit * 2) * (n_mels // 4), fc_unit),
            nn.Dropout(0.5),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (batch_size, time_frames, n_mels)
        x = mel.unsqueeze(1)
        x = self.cnn(x)
        # x: (batch_size, cnn_unit * 2, time_frames, n_mels // 4)
        x = x.transpose(1, 2).flatten(-2)
        # after transpose(1, 2): (batch_size, time_frames, cnn_unit * 2, n_mels // 4)
        # after flatten(-2): (batch_size, time_frames, (cnn_unit * 2) * (n_mels // 4))
        x = self.fc(x)
        return x


class BiLSTM(nn.Module):
    inference_chunk_length = 512

    def __init__(self, input_features: int, recurrent_features: int):
        super().__init__()
        self.rnn = nn.LSTM(
            input_features,
            recurrent_features,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.rnn(x)[0]

        batch_size, sequence_length, _ = x.shape
        hidden_size = self.rnn.hidden_size
        num_directions = 2 if self.rnn.bidirectional else 1

        h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
        c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
        output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

        slices = list(range(0, sequence_length, self.inference_chunk_length))
        for start in slices:
            end = min(start + self.inference_chunk_length, sequence_length)
            output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

        if self.rnn.bidirectional:
            h.zero_()
            c.zero_()
            for start in reversed(slices):
                end = min(start + self.inference_chunk_length, sequence_length)
                result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

        return output


class BasicOnsetsAndFrames(nn.Module):
    def __init__(self, cnn_unit: int, fc_unit: int):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()
        self.onset_stack = nn.Sequential(
            ConvStack(N_MELS, cnn_unit, fc_unit),
            nn.Linear(fc_unit, N_PIANO_KEYS),
            nn.Sigmoid(),
        )
        self.frame_stack = nn.Sequential(
            ConvStack(N_MELS, cnn_unit, fc_unit),
            nn.Linear(fc_unit, N_PIANO_KEYS),
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel = self.melspectrogram(audio)
        onset_pred = self.onset_stack(mel)
        frame_pred = self.frame_stack(mel)
        return frame_pred, onset_pred


class OnsetsAndFrames(nn.Module):
    def __init__(self, cnn_unit: int, fc_unit: int, rnn_unit: int):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()
        self.onset_stack = nn.Sequential(
            ConvStack(N_MELS, cnn_unit, fc_unit),
            BiLSTM(fc_unit, rnn_unit),
            nn.Linear(rnn_unit * 2, N_PIANO_KEYS),
            nn.Sigmoid(),
        )
        self.frame_stack_conv = nn.Sequential(
            ConvStack(N_MELS, cnn_unit, fc_unit),
            nn.Linear(fc_unit, N_PIANO_KEYS),
            nn.Sigmoid(),
        )
        self.frame_stack_rnn = nn.Sequential(
            BiLSTM(N_PIANO_KEYS * 2, rnn_unit),
            nn.Linear(rnn_unit * 2, N_PIANO_KEYS),
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel = self.melspectrogram(audio)
        onset_pred = self.onset_stack(mel)
        frame_pred_conv_output = self.frame_stack_conv(mel)
        frame_input = torch.cat([onset_pred.detach(), frame_pred_conv_output], dim=-1)
        frame_pred = self.frame_stack_rnn(frame_input)
        return frame_pred, onset_pred


class OffsetConditionedOnsetsAndFrames(nn.Module):
    """Onsets-and-Frames variant with an explicit offset head.

    Design:
    1. Predict offsets from shared acoustic features.
    2. Use offset predictions to condition onset prediction.
    3. Use both onset and offset predictions to condition frame prediction.
    """

    def __init__(self, cnn_unit: int, fc_unit: int, rnn_unit: int):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()

        self.onset_stack = nn.Sequential(
            ConvStack(N_MELS, cnn_unit, fc_unit),
            BiLSTM(fc_unit, rnn_unit),
            nn.Linear(rnn_unit * 2, N_PIANO_KEYS),
            nn.Sigmoid(),
        )

        self.offset_stack = nn.Sequential(
            ConvStack(N_MELS, cnn_unit, fc_unit),
            BiLSTM(fc_unit, rnn_unit),
            nn.Linear(rnn_unit * 2, N_PIANO_KEYS),
            nn.Sigmoid(),
        )

        self.frame_stack_conv = nn.Sequential(
            ConvStack(N_MELS, cnn_unit, fc_unit),
            nn.Linear(fc_unit, N_PIANO_KEYS),
            nn.Sigmoid(),
        )
        self.frame_stack_rnn = nn.Sequential(
            BiLSTM(N_PIANO_KEYS * 3, rnn_unit),
            nn.Linear(rnn_unit * 2, N_PIANO_KEYS),
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mel = self.melspectrogram(audio)

        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)

        frame_pred_conv_output = self.frame_stack_conv(mel)
        frame_input = torch.cat(
            [onset_pred.detach(), offset_pred.detach(), frame_pred_conv_output, ],
            dim=-1,
        )
        frame_pred = self.frame_stack_rnn(frame_input)

        return frame_pred, onset_pred, offset_pred
