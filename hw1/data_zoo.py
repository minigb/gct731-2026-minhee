"""Dataset utilities for HW1 piano transcription."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pretty_midi
import soundfile
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from config import HOP_SIZE, MAX_MIDI, MIN_MIDI, SAMPLE_RATE


def allocate_batch(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    """Moves tensor values in batch dict to a target device."""
    for key in batch.keys():
        if key != "path":
            batch[key] = batch[key].to(device)
    return batch


class MAESTRO_small(Dataset):
    def __init__(
        self,
        path: str = "gct731-maestro",
        groups: List[str] | None = None,
        sequence_length: int | None = SAMPLE_RATE * 5,
        hop_size: int = HOP_SIZE,
        seed: int = 42,
        random_sample: bool = True,
    ) -> None:
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        assert all(group in self.available_groups() for group in self.groups)

        self.sample_length = ((sequence_length // hop_size) * hop_size) if sequence_length is not None else None
        self.random = np.random.RandomState(seed)
        self.random_sample = random_sample
        self.hop_size = hop_size
        self.data: List[Dict[str, Tensor]] = []

        print(f"Loading {len(self.groups)} group(s) of {self.__class__.__name__} at {path}")
        for group in self.groups:
            file_list = self.get_file_path_list_of_group(group)
            for input_files in tqdm(file_list, desc=f"Loading group {group}"):
                self.data.append(self.load(*input_files))

    @classmethod
    def available_groups(cls) -> List[str]:
        return ["train", "validation", "test", "debug"]

    def get_file_path_list_of_group(self, group: str) -> List[tuple]:
        metadata_path = os.path.join(self.path, "data.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        subset_name = "train" if group == "debug" else group
        files = sorted(
            [
                (
                    os.path.join(self.path, row["audio_filename"].replace(".wav", ".flac")),
                    os.path.join(self.path, row["midi_filename"]),
                )
                for row in metadata
                if row["split"] == subset_name
            ]
        )

        if group == "debug":
            return files[:10]

        return [
            (audio if os.path.exists(audio) else audio.replace(".flac", ".wav"), midi)
            for audio, midi in files
        ]

    def load(self, audio_path: str, midi_path: str) -> Dict[str, Tensor]:
        """Loads an audio track and the corresponding labels."""
        audio, sr = soundfile.read(audio_path, dtype="int16")
        assert sr == SAMPLE_RATE

        if audio.ndim == 2:
            audio = audio.mean(axis=1).astype(np.int16)

        frames_per_sec = sr / self.hop_size

        audio_tensor = torch.ShortTensor(audio)
        audio_length = len(audio_tensor)

        midi = pretty_midi.PrettyMIDI(midi_path)
        midi_length_sec = midi.get_end_time()
        frame_length = min(int(midi_length_sec * frames_per_sec), (audio_length // self.hop_size) + 1)

        audio_tensor = audio_tensor[: frame_length * self.hop_size]

        frame = midi.get_piano_roll(fs=frames_per_sec)
        onset = np.zeros_like(frame)
        for inst in midi.instruments:
            for note in inst.notes:
                onset[note.pitch, int(note.start * frames_per_sec)] = 1

        data = {
            "path": audio_path,
            "audio": audio_tensor,
            "frame": torch.from_numpy(frame[MIN_MIDI : MAX_MIDI + 1].T),
            "onset": torch.from_numpy(onset[MIN_MIDI : MAX_MIDI + 1].T),
        }
        return data

    def _pad_audio(self, audio: Tensor, target_length: int) -> Tensor:
        if audio.shape[0] >= target_length:
            return audio
        padded = torch.zeros(target_length, dtype=audio.dtype)
        padded[: audio.shape[0]] = audio
        return padded

    def _pad_roll(self, roll: Tensor, target_steps: int) -> Tensor:
        if roll.shape[0] >= target_steps:
            return roll
        padded = torch.zeros((target_steps, roll.shape[1]), dtype=roll.dtype)
        padded[: roll.shape[0], :] = roll
        return padded

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        data = self.data[index]

        audio = data["audio"]
        frames = data["frame"] >= 1
        onsets = data["onset"] >= 1
        frame_len = frames.shape[0]

        if self.sample_length is not None:
            n_steps = self.sample_length // self.hop_size
            max_step_begin = max(frame_len - n_steps, 0)
            if self.random_sample and max_step_begin > 0:
                step_begin = self.random.randint(max_step_begin + 1)
            else:
                step_begin = 0
            step_end = min(step_begin + n_steps, frame_len)

            sample_begin = step_begin * self.hop_size
            sample_end = sample_begin + self.sample_length

            audio_seg = audio[sample_begin:sample_end]
            frame_seg = frames[step_begin:step_end]
            onset_seg = onsets[step_begin:step_end]

            audio_seg = self._pad_audio(audio_seg, self.sample_length)
            frame_seg = self._pad_roll(frame_seg, n_steps)
            onset_seg = self._pad_roll(onset_seg, n_steps)
        else:
            audio_seg = audio
            frame_seg = frames
            onset_seg = onsets

        result = {"path": data["path"]}
        result["audio"] = audio_seg.float().div_(32768.0)
        result["frame"] = frame_seg.float()
        result["onset"] = onset_seg.float()
        return result

    def __len__(self) -> int:
        return len(self.data)
