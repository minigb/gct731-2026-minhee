"""Evaluation utilities for HW1 piano transcription."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from mido import Message, MidiFile, MidiTrack
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import hz_to_midi, midi_to_hz
from tqdm import tqdm

from config import FRAME_THRESHOLD, HOP_SIZE, MIN_MIDI, ONSET_THRESHOLD, SAMPLE_RATE
from data_zoo import allocate_batch


def framewise_eval(
    pred: torch.Tensor,
    label: torch.Tensor,
    threshold: float = FRAME_THRESHOLD,
) -> Tuple[float, float, float]:
    """Point-wise frame evaluation for piano-roll predictions."""
    tp = torch.sum((pred >= threshold) * (label == 1)).cpu().numpy()
    fn = torch.sum((pred < threshold) * (label == 1)).cpu().numpy()
    fp = torch.sum((pred >= threshold) * (label != 1)).cpu().numpy()

    pr = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    re = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * re / float(pr + re) if (pr + re) > 0 else 0.0
    return float(pr), float(re), float(f1)


def extract_notes(
    onsets: torch.Tensor,
    frames: torch.Tensor,
    onset_threshold: float = ONSET_THRESHOLD,
    frame_threshold: float = FRAME_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts note pitches and intervals from onset/frame predictions."""
    onsets = (onsets > onset_threshold).type(torch.int).cpu()
    frames = (frames > frame_threshold).type(torch.int).cpu()
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    pitches: List[int] = []
    intervals: List[List[int]] = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()
        onset = frame
        offset = frame

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            offset += 1
            if offset == onsets.shape[0]:
                break
            if offset != onset and onsets[offset, pitch].item():
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])

    return np.array(pitches), np.array(intervals)


def _safe_note_eval(
    i_ref: np.ndarray,
    p_ref: np.ndarray,
    i_est: np.ndarray,
    p_est: np.ndarray,
    with_offsets: bool,
) -> Tuple[float, float, float, float]:
    try:
        if with_offsets:
            p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        else:
            p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    except ValueError:
        return 0.0, 0.0, 0.0, 0.0

    values = np.nan_to_num(np.array([p, r, f, o], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def evaluate_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, List[float]]:
    metrics: Dict[str, List[float]] = defaultdict(list)
    batch = allocate_batch(batch, device)

    frame_pred, onset_pred = model(batch["audio"])
    criterion = torch.nn.BCELoss()
    frame_loss = criterion(frame_pred, batch["frame"])
    onset_loss = criterion(onset_pred, batch["onset"])
    metrics["metric/loss/frame_loss"].append(float(frame_loss.detach().cpu().item()))
    metrics["metric/loss/onset_loss"].append(float(onset_loss.detach().cpu().item()))

    for batch_idx in range(batch["audio"].shape[0]):
        pr, re, f1 = framewise_eval(frame_pred[batch_idx], batch["frame"][batch_idx])
        metrics["metric/frame/frame_precision"].append(pr)
        metrics["metric/frame/frame_recall"].append(re)
        metrics["metric/frame/frame_f1"].append(f1)

        pr, re, f1 = framewise_eval(onset_pred[batch_idx], batch["onset"][batch_idx])
        metrics["metric/frame/onset_precision"].append(pr)
        metrics["metric/frame/onset_recall"].append(re)
        metrics["metric/frame/onset_f1"].append(f1)

        p_est, i_est = extract_notes(onset_pred[batch_idx], frame_pred[batch_idx])
        p_ref, i_ref = extract_notes(batch["onset"][batch_idx], batch["frame"][batch_idx])

        scaling = HOP_SIZE / SAMPLE_RATE
        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(MIN_MIDI + pitch) for pitch in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + pitch) for pitch in p_est])

        p, r, f, o = _safe_note_eval(i_ref, p_ref, i_est, p_est, with_offsets=False)
        metrics["metric/note/precision"].append(p)
        metrics["metric/note/recall"].append(r)
        metrics["metric/note/f1"].append(f)
        metrics["metric/note/overlap"].append(o)

        p, r, f, o = _safe_note_eval(i_ref, p_ref, i_est, p_est, with_offsets=True)
        metrics["metric/note-with-offsets/precision"].append(p)
        metrics["metric/note-with-offsets/recall"].append(r)
        metrics["metric/note-with-offsets/f1"].append(f)
        metrics["metric/note-with-offsets/overlap"].append(o)

    return metrics


def aggregate_metrics(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        key: float(np.mean(values)) if len(values) > 0 else 0.0
        for key, values in metrics.items()
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    desc: str = "Eval",
) -> Dict[str, float]:
    """Evaluates a model over a dataloader and returns averaged metrics."""
    was_training = model.training
    model.eval()
    metrics = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            batch_results = evaluate_batch(model, batch, device)
            for key, value in batch_results.items():
                metrics[key].extend(value)

    if was_training:
        model.train()

    return aggregate_metrics(metrics)


def print_f1_metrics(metrics: Dict[str, float]) -> None:
    for key in sorted(metrics.keys()):
        if key.endswith("/f1"):
            print(f"{key:32}: {metrics[key]:.4f}")


def save_midi(path: str, pitches: np.ndarray, intervals: np.ndarray, velocities: List[int]) -> None:
    """Saves extracted notes as a MIDI file."""
    file = MidiFile()
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(
            {
                "type": "on",
                "pitch": pitches[i],
                "time": intervals[i][0],
                "velocity": velocities[i],
            }
        )
        events.append(
            {
                "type": "off",
                "pitch": pitches[i],
                "time": intervals[i][1],
                "velocity": velocities[i],
            }
        )
    events.sort(key=lambda row: row["time"])

    last_tick = 0
    for event in events:
        current_tick = int(event["time"] * ticks_per_second)
        velocity = min(int(event["velocity"] * 127), 127)
        pitch = int(round(hz_to_midi(event["pitch"])))
        track.append(
            Message(
                "note_" + event["type"],
                note=pitch,
                velocity=velocity,
                time=current_tick - last_tick,
            )
        )
        last_tick = current_tick

    file.save(path)
