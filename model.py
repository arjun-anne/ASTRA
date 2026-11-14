"""Pretrained WLASL100 inference, adapted from Abel1712/ASL-Recognition.

Upstream copyright (c) 2026 Jacob, MIT; see THIRD_PARTY_LICENSE.txt.
Architecture and feature ordering match the published training/inference code.
"""
import hashlib
import json
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
CHECKSUMS = {
    "asl_model_best.pth": "fe60246a6be481fbf10a9f5a4abbc82967ebb0da7bfbf5e143c2cd0626381ba8",
    "idx_to_word.json": "bf2ad731ffe25992714148f917f24d078096d1cadcbe0372cc187e5a65367d31",
}
FRAMES = 30
FEATURES = 201


class ASLModel(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.lstm = nn.LSTM(201, 128, num_layers=2, dropout=0.3, batch_first=True)
        self.gru = nn.GRU(128, 64, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(128, num_classes))

    def forward(self, sequence):
        encoded, _ = self.lstm(sequence)
        encoded, _ = self.gru(encoded)
        return self.fc(encoded[:, -1, :])


def keypoints(results):
    """Raw normalized xyz: first 25 pose landmarks, left hand, right hand.

    Missing groups are zero-filled. Do NOT shoulder-center or change landmark order:
    this checkpoint was trained with the raw MediaPipe representation.
    """
    groups = [(results.pose_landmarks, 25), (results.left_hand_landmarks, 21),
              (results.right_hand_landmarks, 21)]
    return np.concatenate([
        np.asarray([[p.x, p.y, p.z] for p in group.landmark[:count]], dtype=np.float32).ravel()
        if group else np.zeros(count * 3, dtype=np.float32)
        for group, count in groups
    ])


def prepare(sequence):
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != FEATURES or not len(arr):
        raise ValueError("Expected a nonempty sequence of 201-feature frames.")
    if not np.isfinite(arr).all():
        raise ValueError("Landmarks contain invalid values.")
    # Matches upstream training: take first 30, zero-pad shorter sequences.
    arr = arr[:FRAMES]
    return np.pad(arr, ((0, FRAMES - len(arr)), (0, 0)))


class Predictor:
    def __init__(self):
        for name, expected in CHECKSUMS.items():
            path = ROOT / "models" / name
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Missing or modified pretrained artifact: {path.name}")
        mapping = json.loads((ROOT / "models/idx_to_word.json").read_text())
        self.labels = [mapping[str(i)] for i in range(100)]
        torch.set_num_threads(1)
        self.model = ASLModel().cpu()
        state = torch.load(ROOT / "models/asl_model_best.pth", map_location="cpu",
                           weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.lock = Lock()

    def predict(self, sequence):
        tensor = torch.from_numpy(prepare(sequence)).unsqueeze(0)
        with self.lock, torch.inference_mode():
            scores = torch.softmax(self.model(tensor), dim=1)[0].numpy()
        order = np.argsort(scores)[::-1][:5]
        return [{"word": self.labels[int(i)], "score": float(scores[i])} for i in order]


def decide(ranked, threshold, margin):
    """Heuristic rejection; softmax is not a calibrated correctness probability."""
    if len(ranked) < 2:
        return "Not recognized"
    if ranked[0]["score"] < threshold:
        return "Not recognized"
    if ranked[0]["score"] - ranked[1]["score"] < margin:
        return "Not recognized"
    return ranked[0]["word"]
