"""Who is talking: compares an utterance with the user's voiceprint (WeSpeaker ResNet34 embeddings, CPU, ~20 ms).

The voiceprint and its threshold are made by bin/foidbot-enroll from the user's recordings (data/voice/user/*.wav)
and stored in data/voiceprint.json. Without one, every answer is None (unknown) and nothing changes.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/wespeaker_en_voxceleb_resnet34_LM.onnx"
PRINT = ROOT / "data/voiceprint.json"
MIN_S = 1.0   # shorter clips are too unreliable to judge


class Speakers:
    def __init__(self, threads=2):
        import sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL), num_threads=threads)
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.load()

    def load(self):
        self.ref = self.mean = self.accept = self.reject = None
        if PRINT.exists():
            d = json.loads(PRINT.read_text())
            self.ref = np.array(d["embedding"], np.float32)
            self.mean = np.array(d["mean"], np.float32)        # crowd average, removed before comparing
            self.accept, self.reject = float(d["accept"]), float(d["reject"])

    def embed(self, audio, sr=16000, center=True):
        s = self.ex.create_stream()
        s.accept_waveform(sr, np.asarray(audio, np.float32))
        s.input_finished()
        v = np.array(self.ex.compute(s), np.float32)
        v /= np.linalg.norm(v) + 1e-9
        if center and self.mean is not None:
            v = v - self.mean
            v /= np.linalg.norm(v) + 1e-9
        return v

    def score(self, audio, sr=16000):
        if self.ref is None or len(audio) < MIN_S * sr:
            return None
        return float(self.embed(audio, sr) @ self.ref)

    def is_user(self, audio, sr=16000):
        """True (clearly the user) / False (clearly someone else) / None (no voiceprint, too short, or unsure)."""
        s = self.score(audio, sr)
        if s is None:
            return None
        return True if s >= self.accept else False if s < self.reject else None
