"""Extra positive clips from Kokoro (a different TTS family than Piper), with blended voices for variety.

gen_kokoro.py OUT_DIR SEED COUNT "Hey Name" "Okay Name"   (runtime venv; foidbot-train-wake runs it)
"""
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

ROOT = Path(__file__).resolve().parent.parent
k = Kokoro(str(ROOT / "models/kokoro-v1.0.onnx"), str(ROOT / "models/voices-v1.0.bin"))
voices = [v for v in k.get_voices() if v[:2] in ("af", "am", "bf", "bm")]
out, seed, count, phrases = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(seed)
# punctuation changes Kokoro's prosody: "Hey Name." / "Hey, Name." / "Hey Name!"
variants = [p + end for p in phrases for end in (".", "!")] + [p.replace(" ", ", ", 1) + "." for p in phrases]
for i in range(count):
    a, b = rng.sample(voices, 2)
    w = rng.uniform(0, 1)
    style = w * k.get_voice_style(a) + (1 - w) * k.get_voice_style(b)
    audio, sr = k.create(rng.choice(variants), voice=style, speed=rng.uniform(0.8, 1.25))
    idx = (np.arange(int(len(audio) * 16000 / sr)) * sr / 16000).astype(int)
    sf.write(out / f"k{seed}_{i:05d}.wav", audio[idx], 16000)
