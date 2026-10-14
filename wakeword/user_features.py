"""Turn the user's own recorded wake phrases into training features.

user_features.py WORK_DIR DATA_DIR WAKE_ENDS_JSON   (training venv; foidbot-train-wake runs it)
Each clip is cut just after the name (Whisper word timestamps from foidbot-enroll), sped up and slowed down a little,
then augmented with noise and room echo like the synthetic clips.
"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from openwakeword.data import augment_clips
from openwakeword.utils import compute_features_from_generator

work, data, ends_file = sys.argv[1:4]
ends = json.load(open(ends_file))     # {absolute wav path: seconds where the name ends}
tmp = f"{work}/user_clips"
os.makedirs(tmp, exist_ok=True)
clips = []
for f, end in sorted(ends.items()):
    a, sr = sf.read(f, dtype="float32")
    a = a[: int((end + 0.2) * sr)][-int(1.9 * sr):]
    a = a / (np.abs(a).max() + 1e-6) * 0.5
    for up, down in [(20, 17), (12, 11), (1, 1), (11, 12), (17, 20)]:   # ~0.85x-1.18x speed
        b = resample_poly(a, up, down).astype(np.float32)[-int(1.95 * sr):]
        p = f"{tmp}/{os.path.basename(f)[:-4]}_{up}_{down}.wav"
        sf.write(p, (b * 32767).astype(np.int16), sr)
        clips.append(p)
clips = clips * 24
print(len(clips), "augmented clips")
bg = glob.glob(f"{data}/background/*.wav")
rir = glob.glob(f"{data}/mit_rirs/*.wav")
gen = augment_clips(clips, total_length=32000, batch_size=16, background_clip_paths=bg, RIR_paths=rir)
compute_features_from_generator(gen, n_total=len(clips), clip_duration=32000,
                                output_file=f"{work}/user_features.npy", device="gpu", ncpu=1)
