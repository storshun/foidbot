"""Augment the Kokoro positives exactly like train.py does and save their features (2 rounds of augmentation).

kokoro_features.py WORK_DIR DATA_DIR   (training venv)
"""
import glob
import sys

from openwakeword.data import augment_clips
from openwakeword.utils import compute_features_from_generator

work, data = sys.argv[1], sys.argv[2]
clips = sorted(glob.glob(f"{work}/kokoro_pos/*.wav")) * 2
bg = glob.glob(f"{data}/background/*.wav")
rir = glob.glob(f"{data}/mit_rirs/*.wav")
gen = augment_clips(clips, total_length=32000, batch_size=16, background_clip_paths=bg, RIR_paths=rir)
compute_features_from_generator(gen, n_total=len(clips), clip_duration=32000, output_file=f"{work}/kokoro_features.npy",
                                device="gpu", ncpu=1)
