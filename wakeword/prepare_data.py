"""Room echoes and background noise for wake-word augmentation (training venv; needs ffmpeg).

prepare_data.py DATA_DIR
- MIT environmental impulse responses (16 kHz)          -> DATA_DIR/mit_rirs/*.wav
- AudioSet balanced-train shards 00-05 (~3,000 clips)    -> DATA_DIR/background/*.wav (16 kHz mono)
"""
import glob
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

data = Path(sys.argv[1])
rirs, bg = data / "mit_rirs", data / "background"
if not (rirs / ".done").exists():
    raw = data / "mit_rirs_raw"
    snapshot_download("davidscripka/MIT_environmental_impulse_responses", repo_type="dataset", local_dir=raw,
                      allow_patterns=["16khz/*"])
    rirs.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(f"{raw}/16khz/*.wav"):
        shutil.copy(f, rirs)
    shutil.rmtree(raw)
    (rirs / ".done").touch()
    print("room echoes:", len(list(rirs.glob("*.wav"))))
if not (bg / ".done").exists():
    raw = data / "audioset_raw"
    bg.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in range(6):
        shard = hf_hub_download("agkphysics/AudioSet", f"data/bal_train/{i:02d}.parquet", repo_type="dataset",
                                local_dir=raw)
        for r in pq.read_table(shard, columns=["video_id", "audio"]).to_pylist():
            b = r["audio"]["bytes"]
            if b:
                p = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
                                    str(bg / f"{r['video_id']}.wav")], input=b)
                n += p.returncode == 0
        Path(shard).unlink()    # ~650 MB each; keep the disk peak low
        print(f"background shard {i + 1}/6: {n} clips", flush=True)
    shutil.rmtree(raw, ignore_errors=True)
    (bg / ".done").touch()
