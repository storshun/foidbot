"""Downloads for install.sh and foidbot-train-wake: resumable (curl -C -), skipped when already complete.

python3 setup/fetch.py runtime [--model q4|q5]   models + llama.cpp (~9 GB)
python3 setup/fetch.py train                      wake-word training data (~22 GB, a little less after unpacking)
python3 setup/fetch.py heavy                      optional 27B model for website tasks (17.6 GB; needs a 32 GB GPU)
Standard library only, so it runs before any venv exists.
"""
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
VENDOR = ROOT / "vendor"
TRAIN_DATA = ROOT / "wakeword/data"

LLAMA_BUILD = "b11540"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{b}/{name}-{b}-bin-ubuntu-cuda-{cuda}-x64.tar.gz"
HF = "https://huggingface.co"
LLM = {   # size in GB, for the disk check and progress
    "q5": (f"{HF}/empero-ai/Qwythos-9B-v2-GGUF/resolve/main/Qwythos-9B-v2-MTP-Q5_K_M.gguf", 6.71),
    "q4": (f"{HF}/empero-ai/Qwythos-9B-v2-GGUF/resolve/main/Qwythos-9B-v2-MTP-Q4_K_M.gguf", 5.90),
}
RUNTIME = [
    (f"{HF}/empero-ai/Qwythos-9B-v2-GGUF/resolve/main/mmproj-Qwythos-9B-v2-BF16.gguf", 0.92),
    ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx", 0.33),
    ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", 0.03),
    ("https://github.com/snakers4/silero-vad/raw/v6.2/src/silero_vad/data/silero_vad.onnx", 0.002),
    ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
     "wespeaker_en_voxceleb_resnet34_LM.onnx", 0.03),
]
OWW = f"{HF}/datasets/davidscripka/openwakeword_features/resolve/main"
TRAIN = [
    (f"{OWW}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy", 17.0),
    (f"{OWW}/validation_set_features.npy", 0.18),
    ("https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt", 0.2),
]
HEAVY = (f"{HF}/empero-ai/Qwythos-27B-v1-GGUF/resolve/main/Qwythos-27B-MTP-Q4_K_M.gguf", 17.6)
LIBRI = "https://www.openslr.org/resources/12/dev-clean.tar.gz"   # other voices, for the voiceprint threshold


def say(*a):
    print(*a, flush=True)


def download(url, dest, gb=None):
    dest = Path(dest)
    done = dest.with_suffix(dest.suffix + ".done")
    if done.exists() and dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    say(f"  {dest.name}" + (f" ({gb:.1f} GB)" if gb and gb >= 0.1 else ""))
    for attempt in range(5):
        r = subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "3", "-C", "-", "-o", str(dest),
                            "--progress-bar", url])
        if r.returncode in (0, 33):   # 33: server can't resume a file that is already complete
            done.touch()
            return dest
        say(f"  download interrupted (curl {r.returncode}), retrying…")
    sys.exit(f"Could not download {url}")


def free_gb(path):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def need(path, gb):
    if free_gb(path) < gb:
        sys.exit(f"Not enough disk space in {path}: need about {gb:.0f} GB free, have {free_gb(path):.0f} GB.")


def cuda_version():
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def llama():
    """Official llama.cpp CUDA release build + its CUDA runtime libraries, in vendor/llama/."""
    dest = VENDOR / "llama"
    if (dest / "llama-server").exists():
        return
    cuda = "13.4" if cuda_version() >= (13, 0) else "12.8"
    say(f"llama.cpp {LLAMA_BUILD} (CUDA {cuda})")
    for name in ("llama", "cudart-llama"):
        tgz = download(LLAMA_URL.format(b=LLAMA_BUILD, name=name, cuda=cuda), VENDOR / f"{name}-{cuda}.tar.gz")
        with tarfile.open(tgz) as t:
            t.extractall(VENDOR / "tmp", filter="data")
    dest.mkdir(parents=True, exist_ok=True)
    for f in (VENDOR / "tmp").rglob("*"):
        if f.is_file():
            shutil.move(str(f), dest / f.name)
            (dest / f.name).chmod(0o755)
    shutil.rmtree(VENDOR / "tmp")
    for name in ("llama", "cudart-llama"):
        (VENDOR / f"{name}-{cuda}.tar.gz").unlink(missing_ok=True)
        (VENDOR / f"{name}-{cuda}.tar.gz.done").unlink(missing_ok=True)


def runtime(model):
    url, gb = LLM[model]
    need(MODELS, gb + 2.5)
    say("Models:")
    download(url, MODELS / url.rsplit("/", 1)[1], gb)
    for url, gb in RUNTIME:
        download(url, MODELS / url.rsplit("/", 1)[1], gb)
    llama()


def train():
    need(TRAIN_DATA, 40)
    say("Wake-word training data:")
    for url, gb in TRAIN:
        download(url, TRAIN_DATA / url.rsplit("/", 1)[1], gb)
    libri = TRAIN_DATA / "libri"
    if not (libri / "LibriSpeech/dev-clean").exists():
        tgz = download(LIBRI, TRAIN_DATA / "dev-clean.tar.gz", 0.34)
        with tarfile.open(tgz) as t:
            t.extractall(libri, filter="data")
        tgz.unlink()
        tgz.with_suffix(".gz.done").unlink(missing_ok=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "runtime":
        runtime(sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "q5")
    elif what == "train":
        train()
    elif what == "heavy":
        need(MODELS, HEAVY[1] + 1)
        download(HEAVY[0], MODELS / HEAVY[0].rsplit("/", 1)[1], HEAVY[1])
    else:
        sys.exit(__doc__)
