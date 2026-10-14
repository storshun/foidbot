#!/usr/bin/env bash
# Wake-word training environment: pinned openWakeWord + piper-sample-generator checkouts (patched) and a
# Python 3.11 env with PyTorch (~8 GB). Run by `install.sh --train` and by foidbot-onboard. Safe to re-run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OWW_REPO=https://github.com/dscripka/openWakeWord.git
OWW_COMMIT=368c03716d1e92591906a84949bc477f3a834455
PIPER_REPO=https://github.com/dscripka/piper-sample-generator.git
PIPER_COMMIT=f1988a4d54eddb23d99e86f0adfef6226a85acc7
command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"

printf '\n\033[1m%s\033[0m\n' "Wake-word training environment (wakeword/.venv, ~8 GB with PyTorch)"
[[ -d wakeword/openWakeWord ]] || git clone -q "$OWW_REPO" wakeword/openWakeWord
git -C wakeword/openWakeWord checkout -q "$OWW_COMMIT"
[[ -d wakeword/piper-sample-generator ]] || git clone -q "$PIPER_REPO" wakeword/piper-sample-generator
git -C wakeword/piper-sample-generator checkout -q "$PIPER_COMMIT"
python3 wakeword/patch_upstream.py
[[ -x wakeword/.venv/bin/python ]] || uv venv -q -p 3.11 wakeword/.venv
uv pip install -q --python wakeword/.venv/bin/python torch==2.7.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python wakeword/.venv/bin/python -r wakeword/requirements-train.txt
uv pip install -q --python wakeword/.venv/bin/python --no-deps -e wakeword/openWakeWord
wakeword/.venv/bin/python -c "from openwakeword.utils import download_models; download_models(['hey_jarvis_v0.1'])" >/dev/null 2>&1
echo "ok"
