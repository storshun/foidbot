#!/usr/bin/env bash
# foidbot installer. Run from the cloned repo:   ./install.sh
#
#   ./install.sh                 runtime: Python env, models (~9 GB), llama.cpp, services, then onboarding
#   ./install.sh --train         also the wake-word training environment (onboarding offers this anyway)
#   ./install.sh --heavy         also the optional 27B model for website tasks (17.6 GB; 32 GB GPUs)
#   ./install.sh --model q4|q5   pick the 9B model size yourself (default: q5 with 16 GB+ VRAM, else q4)
#   ./install.sh --no-onboard    don't start onboarding at the end
#
# Safe to run again: finished downloads and environments are skipped. Nothing is installed system-wide;
# everything lives in this folder plus systemd user units, an autostart entry and links in ~/.local/bin.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
TRAIN=0 HEAVY=0 ONBOARD=1 MODEL=""
while (($#)); do
  case $1 in
    --train) TRAIN=1 ;;
    --heavy) HEAVY=1 ;;
    --no-onboard) ONBOARD=0 ;;
    --model) MODEL=${2:-}; shift ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*"; }
die() { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

pkg_hint() {   # distro-specific install command for missing tools
  local pkgs="$*"
  if have dnf; then echo "sudo dnf install $pkgs"
  elif have apt-get; then echo "sudo apt install ${pkgs//pipewire-utils/pipewire-bin}"
  elif have pacman; then echo "sudo pacman -S ${pkgs//pipewire-utils/pipewire}"
  else echo "install: $pkgs"; fi
}

# ---------------------------------------------------------------- 1. system checks
bold "Checking this machine"
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die "foidbot needs 64-bit Linux (x86_64)."
have nvidia-smi || die "No NVIDIA driver found (nvidia-smi). foidbot needs an NVIDIA GPU with 12 GB+ of VRAM."
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -1)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU, $((VRAM_MB / 1024)) GB"
if ((VRAM_MB < 11500)); then
  die "$GPU has $((VRAM_MB / 1024)) GB of VRAM; foidbot needs 12 GB or more (brain + speech recognition)."
fi
have systemctl && systemctl --user show-environment >/dev/null 2>&1 || die "systemd user services are required."

missing=()
for tool in curl git tar ffmpeg; do have "$tool" || missing+=("$tool"); done
have pw-record && have pw-cli || missing+=(pipewire-utils)
if ((${#missing[@]})); then
  die "Missing: ${missing[*]}.  Try:  $(pkg_hint "${missing[@]}")"
fi
pactl info 2>/dev/null | grep -q PipeWire || warn "PipeWire doesn't seem to be your sound server; the microphone may not work."

if ! have uv; then
  echo "foidbot uses uv (https://docs.astral.sh/uv/) to build its Python environments."
  read -r -p "Install uv for your user now with its official installer? [Y/n] " a
  [[ ${a,,} == n* ]] && die "Install uv, then run ./install.sh again."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  have uv || die "uv didn't end up on PATH; open a new terminal and run ./install.sh again."
fi

CONTAINER=""
if have podman; then CONTAINER=podman; elif have docker; then CONTAINER=docker; fi
[[ -n $CONTAINER ]] || warn "Neither podman nor docker found: web search (SearXNG) will be off. $(pkg_hint podman)"
have kwriteconfig6 || warn "Not KDE Plasma 6: the widget works, but your desktop decides where it sits."

# ---------------------------------------------------------------- 2. Python environment
bold "Python environment (.venv)"
[[ -x .venv/bin/python ]] || uv venv -q -p 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
uv pip install -q --python .venv/bin/python --no-deps openwakeword==0.6.0
.venv/bin/python -m playwright install chromium >/dev/null
.venv/bin/python -c "from openwakeword.utils import download_models; download_models(['hey_jarvis_v0.1'])" >/dev/null 2>&1
echo "ok"

# ---------------------------------------------------------------- 3. models + llama.cpp
if [[ -z $MODEL ]]; then
  if ((VRAM_MB >= 15500)); then MODEL=q5; else MODEL=q4; fi
fi
case $MODEL in
  q5) MODEL_FILE=Qwythos-9B-v2-MTP-Q5_K_M.gguf ;;
  q4) MODEL_FILE=Qwythos-9B-v2-MTP-Q4_K_M.gguf ;;
  *) die "--model must be q4 or q5" ;;
esac
if ((VRAM_MB >= 15500)); then CTX=32768; else CTX=16384; fi
bold "Downloading models ($MODEL_FILE, context $CTX)"
python3 setup/fetch.py runtime --model "$MODEL"
((HEAVY)) && python3 setup/fetch.py heavy

# ---------------------------------------------------------------- 4. avatar, search settings
bold "Avatar"
if [[ ! -f assets/avatar/frames/blink.png ]]; then bin/foidbot-build-avatar; else echo "already built"; fi

mkdir -p data/searxng
if [[ ! -f data/searxng/settings.yml ]]; then
  sed "s/@SECRET@/$(python3 -c 'import secrets; print(secrets.token_hex(24))')/" config/searxng/settings.yml \
    > data/searxng/settings.yml
fi

# ---------------------------------------------------------------- 5. services, autostart, commands
bold "Services"
UNITS="$HOME/.config/systemd/user"
mkdir -p "$UNITS" "$HOME/.config/autostart" "$HOME/.local/share/applications" "$HOME/.local/bin"
NVLIBS=$(.venv/bin/python -c "import nvidia, os; p = list(nvidia.__path__)[0]
print(':'.join(os.path.join(p, x, 'lib') for x in ('cublas', 'cudnn')))")
render() {
  sed -e "s#@ROOT@#$ROOT#g" -e "s#@MODEL@#$MODEL_FILE#g" -e "s#@CTX@#$CTX#g" -e "s#@NVLIBS@#$NVLIBS#g" "$1" > "$2"
}
for u in foidbot foidbot-llm foidbot-aec; do render "systemd/$u.service.in" "$UNITS/$u.service"; done
[[ -f models/Qwythos-27B-MTP-Q4_K_M.gguf ]] && render systemd/foidbot-llm-heavy.service.in "$UNITS/foidbot-llm-heavy.service"
if [[ $CONTAINER == podman ]]; then
  mkdir -p "$HOME/.config/containers/systemd"
  render systemd/foidbot-searxng.container.in "$HOME/.config/containers/systemd/foidbot-searxng.container"
elif [[ $CONTAINER == docker ]]; then
  render systemd/foidbot-searxng-docker.service.in "$UNITS/foidbot-searxng.service"
fi
render systemd/foidbot.desktop.in "$HOME/.config/autostart/foidbot.desktop"
cp "$HOME/.config/autostart/foidbot.desktop" "$HOME/.local/share/applications/foidbot.desktop"
for b in foidbot-ctl foidbot-onboard foidbot-enroll foidbot-train-wake foidbot-build-avatar; do
  ln -sf "$ROOT/bin/$b" "$HOME/.local/bin/$b"
done
systemctl --user daemon-reload
echo "ok (commands linked into ~/.local/bin)"

# ---------------------------------------------------------------- 6. wake-word training environment
((TRAIN)) && setup/install-train.sh

bold "Installed."
if ((ONBOARD)); then
  exec bin/foidbot-onboard
else
  echo "Next: foidbot-onboard   (sets her name, your details and voice, then starts her)"
fi
