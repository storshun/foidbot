# foidbot

A fully local voice assistant that lives in the corner of your Linux desktop: an animated companion you name
yourself, who wakes to **"Hey &lt;her name&gt;"**, talks back with lip sync and twenty-odd facial expressions,
and actually does things: weather, news, stocks, business hours, timers, notes, media, your clipboard, your
screen, and websites in her own browser. No cloud, no account, no subscription. Her brain, ears and voice all
run on your GPU.

![Her expressions](assets/preview.jpg)

During setup you **choose her name**. Onboarding checks that her voice pronounces it right, teaches her speech
recognition how it gets spelled, and then **trains a wake-word model for that exact name** on your machine,
including recordings of your own voice.

## What she does

- **Talks and listens.** Wake word, a click, or a hotkey. Whisper hears you, Kokoro speaks, and an echo-cancelled
  mic lets you talk over her. After she asks you something, she keeps listening without the wake word.
- **Knows things that change.** Weather and severe-weather watch (US National Weather Service, Open-Meteo
  elsewhere), news, stocks, business hours, encyclopedia lookups and web search through a private SearXNG, with
  sources cross-checked and linked in her speech bubble.
- **Runs your desktop.** Opens apps and links, controls media and volume, sets timers and reminders, saves and
  searches Markdown notes (an Obsidian vault works), reads your clipboard, and looks at your screen when you ask
  about "this".
- **Uses websites for you** in her own Chromium profile, step by step. Anything that buys, sends, posts or deletes
  waits for your spoken "yes", and she never types passwords.
- **Recognizes your voice.** Strangers can't talk over her, and when she hears someone else (or a call or screen
  share is running) she switches to "company mode" and keeps it professional.
- **Has a personality you set,** plus an optional private side that only shows when it's just you.
- **Claude Code mode** (optional): relay your voice into a running Claude Code session and hear its replies.

## Requirements

| | |
|---|---|
| OS | 64-bit Linux with **PipeWire** and **systemd**. KDE Plasma 6 is the best fit (the widget pins itself to a corner). GNOME and others work, but the desktop decides where she sits. |
| GPU | **NVIDIA, 12 GB VRAM minimum**, 16 GB+ recommended (the 12 GB tier gets a smaller model quantization and a shorter context). Recent proprietary driver. |
| Disk | ~15 GB for the app and models. Wake-word training needs ~40 GB free while it runs and keeps ~22 GB of training data afterwards (deletable). |
| Tools | `git curl tar ffmpeg`, PipeWire utilities (`pw-record`, `pw-cli`), and optionally `podman` or `docker` for web search. The installer installs [uv](https://docs.astral.sh/uv/) if you don't have it. |
| Mic | Any. A headset or desk mic helps. |

Speech recognition and wake words are English-only for now.

## Install

```bash
git clone https://github.com/storshun/foidbot
cd foidbot
./install.sh
```

The installer checks your system, builds the Python environment, downloads the models (~9 GB; the model size is
picked from your VRAM) and an official llama.cpp CUDA build, registers the systemd user services, and then starts
**onboarding**. Nothing is installed system-wide: everything lives in the cloned folder, plus systemd user units,
an autostart entry and command links in `~/.local/bin`.

### Onboarding (`foidbot-onboard`)

About 15 minutes of your time:

1. **You**: your name, pronouns and city (for weather and "near me").
2. **Her voice**: listen to a few Kokoro voices and pick one.
3. **Her name**: she says "Hey &lt;name&gt;" out loud. If she gets it wrong, type it the way it sounds
   ("Shah-la", "Zef-er-een") until she gets it right. That spelling and its phonemes drive all the training.
4. **Personality**: keep the default or describe her in your own words; same for her private side, or turn it off.
5. **Notes folder**: finds your Obsidian vault if you have one.
6. **Her ears**: Whisper listens to several voices saying her name, so she also wakes on the ways it gets
   spelled. Then her own brain lists everyday words that *sound like* her name, which the wake-word model will
   learn to ignore. You can edit both lists.
7. **Your voice**: read 20 short sentences and say her wake phrase 24 different ways (~6 min). This builds your
   voiceprint and gives the wake-word training real examples of you.
8. **Her wake word**: training starts in the background (details below). She works right away: until the
   model is done, she wakes when Whisper hears her name.

Run `foidbot-onboard` again any time to change answers; your current settings are the defaults.
To rename her: `foidbot-onboard --rename` (then she retrains).

## Wake-word training

`foidbot-train-wake` builds a small [openWakeWord](https://github.com/dscripka/openWakeWord) model for "Hey
&lt;name&gt;" and "Okay &lt;name&gt;":

1. Downloads the training data once (~22 GB): openWakeWord's negative feature sets, AudioSet background noise,
   MIT room impulse responses and the LibriTTS-R Piper voice model.
2. Synthesizes ~50,000 clips of her name in 900+ Piper voices, plus 6,000 in blended Kokoro voices, and as many
   sound-alike phrases (yours plus ones generated from her phonemes).
3. Adds room echo and background noise, folds in your own recorded wake phrases (cut where the name ends, sped up
   and slowed down), and trains.
4. Installs `models/wake.onnx`, measures it on your recordings, picks a threshold and restarts her.

About **1–2 hours** on a recent NVIDIA GPU (the reference model took ~1 h on an RTX 5090; `--quick` is about
half). On GPUs under 16 GB her brain pauses during the heavy steps. The reference model catches 100% of its
owner's wake phrases at ~0.09 false wakes per hour of everyday audio.

```bash
foidbot-train-wake            # run or resume
foidbot-train-wake --status   # progress (log: wakeword/work/train.log)
foidbot-train-wake --quick    # faster, slightly less accurate
foidbot-train-wake --restart  # start over
```

## Using her

- **Wake her** by saying "Hey &lt;name&gt;" or "Okay &lt;name&gt;", left-clicking her, or a hotkey: bind
  `foidbot-ctl listen` to a shortcut in your desktop's keyboard settings (e.g. Meta+Space).
- **Type** instead: double-click her. Right-click for the menu (stop talking, company mode, ignore wake word,
  Claude Code).
- `foidbot-ctl listen | stop | say <text> | mode auto|private|company | mute on|off`
- Settings live in **`config/foidbot.toml`** (commented). Restart after editing:
  `systemctl --user restart foidbot`.
- Everything she heard, said and did is logged to `data/transcripts/<date>.log` (kept 30 days; configurable).

### Services

| Unit | What |
|---|---|
| `foidbot` | the daemon: ears, voice, tools (websocket on port 8770) |
| `foidbot-llm` | llama.cpp serving her brain on port 8091 |
| `foidbot-aec` | echo-cancelled microphone (a separate PipeWire client; your default mic is untouched) |
| `foidbot-searxng` | private web search on port 8888 (podman or docker) |
| `foidbot-llm-heavy` | optional 27B model for website tasks (`./install.sh --heavy`, 32 GB GPUs), started on demand |

`journalctl --user -u foidbot -f` shows what she's doing.

## Her avatar

The widget is a layered 2D avatar. Each mood is a full expression image, and while she talks a lower-face
overlay (three mouth sizes cut from her open-mouth expressions) follows the audio. Blinks are an eye overlay, and
expressions crossfade. Which expression goes with which mood and state is set in
[`assets/avatar/avatar.toml`](assets/avatar/avatar.toml).

To use your own character, drop a matching expression set (same framing in every image, transparent PNG) into
`assets/avatar/src/` and run `foidbot-build-avatar`. See [`assets/avatar/SPRITES.md`](assets/avatar/SPRITES.md).
With no art at all she falls back to an animated orb.

## Privacy

Her brain, speech recognition, voice, wake word and voiceprint all run locally. She reaches the internet only
for what you ask her to look up: weather (NWS / Open-Meteo), place lookups (OpenStreetMap Nominatim), news feeds,
stock quotes, web pages, and searches through your own SearXNG instance. Your recordings, voiceprint and
transcripts stay in `data/`.

## Troubleshooting

- **She doesn't hear me**: `journalctl --user -u foidbot -f` shows `heard: …` lines. Check that the
  "foidbot echo-cancelled mic" source exists (`pw-cli ls Node | grep foidbot`); set `ears.device = ""` to use the
  default mic.
- **Out of GPU memory**: `./install.sh --model q4`, or close other GPU apps. Whisper plus a 9B model need ~12 GB.
- **Wakes too often / not enough**: rerun `foidbot-enroll --wake` and `foidbot-train-wake`, or adjust
  `ears.oww_threshold` (higher = stricter).
- **No web search**: install podman (or docker) and rerun `./install.sh`.

## Uninstall

```bash
./uninstall.sh         # stops and removes services, autostart and commands; keeps this folder
./uninstall.sh --all   # also deletes models, environments, training data and your data
```

## Credits

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp), [Qwythos-9B-v2](https://huggingface.co/empero-ai/Qwythos-9B-v2)
(Apache-2.0), [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx),
[openWakeWord](https://github.com/dscripka/openWakeWord),
[piper-sample-generator](https://github.com/rhasspy/piper-sample-generator),
[Silero VAD](https://github.com/snakers4/silero-vad), [WeSpeaker](https://github.com/wenet-e2e/wespeaker) via
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [SearXNG](https://github.com/searxng/searxng),
[Playwright](https://playwright.dev) and [PySide6](https://doc.qt.io/qtforpython/).

foidbot's own code is MIT-licensed (see [LICENSE](LICENSE)). The models and datasets it downloads keep their own
licenses, and some of the wake-word training data is licensed for non-commercial use only, so treat the wake-word
models you train as personal-use. Check each project's terms before using foidbot commercially.
