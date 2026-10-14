"""foidbot daemon: ears (VAD + Whisper), brain (local LLM + tools), voice (Kokoro), and a websocket for the widget.

ws://127.0.0.1:<daemon.port, default 8770>
  widget -> daemon: {"t": "listen"} | {"t": "stop"} | {"t": "text", "text": ...} | {"t": "mode", "setting": auto|company|private}
                    {"t": "mute", "on": bool}
  daemon -> widget: state, mood, user, reply_start, reply, reply_end, viseme, notice, mode
"""
import asyncio
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import websockets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from config import load, path as cfg_path, pronouns  # noqa: E402
from persona import PRIVATE_MOODS, PUBLIC_MOODS, system_prompt  # noqa: E402
from browser import YES as YES_RE, Browser  # noqa: E402
from browser_agent import BrowserAgent  # noqa: E402
from heavy import Heavy  # noqa: E402
from desktop import Timers  # noqa: E402
from claude_code import ClaudeCode, match_session, speechify  # noqa: E402
from tools import Tools, WeatherWatch, http  # noqa: E402

CFG = load()
ME, USER, P = CFG["assistant"]["name"], CFG["user"]["name"], pronouns(CFG)
SR = 16000
BLOCK = 512  # 32 ms, what Silero expects at 16 kHz


TRANSCRIPTS = ROOT / "data/transcripts"


def log(*a):
    now = datetime.now()
    line = " ".join(str(x) for x in a)
    print(now.strftime("%H:%M:%S"), line, flush=True)
    try:   # kept for troubleshooting old conversations; pruned after [log] transcript_days
        with open(TRANSCRIPTS / f"{now:%Y-%m-%d}.log", "a") as f:
            f.write(f"{now:%H:%M:%S} {line}\n")
    except OSError:
        pass


def prune_transcripts():
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    keep = CFG.get("log", {}).get("transcript_days", 30)
    cutoff = f"{datetime.fromtimestamp(time.time() - keep * 86400):%Y-%m-%d}"
    for f in TRANSCRIPTS.glob("*.log"):
        if f.stem < cutoff:
            f.unlink()


# Whisper's usual spellings of "Claude" (cloud, clod, clawed…) when it doesn't know the name
CLAUDE_RE = re.compile(r"\b(?:cloud|clod|clawed|claud|clawd|klaud|clyde|clode|claw|cloudy)(?=[ -]?(?:code|coat|cold|codes)\b)"
                       r"|(?<=anthropic's )(?:cloud|clod|clawed|claud|clawd|clyde)\b"
                       r"|(?<=anthropic )(?:cloud|clod|clawed|claud|clawd|clyde)\b", re.I)
CODE_AFTER_RE = re.compile(r"\bClaude[ -]?(?:coat|cold|codes)\b")


def fix_names(text):
    text = CLAUDE_RE.sub("Claude", text)
    return CODE_AFTER_RE.sub("Claude Code", text)


# ---------------------------------------------------------------- ears
class Silero:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.s = ort.InferenceSession(str(ROOT / "models/silero_vad.onnx"), opts, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros((1, 64), np.float32)

    def __call__(self, chunk):
        x = np.concatenate([self.ctx, chunk[None, :]], axis=1)
        out, self.state = self.s.run(None, {"input": x, "state": self.state, "sr": np.array(SR, np.int64)})
        self.ctx = x[:, -64:]
        return float(out[0][0])


class Ears:
    """Turns the mic into utterances. Each finished utterance is handed to on_utterance(audio, armed).

    The mic is read with pw-record from the echo-cancelled source (foidbot-aec.service) when it exists, so her own
    voice is removed and the user can talk over her. A trained openWakeWord model, if present, fires on_wake the
    moment "Hey/Okay <her name>" is heard, without waiting for Whisper.
    """

    def __init__(self, loop, on_utterance, on_speech_start, on_wake=None):
        self.loop, self.on_utterance, self.on_speech_start, self.on_wake = loop, on_utterance, on_speech_start, on_wake
        self.vad = Silero()
        self.q = queue.Queue()
        self.muted = False
        self.armed = False          # True after the hotkey/wake word: next utterance is a command
        self.armed_at = 0.0
        self.armed_by = None        # "hotkey" | "wake"
        self.end_blocks = int(CFG["ears"]["end_silence_ms"] / 32)
        self.oww, self.ww_buf, self.ww_last = None, np.zeros(0, np.float32), 0.0
        engine = CFG["ears"].get("wake_engine", "both")
        model = cfg_path(CFG["ears"].get("wake_model", "models/wake.onnx"))
        if engine in ("both", "oww") and model.exists():
            from openwakeword.model import Model
            self.oww = Model(wakeword_models=[str(model)], inference_framework="onnx")
            log("wake word model loaded:", model.name)
        self.engine = engine if self.oww else "whisper"

    def start(self):
        threading.Thread(target=self._capture, daemon=True).start()
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _node_exists(name):
        try:
            out = subprocess.run(["pw-cli", "ls", "Node"], capture_output=True, text=True, timeout=3).stdout
            return f'node.name = "{name}"' in out
        except Exception:
            return False

    def _capture(self):
        import select
        nbytes = BLOCK * 4
        while True:
            dev = CFG["ears"].get("device", "foidbot_aec_source")
            target = dev if dev and self._node_exists(dev) else None
            cmd = ["pw-record", "--rate", str(SR), "--channels", "1", "--format", "f32", "--raw", "--latency", "32ms",
                   "-P", "{ node.name = foidbot_ears application.name = foidbot }", "-"]
            if target:
                cmd[1:1] = ["--target", target]
            self.device = target or "default mic"
            log("mic:", self.device)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            fd, buf = p.stdout.fileno(), b""
            while True:
                ready, _, _ = select.select([fd], [], [], 3)
                if not ready:
                    log("mic: no audio for 3 s, reconnecting")
                    break
                data = os.read(fd, 65536)
                if not data:
                    break
                buf += data
                while len(buf) >= nbytes:
                    self.q.put(np.frombuffer(buf[:nbytes], np.float32).copy())
                    buf = buf[nbytes:]
            p.kill()
            p.wait()
            time.sleep(1)

    def arm(self, by="hotkey"):
        self.armed, self.armed_at, self.armed_by = True, time.time(), by

    def _wake_check(self, chunk):
        self.ww_buf = np.concatenate([self.ww_buf, chunk])
        while len(self.ww_buf) >= 1280:   # openWakeWord works on 80 ms frames of 16-bit audio
            frame, self.ww_buf = self.ww_buf[:1280], self.ww_buf[1280:]
            score = max(self.oww.predict((np.clip(frame, -1, 1) * 32767).astype(np.int16)).values())
            if score >= CFG["ears"].get("oww_threshold", 0.5) and time.time() - self.ww_last > 2:
                self.ww_last = time.time()
                log(f"wake word heard ({score:.2f})")
                self.arm("wake")
                if self.on_wake:
                    asyncio.run_coroutine_threadsafe(self.on_wake(), self.loop)

    def _run(self):
        buf, speaking, silent, pre = [], False, 0, []
        while True:
            chunk = self.q.get()
            if self.muted and not self.armed:
                buf, speaking = [], False
                continue
            if self.oww and not self.muted:
                self._wake_check(chunk)
            p = self.vad(chunk)
            if not speaking:
                pre = (pre + [chunk])[-8:]  # keep ~250 ms before speech starts
                if p > 0.55:
                    speaking, silent, buf = True, 0, pre[:]
                    asyncio.run_coroutine_threadsafe(self.on_speech_start(), self.loop)
                elif self.armed and time.time() - self.armed_at > CFG["ears"].get(
                        "follow_up_timeout_s" if self.armed_by == "followup" else "listen_timeout_s", 8):
                    self.armed = False
                    asyncio.run_coroutine_threadsafe(self.on_utterance(None, True), self.loop)
                continue
            buf.append(chunk)
            silent = silent + 1 if p < 0.35 else 0
            if silent >= self.end_blocks or len(buf) > 30 * SR / BLOCK:
                audio = np.concatenate(buf)
                speaking, buf = False, []
                self.vad.reset()
                if len(audio) > 0.35 * SR:
                    armed, self.armed = self.armed, False
                    asyncio.run_coroutine_threadsafe(self.on_utterance(audio, armed), self.loop)


class Whisper:
    def __init__(self):
        import nvidia
        base = Path(nvidia.__path__[0])
        for lib in ["cublas/lib/libcublas.so.12", "cublas/lib/libcublasLt.so.12", "cudnn/lib/libcudnn.so.9"]:
            try:
                import ctypes
                ctypes.CDLL(str(base / lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        from faster_whisper import WhisperModel
        try:
            self.m = WhisperModel(CFG["ears"]["whisper_model"], device="cuda", compute_type="float16")
        except Exception as e:
            log("whisper on CPU:", e)
            self.m = WhisperModel("small.en", device="cpu", compute_type="int8")
        self.lock = threading.Lock()

    def __call__(self, audio):
        with self.lock:
            segs, _ = self.m.transcribe(audio, language="en", beam_size=1, vad_filter=False,
                                        initial_prompt=WHISPER_PROMPT)
            return fix_names(" ".join(s.text.strip() for s in segs).strip())


# Whisper is primed with her name and the home town so it spells them right.
WHISPER_PROMPT = (f"Hey {ME}. OK {ME}. {CFG['location']['name'].split(',')[0]}."
                  + (" Talk to Claude Code." if CFG["claude_code"].get("enabled", True) else ""))
WAKE_WORDS = sorted({w.lower() for w in CFG["ears"].get("wake_words", [])} | {ME.lower()}, key=len, reverse=True)
WAKE_RE = re.compile(r"^\W*(?:(?:hey|hi|ok|okay|oh)\W+)?(?:" + "|".join(map(re.escape, WAKE_WORDS)) + r")\b\W*(.*)$",
                     re.IGNORECASE)
STALL_RE = re.compile(r"(give me a (sec|second|moment|minute)|one moment|hang on|let me (check|look|read|see|find|pull)|i'?ll (check|look|read|find)|(reading|checking|looking at) (that|it|this|the))\W*$",
                      re.IGNORECASE)
def looping(text):
    """True once the tail of the reply has already appeared earlier (the model is repeating itself)."""
    if len(text) < 300:
        return False
    tail = text[-120:]
    return text.find(tail) < len(text) - 120


INTENT_RE = re.compile(r"\b(I'll|I will|let me|next,? I|now I'?ll|first,? I'?ll|going to)\b[^.?!]*[.!]?\s*$", re.IGNORECASE)
DONE_RE = re.compile(r"\b(done|finished|submitted|complete[d]?|all set|that's it|needs? your|confirm|couldn't|can't|unable)\b", re.I)
DONE_CLAIM_RE = re.compile(r"\b(submitted|placed|sent|ordered|posted|purchased|bought|booked|signed up|"
                           r"order (?:is|has been) (?:in|on its way)|has been (?:submitted|placed|sent))\b", re.I)
CLAIM_RE = re.compile(r"\bI(?:'ve| have| just)? (set|saved|added|opened|started|created|noted|clicked|copied|paused|played|"
                      r"skipped|launched|cancell?ed|reminded|scheduled|turned|muted|typed|submitted|made a note)\b", re.IGNORECASE)
CC_ENTER_RE = re.compile(r"\b(talk|talking|mean|meant|speak|chat|connect|switch|relay|send (a )?message|go|get|hook|use|open|interact|work|"
                         r"check|reach|ask|tell)\b.{0,25}\bclaude[ -]?code\b|\bclaude[ -]?code (mode|sessions?)|"
                         r"^\W*(ok(ay)?|hey)?\W*claude[ -]?code\W*$", re.I)
CC_EXIT_RE = re.compile(r"\b(exit|leave|quit|stop|end|close|done with|finished with|get out of)\b.{0,25}\b(claude|code|mode|session)\b|"
                        r"^\W*(that's all|that is all|we're done|back to normal|goodbye claude)\b", re.I)
CC_SWITCH_RE = re.compile(r"\b(switch|change|different|another|other) (to )?(a |the )?(session|one)\b|\bpick (a )?(different|another)", re.I)
CC_REPEAT_RE = re.compile(r"^\W*(read (that|it) (again|back)|repeat( that)?|say that again|what did (it|claude) say)\W*$", re.I)
CC_FULL_RE = re.compile(r"^\W*(read (me )?(the )?(whole|full|entire) (thing|reply|message|answer)|read it all)\W*$", re.I)
CC_STATUS_RE = re.compile(r"^\W*(what'?s it doing|is it (done|finished|still working)|status|how'?s it going)\W*$", re.I)
CLAUDE_CODE = CFG["claude_code"].get("enabled", True)
STOP_RE = re.compile(r"\b(stop|quiet|hush|never ?mind|that's enough|cancel)\b", re.IGNORECASE)


# ---------------------------------------------------------------- voice
class Voice:
    """Kokoro TTS on CPU + a playback thread that reports mouth shapes while it plays."""

    def __init__(self, loop, send):
        from kokoro_onnx import Kokoro
        self.k = Kokoro(str(ROOT / "models/kokoro-v1.0.onnx"), str(ROOT / "models/voices-v1.0.bin"))
        self.loop, self.send = loop, send
        self.play_q = queue.Queue()
        self.stop_flag = threading.Event()
        self.busy = threading.Event()
        threading.Thread(target=self._player, daemon=True).start()
        self.voice_name = CFG["assistant"]["voice"]
        say_as = CFG["assistant"].get("say_as") or ME
        # her name spelled the way the voice pronounces it right (set during onboarding)
        self.say_as = (re.compile(rf"\b{re.escape(ME)}\b", re.I), say_as) if say_as.lower() != ME.lower() else None
        self.k.create("Ready.", voice=self.voice_name)  # warm up

    def synth(self, text):
        if self.say_as:
            text = self.say_as[0].sub(self.say_as[1], text)
        audio, sr = self.k.create(text, voice=self.voice_name, speed=CFG["assistant"].get("speed", 1.0), lang="en-us")
        return audio.astype(np.float32), sr

    def enqueue(self, audio, sr):
        self.busy.set()
        self.play_q.put((audio, sr))

    def stop(self):
        self.stop_flag.set()
        with self.play_q.mutex:
            self.play_q.queue.clear()

    def idle(self):
        return self.play_q.empty() and not self.busy.is_set()

    @staticmethod
    def viseme(block):
        rms = float(np.sqrt(np.mean(block ** 2)))
        if rms < 0.012:
            return "rest"
        zcr = float(np.mean(np.abs(np.diff(np.sign(block))))) / 2
        if rms < 0.035:
            return "m" if zcr < 0.08 else "u"
        if zcr > 0.16:
            return "e"
        return "a" if rms > 0.09 else "o"

    def _player(self):
        with sd.OutputStream(samplerate=24000, channels=1, dtype="float32", blocksize=1024) as out:
            while True:
                audio, sr = self.play_q.get()
                self.stop_flag.clear()
                last = None
                for i in range(0, len(audio), 1024):
                    if self.stop_flag.is_set():
                        break
                    block = audio[i:i + 1024]
                    v = self.viseme(block)
                    if v != last:
                        asyncio.run_coroutine_threadsafe(self.send({"t": "viseme", "v": v}), self.loop)
                        last = v
                    out.write(block.reshape(-1, 1))
                asyncio.run_coroutine_threadsafe(self.send({"t": "viseme", "v": "rest"}), self.loop)
                if self.play_q.empty():
                    self.busy.clear()


# ---------------------------------------------------------------- privacy
def others_present():
    """True if another app is recording the mic or the screen (call, stream, screen share)."""
    try:
        outs = json.loads(subprocess.run(["pactl", "-f", "json", "list", "source-outputs"],
                                         capture_output=True, text=True, timeout=2).stdout or "[]")
        for o in outs:
            pr = o.get("properties", {})
            if str(pr.get("application.process.id")) == str(os.getpid()):
                continue
            if "monitor" in str(pr.get("target.object", "")).lower():
                continue
            if str(pr.get("node.name", "")).startswith("foidbot") or pr.get("application.name") == "foidbot":
                continue   # her own ears and the echo-canceller
            name = (pr.get("application.name") or pr.get("application.process.binary") or "").lower()
            if name and not any(x in name for x in ("pavucontrol", "plasma", "kmix", "speech-dispatcher")):
                return True
        nodes = json.loads(subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=3).stdout or "[]")
        for n in nodes:
            pr = (n.get("info") or {}).get("props") or {}
            if pr.get("media.class") == "Stream/Input/Video" or "screencast" in str(pr.get("node.name", "")).lower():
                return True
    except Exception as e:
        log("privacy check:", e)
    return False


# ---------------------------------------------------------------- the helper
class Assistant:
    def __init__(self):
        self.clients = set()
        self.history = []
        self.mode_setting = "auto"
        self.private = False
        self.busy = False
        self.task = None
        self.tools = Tools(CFG)
        self.state = "idle"
        self.last_activity = time.time()
        self.last_notice = None
        self.agent = None
        self.company_until = 0.0    # an unfamiliar voice spoke to her: stay reserved until then
        self.speakers = None
        self.cc = ClaudeCode(log)
        self.cc_mode = None         # None | "pick" | "relay"
        self.cc_choices = []
        self.cc_last = ""           # Claude Code's last full reply

    # ---- websocket
    async def send(self, msg):
        if self.clients:
            data = json.dumps(msg)
            await asyncio.gather(*(c.send(data) for c in list(self.clients)), return_exceptions=True)

    async def set_state(self, s):
        if s != self.state:
            self.state = s
            await self.send({"t": "state", "state": s})

    async def ws_handler(self, ws):
        self.clients.add(ws)
        await ws.send(json.dumps({"t": "state", "state": self.state}))
        await ws.send(json.dumps({"t": "mode", "setting": self.mode_setting, "private": self.private}))
        if self.cc_mode == "relay" and self.cc.target:
            await ws.send(json.dumps({"t": "cc", "session": self.cc.target["name"], "title": self.cc.target["title"]}))
        if self.last_notice:
            await ws.send(json.dumps(self.last_notice))
            self.last_notice = None
        try:
            async for raw in ws:
                m = json.loads(raw)
                t = m.get("t")
                if t == "listen":
                    if self.state in ("speaking", "thinking", "working"):
                        await self.interrupt()
                    elif self.ears.armed:
                        self.ears.armed = False
                        await self.set_state("idle")
                        continue
                    self.ears.arm("hotkey")
                    await self.set_state("listening")
                elif t == "stop":
                    await self.interrupt()
                elif t == "text" and m.get("text", "").strip():
                    await self.interrupt()
                    self.task = asyncio.create_task(self.converse(m["text"].strip()))
                elif t == "mode":
                    self.mode_setting = m.get("setting", "auto")
                    await self.refresh_privacy()
                elif t == "mute":
                    self.ears.muted = bool(m.get("on"))
                    await self.send({"t": "muted", "on": self.ears.muted})
        finally:
            self.clients.discard(ws)

    # ---- privacy
    async def refresh_privacy(self):
        if self.mode_setting == "auto":
            private = time.time() > self.company_until and not await asyncio.to_thread(others_present)
        else:
            private = self.mode_setting == "private"
        if private != self.private:
            log("private" if private else "company")
        self.private = private
        await self.send({"t": "mode", "setting": self.mode_setting, "private": private})

    async def privacy_loop(self):
        while True:
            await self.refresh_privacy()
            await asyncio.sleep(15)

    # ---- ears callbacks
    async def on_speech_start(self):
        if self.state == "sleep":
            await self.set_state("idle")

    async def on_wake(self):
        """The wake-word model heard "Hey/Okay <her name>" (Ears has already armed itself)."""
        if self.state in ("speaking", "thinking", "working"):
            log("interrupted by the wake word")
            self.voice.stop()
            if self.cc_mode != "relay" or self.state == "speaking":
                await self.interrupt()
            self.ears.arm("wake")
        await self.set_state("listening")
        await self.send({"t": "chime"})

    async def on_utterance(self, audio, armed):
        if audio is None:  # woke but nothing said
            if self.state == "listening":
                await self.set_state("idle")
            return
        speaking = self.state in ("speaking", "thinking", "working")
        if self.ears.engine == "oww" and not armed and not speaking:
            return  # the wake-word model decides when she listens; skip transcribing background speech
        armed_by = self.ears.armed_by if armed else None
        who_task = asyncio.create_task(asyncio.to_thread(self.speakers.is_user, audio)) if self.speakers else None
        text = await asyncio.to_thread(self.whisper, audio)
        who = await who_task if who_task else None
        if not text:
            return
        if CFG.get("log", {}).get("log_heard", True):
            log("heard:", repr(text), f"[voice={'user' if who else 'other' if who is False else '?'}",
                f"listening={armed_by or ('while ' + self.state if speaking else 'no')}]")
        m = WAKE_RE.match(text)
        words = len(text.split())
        if speaking:
            barge = CFG["ears"].get("barge_in", True) and who is True and words >= 2
            if m or STOP_RE.search(text) and words <= 4 or barge:
                log("interrupted by:", text, "(the user's voice)" if barge and not m else "")
                await self.interrupt()
                cmd = m.group(1) if m else text
                if STOP_RE.search(cmd) and len(cmd.split()) <= 4:
                    return
                if len(cmd.split()) >= 2:
                    self.task = asyncio.create_task(self.converse(cmd))
                elif m:
                    self.ears.arm("wake")
                    await self.set_state("listening")
            return
        if armed:
            if armed_by == "followup" and (who is False or words < 2 and re.fullmatch(
                    r"\W*(um+|uh+|hm+|huh|ah+|oh|mm+|you|thank you|bye)\W*", text, re.I)):
                return   # follow-up window: only the user's own voice, and not filler or Whisper's phantom words
            if m:
                command = m.group(1)
            elif armed_by == "wake":   # the model heard the name; Whisper may have spelled it differently
                command = re.sub(r"^\W*(?:(?:hey|hi|ok|okay|oh)\W+)?\w+\W*", "", text, flags=re.I) if words <= 2 or \
                    re.match(r"^\W*(hey|hi|ok|okay|oh)\b", text, re.I) else text
            else:
                command = text
            if not command.strip():
                self.ears.arm(armed_by or "wake")   # just the name: wait for the question
                await self.set_state("listening")
                return
            if STOP_RE.fullmatch(command.strip(" .!?,")):
                await self.set_state("idle")
                return
        elif m:
            command = m.group(1)
            if len(command.split()) < 2:  # just "Hey <name>" -> wait for the question
                self.ears.arm("wake")
                await self.set_state("listening")
                await self.send({"t": "chime"})
                return
        else:
            return  # background speech, not for her
        if who is False and armed_by != "followup":
            log("an unfamiliar voice is talking to her: company mode for 10 minutes")
            self.company_until = time.time() + 600
        elif who is True and self.company_until and self.mode_setting == "auto":
            pass  # the user; company mode still ends on its own timer
        self.task = asyncio.create_task(self.converse(command))

    async def interrupt(self):
        self.voice.stop()
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
        await self.send({"t": "reply_end", "sources": []})
        await self.set_state("idle")

    # ---- brain
    async def converse(self, text):
        self.last_activity = time.time()
        log("you:", text)
        await self.send({"t": "user", "text": text})
        if self.cc_mode or CLAUDE_CODE and CC_ENTER_RE.search(text) and len(text.split()) <= 16:
            return await self.claude_turn(text)
        await self.set_state("thinking")
        await self.refresh_privacy()
        self.history.append({"role": "user", "content": text})
        self.tools.last_user = (text, time.time())
        if self.tools.browser:
            self.tools.browser.last_user = self.tools.last_user
            self.tools.browser.risky_done = False
            self.tools.browser.prev_ask = next((m["content"] for m in reversed(self.history[:-1]) if m["role"] == "assistant"), "")
        msgs = [{"role": "system", "content": system_prompt(CFG, self.private, self.weather.brief)}] + \
            self.history[-CFG["llm"]["history_turns"] * 2:]
        sources, reply_started = [], False
        group = await self.route(text)
        if group == "browser" and self.agent:
            log("route: browser (agent)")
            return await self.web_task(text)
        speaker = Speaker(self, hold=group == "browser")
        must_look_up = group != "chat"
        log("route:", group)
        try:
            nudges, browser_steps = 0, 0
            for _round in range(14 if group == "browser" else 6):
                content, raw, calls, skipped = "", "", {}, 0
                used_tools = any(m.get("role") == "tool" for m in msgs[-12:]) and _round > 0
                forced = must_look_up and not used_tools
                async for kind, payload in self.stream_llm(msgs, group, "required" if forced else "auto"):
                    if kind == "text" and forced:
                        skipped += len(payload)
                        if skipped > 240:
                            break  # rambling instead of calling a tool; retry with a nudge below
                        continue  # text streamed before a forced tool call is a guess made without the results
                    if kind == "text":
                        raw += payload
                        if looping(raw):
                            log("repetition detected, cutting the reply")
                            break
                        if "</think>" in content + payload:   # leaked reasoning: drop everything before it
                            content, payload = "", (content + payload).split("</think>")[-1]
                        content += payload
                        if not reply_started and not calls:
                            if len(content) < 24 and content.lstrip().startswith("[") and "]" not in content:
                                continue  # still inside the mood tag
                            reply_started = True
                            await self.send({"t": "reply_start"})
                            await self.set_state("speaking")
                            payload = content
                        if reply_started:
                            payload = await self.strip_mood(payload, speaker)
                            if payload:
                                if not speaker.hold:
                                    await self.send({"t": "reply", "delta": payload})
                                await speaker.feed(payload)
                    elif kind == "tool":
                        i = payload["index"]
                        c = calls.setdefault(i, {"id": "", "name": "", "args": ""})
                        c["id"] = payload.get("id") or c["id"]
                        fn = payload.get("function") or {}
                        c["name"] += fn.get("name") or ""
                        c["args"] += fn.get("arguments") or ""
                if not calls:
                    if forced and nudges < 2:
                        nudges += 1
                        log("no tool call in a forced round; nudging")
                        msgs.append({"role": "user", "content": "(Call the right tool now. Do not answer in text yet.)"})
                        continue
                    if (group == "browser" and browser_steps and nudges < 4 and not content.rstrip().endswith("?")
                            and not DONE_RE.search(content)):
                        nudges += 1
                        msgs.append({"role": "assistant", "content": content or "(working)"})
                        msgs.append({"role": "user", "content": "(Keep going with your tools until the task is finished, "
                                                                f"a step needs {USER}'s confirmation, or you need to ask {P['obj']} something.)"})
                        continue
                    b = self.tools.browser
                    if group == "browser" and b and not b.risky_done and DONE_CLAIM_RE.search(content) and nudges < 5:
                        nudges += 1
                        log("claimed a submit/send that never ran; asking for a correction")
                        speaker.discard()
                        msgs.append({"role": "assistant", "content": content})
                        msgs.append({"role": "user", "content": "(Check yourself: no submit, order, send or post action has "
                                     "actually run. Look at the page with browse_look, then either continue the task or tell "
                                     f"{USER} honestly where things stand.)"})
                        continue
                    if _round == 0 and not must_look_up and CLAIM_RE.search(content):
                        log("claimed an action without a tool; retrying with a tool required")
                        must_look_up, group = True, "computer"
                        speaker.worker.cancel()
                        self.voice.stop()
                        speaker = Speaker(self, hold=group == "browser")
                        await self.send({"t": "reply_start"})
                        reply_started = False
                        continue
                    short_promise = len(content) < 140 and re.match(r"\W*(let me|i'?ll|one moment|give me|checking|looking)", content.strip(), re.I)
                    if _round < 5 and (short_promise or STALL_RE.search(content[-80:]) or (group == "browser" and INTENT_RE.search(content[-160:]))):
                        msgs.append({"role": "assistant", "content": content})
                        msgs.append({"role": "user", "content": "(Go ahead: use your tools now, then give the answer.)"})
                        await speaker.feed(" ")
                        continue
                    break
                if content.strip():
                    await speaker.feed(" ")
                    if not speaker.hold:
                        await self.send({"t": "reply", "delta": " "})
                await self.set_state("working")
                msgs.append({"role": "assistant", "content": content or None, "tool_calls": [
                    {"id": c["id"] or f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                    for i, c in sorted(calls.items())]})
                for i, c in sorted(calls.items()):
                    try:
                        args = json.loads(c["args"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    log("tool:", c["name"], args)
                    browser_steps += c["name"].startswith("browse")
                    res = await self.tools.call(c["name"], args)
                    log("   ->", json.dumps(res, default=str)[:160])
                    for s in res.pop("sources", []):
                        if s["url"] not in [x["url"] for x in sources]:
                            sources.append(s)
                    msgs.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}",
                                 "content": json.dumps(res, default=str)[:12000]})
                await self.set_state("thinking")
            await speaker.flush()
            full = speaker.spoken_text
            if speaker.hold and full.strip():
                await self.send({"t": "reply_start"})
                await self.send({"t": "reply", "delta": full.strip()})
            self.history.append({"role": "assistant", "content": full})
            log(f"{ME.lower()}:", full)
            await self.send({"t": "reply_end", "sources": sources[:5]})
            while not self.voice.idle():
                await asyncio.sleep(0.1)
            await self.maybe_follow_up(full)
        except asyncio.CancelledError:
            speaker.worker.cancel()
            if speaker.spoken_text:
                self.history.append({"role": "assistant", "content": speaker.spoken_text + " (interrupted)"})
            raise
        except Exception as e:
            log("converse error:", repr(e))
            await self.speak_line("Sorry, something went wrong on my end.", "concerned")
        finally:
            self.last_activity = time.time()
            if self.state != "listening":
                await self.set_state("idle")

    async def maybe_follow_up(self, said):
        """She asked the user something: listen for the answer without needing the wake word."""
        mode = CFG["ears"].get("follow_up", "questions")   # questions | always | off
        if mode == "off" or not said.strip() or self.state == "listening":
            return
        if mode == "always" or "?" in said.strip()[-120:]:
            self.ears.arm("followup")
            await self.set_state("listening")
            if CFG["ears"].get("follow_up_chime", False):
                await self.send({"t": "chime"})

    # ---- Claude Code interaction mode
    async def claude_turn(self, text):
        """Relay mode for a running Claude Code session (claude_code.py). Her own lines here are written in code."""
        await self.set_state("thinking")
        self.history.append({"role": "user", "content": text})
        try:
            short = len(text.split()) <= 7   # longer sentences are messages for Claude Code, even if they say "code"
            if self.cc_mode and short and CC_EXIT_RE.search(text):
                self.cc_mode, self.cc.target = None, None
                await self.send({"t": "cc", "session": None})
                say = "Okay, I've stepped away from Claude Code. It's just us again."
            elif not self.cc_mode or short and (CC_SWITCH_RE.search(text) or self.cc_mode == "relay" and CC_ENTER_RE.search(text)):
                say = await self.cc_list(text)
            elif self.cc_mode == "pick":
                say = await self.cc_pick(text)
            else:
                say = await self.cc_relay(text)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            import traceback
            log("claude code error:", repr(e), traceback.format_exc()[-500:])
            say = "Something went wrong while I was talking to Claude Code."
        self.history.append({"role": "assistant", "content": say})
        log(f"{ME.lower()}:", say)
        await self.speak_line(say, "calm")
        if self.cc_mode:   # let the user answer without the wake word
            self.ears.arm("followup")
            await self.set_state("listening")

    async def cc_list(self, text):
        sessions = await self.cc.sessions()
        usable = [x for x in sessions if x["tab"] and not x["status"].startswith("paused")]
        paused = [x for x in sessions if x["status"].startswith("paused")]
        note = (f" {len(paused)} more {'is' if len(paused) == 1 else 'are'} paused in the background, so I can't reach "
                f"{'it' if len(paused) == 1 else 'them'}.") if paused else ""
        if not usable:
            self.cc_mode = None
            return "I don't see a Claude Code session I can type into." + note
        rest = CC_ENTER_RE.sub(" ", text)
        pick = match_session(rest, usable) if len(re.findall(r"[a-z0-9]+", rest.lower())) > 2 else None
        if pick or len(usable) == 1:
            return await self.cc_select(pick or usable[0], note)
        self.cc_mode, self.cc_choices = "pick", usable
        ords = ["First", "Second", "Third", "Fourth", "Fifth"]
        parts = [f"{ords[i] if i < 5 else 'Then'}, {x['title'] or x['name']}{' (busy right now)' if x['status'] == 'busy' else ''}."
                 for i, x in enumerate(usable[:5])]
        return f"You have {len(usable)} Claude Code sessions open. " + " ".join(parts) + note + " Which one?"

    async def cc_pick(self, text):
        pick = match_session(text, self.cc_choices)
        if not pick:
            return "Sorry, which one did you mean? You can say first, second, or part of its name."
        return await self.cc_select(pick)

    async def cc_select(self, target, note=""):
        self.cc.target, self.cc_mode, self.cc_last = target, "relay", ""
        await self.send({"t": "cc", "session": target["name"], "title": target["title"]})
        busy = " It's busy right now, so anything you say will wait in line." if target["status"] == "busy" else ""
        prompt = await asyncio.to_thread(self.cc.waiting_prompt)
        if prompt:
            return f"I'm with {target['title'] or target['name']}. " + await self.cc_describe_prompt(prompt)
        return f"I'm with {target['title'] or target['name']}.{busy}{note} What should I tell it?"

    async def cc_relay(self, text):
        if CC_REPEAT_RE.search(text):
            return self.cc_spoken if getattr(self, "cc_spoken", "") else "It hasn't said anything since we connected."
        if CC_FULL_RE.search(text):
            return speechify(self.cc_last) if self.cc_last else "There's nothing to read yet."
        prompt = await asyncio.to_thread(self.cc.waiting_prompt)
        if CC_STATUS_RE.search(text):
            if prompt:
                return await self.cc_describe_prompt(prompt)
            return {"busy": "It's still working.", "idle": "It's finished and waiting for you."}.get(
                self.cc.status(), "I can't tell what it's doing right now.")
        if prompt:   # a permission or choice prompt is open: answer it instead of typing a message
            m = re.search(r"\b(?:option|number|choice)\s+(one|two|three|four|[1-4])\b", text, re.I)
            if m:
                answer = int({"one": 1, "two": 2, "three": 3, "four": 4}.get(m.group(1).lower(), m.group(1)))
            elif re.search(r"\b(don't ask again|always|every time)\b", text, re.I):
                answer = "always"
            elif YES_RE.search(text):
                answer = "yes"
            elif re.match(r"^\W*(no|nope|don't|do not|cancel|stop|deny)\b", text, re.I) and len(text.split()) <= 3:
                answer = "no"
            else:
                answer = None
            if answer is not None:
                if err := await self.cc.choose(answer):
                    return err[0].upper() + err[1:] + "."
                await self.set_state("working")
                return await self.cc_wait()
            await self.cc.key("\x1b")   # anything else: decline the prompt and pass the user's words along instead
            await asyncio.sleep(0.8)
        res = await self.cc.send(text)
        if res.get("error"):
            return res["error"][0].upper() + res["error"][1:] + "."
        await self.set_state("working")
        return await self.cc_wait()

    async def cc_wait(self):
        texts, state = await self.cc.wait_reply(timeout=90)
        if state == "gone":
            self.cc_mode = None
            await self.send({"t": "cc", "session": None})
            return "That Claude Code session has closed, so I've left Claude Code mode."
        if state == "prompt":
            lead = await self.cc_summary(texts) + " " if texts else ""
            return lead + await self.cc_describe_prompt(await asyncio.to_thread(self.cc.waiting_prompt))
        if state == "working":
            if self.cc.watch and not self.cc.watch.done():
                self.cc.watch.cancel()
            self.cc.watch = asyncio.create_task(self.cc_watch(self.cc.target, self.cc.sent_at))
            return "It's still working on that. I'll tell you when it's done."
        return await self.cc_summary(texts)

    async def cc_watch(self, target, since):
        texts, state = await self.cc.wait_reply(timeout=3600, target=target, since=since)
        if state in ("done", "prompt") and texts:
            name = target["title"] or target["name"]
            msg = await self.cc_summary(texts)
            await self.announce(f"Claude Code finished in {name}. {msg}" if state == "done"
                                else f"Claude Code needs you in {name}. {msg}", "info")

    async def cc_summary(self, texts):
        full = "\n\n".join(texts).strip()
        self.cc_last = full
        if not full:
            self.cc_spoken = "It finished without saying anything."
            return self.cc_spoken
        plain = speechify(full)
        if len(plain) > 380:
            body = {"model": "foidbot", "temperature": 0.3, "max_tokens": 220,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "system", "content":
                                  f"You are {ME}, {USER}'s assistant. Claude Code (an AI coding assistant) just replied to "
                                  f"{USER}, and you tell {P['obj']} the gist out loud. Report it in the third person, e.g. 'It "
                                  "explained that…', 'It fixed…', 'It's asking whether…'. At most three short spoken "
                                  f"sentences, plain English, no markdown, no code. Keep any question it asks {P['obj']} and "
                                  f"anything it needs {P['obj']} to do. Never answer or respond to the message yourself."},
                                 {"role": "user", "content": "Claude Code's reply:\n<<<\n" + full[-6000:] +
                                  f"\n>>>\nTell {USER} the gist in at most three sentences."}]}
            try:
                plain = (await self.llm_post(body))["choices"][0]["message"]["content"].strip()
                plain += " Say read it all if you want the whole thing."
            except Exception as e:
                log("cc summary:", e)
                plain = plain[:380].rsplit(" ", 1)[0] + "…"
        self.cc_spoken = plain
        return plain

    async def cc_describe_prompt(self, screen):
        body = {"model": "foidbot", "temperature": 0.2, "max_tokens": 60,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content":
                              "This is the bottom of a Claude Code terminal showing a question or permission prompt. In ONE "
                              "short spoken sentence (under 20 words), starting with 'It wants to' or 'It's asking', say "
                              "what it wants to do, in plain words. Don't list the choices. No markdown, no full paths."},
                             {"role": "user", "content": screen[-2500:]}]}
        try:
            desc = speechify((await self.llm_post(body))["choices"][0]["message"]["content"].strip())
        except Exception:
            desc = "It's asking you something on screen."
        opts, _ = await asyncio.to_thread(self.cc.options)
        words = ["one", "two", "three", "four", "five", "six"]
        if opts:
            short = lambda o: o if len(o) <= 45 else o[:45].rsplit(" ", 1)[0] + "…"
            choices = "; ".join(f"{words[i]}, {short(o)}" for i, o in enumerate(opts[:6]))
            self.cc_spoken = f"{desc} Choices: {choices}. Say yes, no, or an option number."
        else:
            self.cc_spoken = desc + " Say yes or no."
        return self.cc_spoken

    async def web_task(self, text):
        """Website tasks run as a code-driven loop (browser_agent); she just voices the outcome."""
        b, a = self.tools.browser, self.agent
        await self.set_state("working")
        try:
            if b.pending and re.match(r"^\W*(no|nope|don't|do not|cancel|stop|wait)\b", text, re.I):
                msg, _ = await a.confirmed(False, self.tools.last_user)
            elif b.pending and YES_RE.search(text):
                msg, _ = await a.confirmed(True, self.tools.last_user)
            elif a.task and a.awaiting:
                a.awaiting = False
                a.steps.append(f"- {USER} said: {text}")
                msg, _ = await a.run()
            else:
                b.pending = None   # a new request replaces anything left half-done
                recent = " / ".join(m["content"][:200] for m in self.history[-5:-1] if m.get("content"))
                msg, _ = await a.run(task=f"{text}" + (f"  (Recent conversation: {recent})" if recent else ""))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            import traceback
            log("web task error:", repr(e), traceback.format_exc()[-600:])
            msg = "Something went wrong in the browser."
        sentences = re.split(r"(?<=[.!?])\s+", msg.strip())
        msg = " ".join(sentences[:3])   # spoken: keep web results to three sentences
        self.history.append({"role": "assistant", "content": msg})
        log(f"{ME.lower()}:", msg)
        await self.speak_line(msg, "calm")
        await self.maybe_follow_up(msg)

    async def llm_post(self, body, url=None):
        r = await http.post(url or CFG["llm"]["url"], json=body, timeout=90)
        if url and r.status_code != 200:  # heavy brain went away; fall back to the 9B
            r = await http.post(CFG["llm"]["url"], json=body, timeout=60)
        return r.json()

    async def strip_mood(self, text, speaker):
        """Pulls [mood] tags out of the stream (they can arrive split across chunks) and sends the first one."""
        speaker.tagbuf += text
        out = ""
        while speaker.tagbuf:
            i = speaker.tagbuf.find("[")
            if i < 0:
                out, speaker.tagbuf = out + speaker.tagbuf, ""
                break
            out += speaker.tagbuf[:i]
            rest = speaker.tagbuf[i:]
            stage = re.match(r"\[[^\]\n]{3,40}\]\s*", rest)
            m = re.match(r"\[(\w{2,12})\]\s*", rest)
            if stage and not m:
                speaker.tagbuf = rest[stage.end():]
                continue
            if m:
                mood = m.group(1).lower()
                if not speaker.mood_done:
                    allowed = PUBLIC_MOODS + (PRIVATE_MOODS if self.private else [])
                    await self.send({"t": "mood", "mood": mood if mood in allowed else "calm"})
                    speaker.mood_done = True
                speaker.tagbuf = rest[m.end():]
            elif len(rest) < 42 and re.fullmatch(r"\[[^\]\n]*", rest):
                speaker.tagbuf = rest  # wait for the rest of the tag
                break
            else:
                out, speaker.tagbuf = out + "[", rest[1:]
        if not speaker.mood_done and out.strip():
            speaker.mood_done = True
        return out.lstrip() if not speaker.spoken_text else out

    async def route(self, text):
        """One-word pass that picks which tools (if any) this message needs. Small menus keep a 9B model reliable."""
        if self.tools.browser and self.tools.browser.pending and re.match(r"^\W*(yes|yeah|yep|sure|ok|okay|go ahead|do it|confirm|no|nope|don't|cancel)\b", text, re.I):
            return "browser"
        recent = " / ".join(m["content"][:160] for m in self.history[-4:-1] if m.get("content"))
        body = {"model": "foidbot", "max_tokens": 4, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content":
                              "Classify the new message with one word:\n"
                              "lookup - needs facts: the world, people, places, events, sports, science, history, numbers, "
                              "prices, stocks, business hours, news, weather\n"
                              "browser - wants something done on a website: fill a form, order, sign up, click, add to cart, "
                              "check an account, or continue a website task already under way\n"
                              "computer - timers, reminders, notes, opening apps or links, music, volume, clipboard, "
                              f"or something on {P['poss']} screen\n"
                              "chat - greetings, small talk, feelings, opinions, flirting, jokes, stories, or follow-ups the "
                              "conversation already answers\nReply with just the word."},
                             {"role": "user", "content": f"Conversation so far: {recent or '(none)'}\nNew message: {text}"}]}
        try:
            r = await http.post(CFG["llm"]["url"], json=body, timeout=10)
            ans = r.json()["choices"][0]["message"]["content"].strip().lower()
            return next((g for g in ("lookup", "browser", "computer", "chat") if ans.startswith(g)), "chat")
        except Exception as e:
            log("route:", e)
            return "lookup"

    async def stream_llm(self, msgs, group, tool_choice="auto"):
        tools = self.tools.schema(group) if group != "chat" else []
        body = {"model": "foidbot", "messages": msgs, "stream": True,
                **({"tools": tools, "tool_choice": tool_choice} if tools else {}),
                "temperature": CFG["llm"]["temperature"], "max_tokens": 3000,
                "chat_template_kwargs": {"enable_thinking": CFG["llm"].get("thinking", False)}}
        async with http.stream("POST", CFG["llm"]["url"], json=body, timeout=120) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                delta = json.loads(line[6:])["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield "text", delta["content"]
                for tc in delta.get("tool_calls") or []:
                    yield "tool", tc

    async def speak_line(self, text, mood="calm"):
        await self.send({"t": "mood", "mood": mood})
        await self.send({"t": "reply_start"})
        await self.send({"t": "reply", "delta": text})
        await self.send({"t": "reply_end", "sources": []})
        await self.set_state("speaking")
        audio, sr = await asyncio.to_thread(self.voice.synth, text)
        self.voice.enqueue(audio, sr)
        while not self.voice.idle():
            await asyncio.sleep(0.1)
        await self.set_state("idle")

    async def announce(self, text, level):
        log("announce:", level, text)
        hour = datetime.now().hour
        quiet = hour >= 23 or hour < 7
        if level not in ("alert", "reminder") and quiet:
            self.last_notice = {"t": "notice", "text": text, "level": level}
            await self.send(self.last_notice)
            return
        while self.task and not self.task.done():
            await asyncio.sleep(1)
        self.history.append({"role": "assistant", "content": text})
        await self.speak_line(text, "happy" if level == "reminder" else level)

    async def ask_vision(self, question, jpeg_b64):
        body = {"model": "foidbot", "max_tokens": 500, "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}},
                    {"type": "text", "text": f"This is a screenshot of {USER}'s screen. {question}\n"
                                             "Answer factually from what is visible; quote on-screen text exactly when relevant."}]}]}
        r = await http.post(CFG["llm"]["url"], json=body, timeout=90)
        return r.json()["choices"][0]["message"]["content"]

    async def sleep_loop(self):
        while True:
            await asyncio.sleep(30)
            if self.cc_mode and time.time() - self.last_activity > 30 * 60:
                self.cc_mode, self.cc.target = None, None
                await self.send({"t": "cc", "session": None})
            if datetime.now().minute == 0 and datetime.now().second < 30:
                prune_transcripts()
            if self.state == "idle" and time.time() - self.last_activity > 15 * 60:
                await self.set_state("sleep")

    async def main(self):
        loop = asyncio.get_running_loop()
        prune_transcripts()
        log("loading voice and ears…")
        self.voice = Voice(loop, self.send)
        self.whisper = Whisper()
        self.ears = Ears(loop, self.on_utterance, self.on_speech_start, self.on_wake)
        try:
            from speaker import Speakers
            self.speakers = Speakers()
            log("voiceprint:", "loaded" if self.speakers.ref is not None else "none yet (run foidbot-enroll)")
        except Exception as e:
            log("speaker recognition off:", e)
        self.weather = WeatherWatch(self.tools, self.announce)
        self.tools.browser = Browser()
        self.heavy = Heavy(http, log, CFG)
        self.agent = BrowserAgent(self.tools.browser, self.llm_post, log, self.heavy.acquire, self.tools.t_web_search)
        self.tools.timers = self.timers = Timers(self.announce)
        self.tools.ask_vision = self.ask_vision
        async with websockets.serve(self.ws_handler, "127.0.0.1", CFG["daemon"].get("port", 8770)):
            self.ears.start()
            log(f"{ME} is listening.")
            await asyncio.gather(self.weather.run(), self.timers.run(), self.heavy.reaper(), self.privacy_loop(), self.sleep_loop())


class Speaker:
    """Collects streamed text, cuts it into sentences, synthesizes each while the previous one plays."""

    def __init__(self, s: "Assistant", hold=False):
        self.s, self.buf, self.spoken_text, self.mood_done, self.tagbuf = s, "", "", False, ""
        self.hold, self.held = hold, []   # hold: queue sentences until the turn is verified (browser tasks)
        self.pending = asyncio.Queue()
        self.worker = asyncio.create_task(self._work())

    async def feed(self, text):
        self.buf += text
        self.spoken_text += text
        while True:
            m = re.search(r"(.+?[.!?…])(\s+|$)(?=\S)", self.buf, re.S)
            if not m or len(m.group(1)) < 12:
                break
            await self._put(m.group(1).strip())
            self.buf = self.buf[m.end():]

    async def _put(self, sentence):
        if self.hold:
            self.held.append(sentence)
        else:
            await self.pending.put(sentence)

    def discard(self):
        """Drop an unverified reply (held sentences and its text)."""
        self.held, self.buf, self.spoken_text = [], "", ""

    async def flush(self):
        if self.buf.strip():
            await self._put(self.buf.strip())
        for sentence in self.held:
            await self.pending.put(sentence)
        self.held = []
        self.buf = ""
        await self.pending.put(None)
        await self.worker

    async def _work(self):
        try:
            while (sentence := await self.pending.get()) is not None:
                sentence = re.sub(r"https?://\S+", "", sentence).replace("*", "")
                if sentence.strip():
                    audio, sr = await asyncio.to_thread(self.s.voice.synth, sentence)
                    self.s.voice.enqueue(audio, sr)
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(Assistant().main())
