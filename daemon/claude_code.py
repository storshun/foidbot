"""Claude Code interaction mode: relay the user's words into a running Claude Code session and read its reply back.

Sessions come from ~/.claude/sessions/<pid>.json. Text is typed into the session's own terminal, so it arrives
exactly as if the user had typed it: with tmux send-keys when the session runs inside tmux, otherwise into its
Konsole tab over D-Bus (Session.sendText; Konsole only allows that with "Enable the security sensitive parts of
the DBus API" switched on in its settings). The reply is read from the session's
transcript (~/.claude/projects/<cwd>/<sessionId>.jsonl) once the session's status goes back to idle.
"""
import asyncio
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

CLAUDE = Path.home() / ".claude"
QDBUS = "qdbus-qt6"
PROMPT_RE = re.compile(r"(Do you want to|Would you like to|❯ ?1\.|│ ?❯|\b1\. Yes\b|Esc to cancel)", re.I)


def _q(*args, timeout=3):
    r = subprocess.run([QDBUS, *args], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def _proc_state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return None


def _konsole_tabs():
    """{foreground pid: (dbus service, session path)} for every Konsole tab."""
    out = {}
    try:
        names = subprocess.run(["gdbus", "call", "--session", "--dest", "org.freedesktop.DBus", "--object-path",
                                "/org/freedesktop/DBus", "--method", "org.freedesktop.DBus.ListNames"],
                               capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return out
    for svc in re.findall(r"'(org\.kde\.konsole-\d+)'", names):
        for path in _q(svc).splitlines():
            if re.fullmatch(r"/Sessions/\d+", path):
                try:
                    out[int(_q(svc, path, "foregroundProcessId"))] = (svc, path)
                except ValueError:
                    pass
    return out


def _tty(pid):
    try:
        return os.readlink(f"/proc/{pid}/fd/0")
    except OSError:
        return None


def _tmux_panes():
    """{tty: pane id} for every tmux pane."""
    try:
        out = subprocess.run(["tmux", "list-panes", "-a", "-F", "#{pane_tty} #{pane_id}"],
                             capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return {}
    return dict(line.split() for line in out.splitlines() if len(line.split()) == 2)


def _transcript(info):
    proj = re.sub(r"[^A-Za-z0-9]", "-", info.get("cwd", ""))
    return CLAUDE / "projects" / proj / f"{info['sessionId']}.jsonl"


def _title(path):
    """The session's auto-generated title and the last thing the user typed into it."""
    title = prompt = ""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 400_000))
            for line in f.read().decode("utf-8", "ignore").splitlines():
                if '"ai-title"' in line or '"last-prompt"' in line:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    title = d.get("aiTitle") or title
                    prompt = d.get("lastPrompt") or prompt
    except OSError:
        pass
    return title, prompt


def list_sessions():
    tabs, panes = _konsole_tabs(), _tmux_panes()
    found = []
    for f in sorted((CLAUDE / "sessions").glob("*.json")):
        try:
            info = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        pid = info.get("pid")
        state = _proc_state(pid)
        if not state or info.get("kind", "interactive") != "interactive":
            continue
        path = _transcript(info)
        title, prompt = _title(path)
        found.append({"pid": pid, "name": info.get("name") or str(pid), "cwd": info.get("cwd", ""),
                      "status": "paused (Ctrl+Z)" if state == "T" else info.get("status", "unknown"),
                      "title": title, "last_prompt": prompt[:160], "session_id": info["sessionId"],
                      "transcript": str(path), "updated": info.get("updatedAt", 0) / 1000,
                      "tab": ("tmux", panes[_tty(pid)]) if _tty(pid) in panes else
                             ("konsole", *tabs[pid]) if pid in tabs else None})
    found.sort(key=lambda s: -s["updated"])
    return found


def _status(pid):
    try:
        return json.loads((CLAUDE / "sessions" / f"{pid}.json").read_text()).get("status", "unknown")
    except (OSError, json.JSONDecodeError):
        return "gone"


def _assistant_text_since(transcript, since_iso):
    """All assistant text (not thinking/tool calls) written after since_iso, main thread only."""
    texts = []
    try:
        with open(transcript, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 2_000_000))
            lines = f.read().decode("utf-8", "ignore").splitlines()
    except OSError:
        return texts
    for line in lines:
        if '"assistant"' not in line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "assistant" or d.get("isSidechain") or (d.get("timestamp") or "") <= since_iso:
            continue
        for block in (d.get("message") or {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                texts.append(block["text"].strip())
    return texts


class ClaudeCode:
    def __init__(self, log):
        self.log = log
        self.target = None        # the session dict the user picked
        self.sent_at = None       # ISO time of the last relayed message
        self.watch = None         # background task waiting for a long reply

    async def sessions(self):
        return await asyncio.to_thread(list_sessions)

    def screen(self):
        tab = self.target and self.target.get("tab")
        if not tab:
            return ""
        try:
            if tab[0] == "tmux":
                return subprocess.run(["tmux", "capture-pane", "-p", "-t", tab[1]], capture_output=True, text=True,
                                      timeout=3).stdout[-4000:]
            return _q(tab[1], tab[2], "getAllDisplayedText", "true")[-4000:]
        except Exception:
            return ""

    def _type(self, text):
        """Type text into the session's terminal. Returns an error string or None."""
        tab = self.target["tab"]
        if tab[0] == "tmux":
            keys = ["Enter"] if text == "\r" else ["Escape"] if text == "\x1b" else ["-l", text]
            r = subprocess.run(["tmux", "send-keys", "-t", tab[1], *keys], capture_output=True, text=True, timeout=3)
        else:
            r = subprocess.run([QDBUS, tab[1], tab[2], "sendText", text], capture_output=True, text=True, timeout=3)
        if r.returncode:
            if "Security sensitive" in r.stderr:
                return ("Konsole won't let me type into its tabs until “Enable the security sensitive parts of the DBus "
                        "API” is turned on in Konsole's settings, or the session runs inside tmux")
            return f"I couldn't type into that terminal ({(r.stderr or 'unknown error').strip()[:80]})"
        return None

    def waiting_prompt(self):
        """The permission/choice prompt on the session's screen, if one is open."""
        text = self.screen()
        tail = "\n".join(text.splitlines()[-25:])
        return tail if PROMPT_RE.search(tail) else ""

    def options(self):
        """The choices of the open prompt as (list of labels, index of the highlighted one), or ([], None)."""
        lines = self.screen().splitlines()[-30:]
        foot = [i for i, l in enumerate(lines) if re.search(r"enter to confirm|esc to cancel|esc to (go back|interrupt)", l, re.I)]
        body = lines[:foot[-1]] if foot else lines
        cur = next((i for i in range(len(body) - 1, -1, -1) if "❯" in body[i]), None)
        if cur is None:
            return [], None
        strip = lambda l: re.sub(r"^[\s│|]*(❯\s*)?(\d+\.\s+)?", "", l).rstrip(" │|")
        indent = len(re.match(r"^[\s│|]*❯\s*", body[cur]).group(0))
        same = lambda l: l.strip() and "❯" not in l and len(l) - len(l.lstrip(" │|")) == indent
        top = cur
        while top > 0 and same(body[top - 1]):
            top -= 1
        bot = cur
        while bot + 1 < len(body) and same(body[bot + 1]):
            bot += 1
        return [strip(l) for l in body[top:bot + 1]], cur - top

    async def choose(self, answer):
        """Pick an option in the open prompt: an int (1-based), or "yes" / "no" / "always". Returns error or None."""
        opts, cur = await asyncio.to_thread(self.options)
        if not opts:
            return "I can't see the choices on screen"
        if isinstance(answer, int):
            idx = answer - 1
        else:
            pat = {"yes": r"^(yes|allow|trust|approve|continue|proceed|ok)\b", "always": r"don't ask again|always",
                   "no": r"^(no|deny|exit|cancel|reject|don't|decline)\b"}[answer]
            idx = next((i for i, o in enumerate(opts) if re.search(pat, o, re.I)), None)
        if idx is None or not 0 <= idx < len(opts):
            return f"there's no matching choice (the options are: {'; '.join(opts)})"
        self.sent_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        tmux = self.target["tab"][0] == "tmux"
        for _ in range(abs(idx - cur)):
            key = ("Down" if idx > cur else "Up") if tmux else ("\x1b[B" if idx > cur else "\x1b[A")
            if tmux:
                await asyncio.to_thread(subprocess.run, ["tmux", "send-keys", "-t", self.target["tab"][1], key])
            elif err := await asyncio.to_thread(self._type, key):
                return err
            await asyncio.sleep(0.08)
        self.log("claude code: chose", repr(opts[idx]))
        return await asyncio.to_thread(self._type, "\r")

    async def send(self, text, enter=True):
        if not self.target:
            return {"error": "no Claude Code session picked"}
        if not self.target.get("tab"):
            return {"error": "I can't find that session's terminal, so I can't type into it"}
        if _proc_state(self.target["pid"]) == "T":
            return {"error": "that session is paused with Ctrl+Z; bring it back with fg first"}
        self.sent_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        text = re.sub(r"\s*\n\s*", " ", text).strip()
        if err := await asyncio.to_thread(self._type, text):
            return {"error": err}
        if enter:
            await asyncio.sleep(0.35)   # separate keystroke, so the TUI doesn't treat Enter as part of a paste
            await asyncio.to_thread(self._type, "\r")
        self.log("claude code: sent to", self.target["name"], repr(text[:80]))
        return {"sent": True}

    async def key(self, keys):
        """Answer an on-screen prompt ("1", "2", "\\r", "\\x1b" for Esc...)."""
        self.sent_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        return await asyncio.to_thread(self._type, keys)

    async def wait_reply(self, timeout=90, target=None, since=None):
        """Wait for the session to finish its turn. Returns (texts, state): state is done | prompt | working | gone."""
        target, since = target or self.target, since or self.sent_at
        pid, start = target["pid"], time.time()
        went_busy = False
        while time.time() - start < timeout:
            await asyncio.sleep(1.0)
            st = _status(pid)
            went_busy |= st == "busy"
            if st == "gone" or not _proc_state(pid):
                return [], "gone"
            if time.time() - start > 2 and target is self.target and await asyncio.to_thread(self.waiting_prompt):
                return await asyncio.to_thread(_assistant_text_since, target["transcript"], since), "prompt"
            if st == "idle" and (went_busy or time.time() - start > 6):
                await asyncio.sleep(0.5)  # let the transcript flush
                return await asyncio.to_thread(_assistant_text_since, target["transcript"], since), "done"
        return await asyncio.to_thread(_assistant_text_since, target["transcript"], since), "working"

    def status(self):
        return _status(self.target["pid"]) if self.target else "none"


def speechify(text, limit=1500):
    """Markdown from Claude Code -> something Kokoro can read aloud."""
    text = re.sub(r"```.*?```", " (there's some code on screen) ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*\|.*\|\s*$", " ", text, flags=re.M)          # tables
    text = re.sub(r"^\s*(#+|[-*+]|\d+\.)\s+", "", text, flags=re.M)  # headings, bullets
    text = re.sub(r"[*_#>|]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


ORDINALS = {"first": 0, "one": 0, "1": 0, "second": 1, "two": 1, "2": 1, "third": 2, "three": 2, "3": 2,
            "fourth": 3, "four": 3, "4": 3, "fifth": 4, "five": 4, "5": 4}


def match_session(text, sessions):
    """Which session the user means: the session's name or words from its title / project, else an ordinal."""
    low = text.lower()
    squashed = re.sub(r"[^a-z0-9]", "", low)
    best, best_score = None, 0
    words = set(re.findall(r"[a-z]{4,}", low)) - set(ORDINALS) - {
        "that", "this", "session", "claude", "code", "talk", "with", "about", "please", "number", "last"}
    for s in sessions:
        suffix = s["name"].split("-")[-1].lower()
        score = 3 if len(suffix) >= 2 and (f" {suffix} " in f" {low} " or s["name"].lower().replace("-", "") in squashed) else 0
        hay = f'{s["title"]} {Path(s["cwd"]).name} {s["last_prompt"]}'.lower()
        score += sum(1 for w in words if w in hay)
        if score > best_score:
            best, best_score = s, score
    if best:
        return best
    if re.search(r"\blast\b", low) and sessions:
        return sessions[-1]
    m = re.search(r"\b(first|second|third|fourth|fifth|one|two|three|four|five|[1-5])\b", low)
    if m and len(sessions) > ORDINALS[m.group(1)]:
        return sessions[ORDINALS[m.group(1)]]
    return None
