"""Desktop actions: apps, media, volume, timers/reminders, Markdown notes, clipboard, screen reading."""
import asyncio
import base64
import configparser
import difflib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from config import ROOT, load, path as cfg_path

CFG = load()
TIMERS = ROOT / "data/timers.json"
VAULT = cfg_path(CFG["notes"]["dir"])                    # any folder of Markdown files; an Obsidian vault works
NOTES_DIR = VAULT / CFG["notes"].get("folder", CFG["assistant"]["name"])
SCREENSHOT_TOOLS = ("spectacle", "grim", "gnome-screenshot")


def run(*cmd, timeout=5):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


# ---------------------------------------------------------------- apps
def desktop_entries():
    dirs = [Path.home() / ".local/share/applications", Path("/usr/share/applications"),
            Path.home() / ".local/share/flatpak/exports/share/applications", Path("/var/lib/flatpak/exports/share/applications")]
    out = {}
    for d in dirs:
        for f in d.glob("*.desktop") if d.exists() else []:
            cp = configparser.ConfigParser(interpolation=None, strict=False)
            try:
                cp.read(f, encoding="utf-8")
                e = cp["Desktop Entry"]
            except Exception:
                continue
            if e.get("NoDisplay", "false") == "true" or e.get("Type") != "Application":
                continue
            out.setdefault(f.stem, {"name": e.get("Name", f.stem), "generic": e.get("GenericName", ""),
                                    "keywords": e.get("Keywords", "")})
    return out


async def open_app(name):
    apps = await asyncio.to_thread(desktop_entries)
    q = name.lower().strip()
    scored = []
    for app_id, a in apps.items():
        hay = f'{a["name"]} {a["generic"]} {a["keywords"]} {app_id}'.lower()
        score = difflib.SequenceMatcher(None, q, a["name"].lower()).ratio() + (0.6 if q in hay else 0)
        scored.append((score, app_id, a["name"]))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < 0.55:
        return {"error": f"no app matching '{name}'", "closest": [s[2] for s in scored[:3]]}
    _, app_id, label = scored[0]
    subprocess.Popen(["gtk-launch", app_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"opened": label}


async def open_link(url):
    subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"opened_in_users_browser": url}


# ---------------------------------------------------------------- media (MPRIS over D-Bus)
def _players():
    names = run("gdbus", "call", "--session", "--dest", "org.freedesktop.DBus", "--object-path", "/org/freedesktop/DBus",
                "--method", "org.freedesktop.DBus.ListNames")
    return re.findall(r"'(org\.mpris\.MediaPlayer2\.[^']+)'", names)


def _prop(player, prop):
    return run("gdbus", "call", "--session", "--dest", player, "--object-path", "/org/mpris/MediaPlayer2", "--method",
               "org.freedesktop.DBus.Properties.Get", "org.mpris.MediaPlayer2.Player", prop)


async def media(action):
    def go():
        players = _players()
        if not players:
            return {"error": "nothing is playing media right now"}
        status = {p: _prop(p, "PlaybackStatus") for p in players}
        playing = [p for p in players if "Playing" in status[p]]
        target = (playing or [p for p in players if "Paused" in status[p]] or players)[0]
        method = {"play": "Play", "pause": "Pause", "toggle": "PlayPause", "next": "Next", "previous": "Previous",
                  "stop": "Stop"}.get(action)
        if method:
            run("gdbus", "call", "--session", "--dest", target, "--object-path", "/org/mpris/MediaPlayer2",
                "--method", f"org.mpris.MediaPlayer2.Player.{method}")
            time.sleep(0.4)
        meta = _prop(target, "Metadata")
        title = re.search(r"'xesam:title': <'([^']*)'>", meta)
        artist = re.search(r"'xesam:artist': <\['([^']*)'", meta)
        return {"player": target.split(".")[-1].split(".")[0], "status": _prop(target, "PlaybackStatus").strip("(<>',)"),
                "title": title.group(1) if title else None, "artist": artist.group(1) if artist else None}
    return await asyncio.to_thread(go)


async def volume(level=None, change=None, mute=None):
    if mute is not None:
        run("wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1" if mute else "0")
    if level is not None:
        run("wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{max(0, min(100, int(level))) / 100:.2f}")
    if change is not None:
        sign = "+" if change > 0 else "-"
        run("wpctl", "set-volume", "-l", "1.0", "@DEFAULT_AUDIO_SINK@", f"{abs(int(change))}%{sign}")
    v = run("wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@")
    m = re.search(r"([\d.]+)", v)
    return {"volume_pct": round(float(m.group(1)) * 100) if m else None, "muted": "MUTED" in v}


# ---------------------------------------------------------------- timers & reminders
class Timers:
    def __init__(self, announce):
        self.announce = announce
        self.items = json.loads(TIMERS.read_text()) if TIMERS.exists() else []

    def save(self):
        TIMERS.parent.mkdir(exist_ok=True)
        TIMERS.write_text(json.dumps(self.items, indent=1))

    @staticmethod
    def parse_when(minutes=None, at=None):
        now = datetime.now()
        if minutes is not None:
            return now + timedelta(minutes=float(minutes))
        if not at:
            raise ValueError("give minutes or at")
        at = at.strip()
        try:
            t = datetime.fromisoformat(at)
            return t.replace(tzinfo=None) if t.tzinfo is None else t.astimezone().replace(tzinfo=None)
        except ValueError:
            pass
        m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)?", at, re.I)
        if not m:
            raise ValueError(f"can't read time '{at}'")
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
        if ap.startswith("p") and h < 12:
            h += 12
        if ap.startswith("a") and h == 12:
            h = 0
        t = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if t <= now and not ap and h < 12 and (t + timedelta(hours=12)) > now:
            t += timedelta(hours=12)   # "at 5" in the afternoon means 5 pm
        elif t <= now:
            t += timedelta(days=1)
        return t

    async def add(self, text, minutes=None, at=None, kind="reminder"):
        when = self.parse_when(minutes, at)
        item = {"id": uuid.uuid4().hex[:6], "text": text or kind, "kind": kind, "at": when.timestamp()}
        self.items.append(item)
        self.save()
        return {"set": kind, "for": when.strftime("%-I:%M %p %A"), "id": item["id"], "text": item["text"]}

    async def list(self):
        return {"pending": [{"id": i["id"], "text": i["text"], "kind": i["kind"],
                             "at": datetime.fromtimestamp(i["at"]).strftime("%-I:%M %p %a")}
                            for i in sorted(self.items, key=lambda i: i["at"])] or "none"}

    async def cancel(self, id=None, text=None):
        before = len(self.items)
        words = {w for w in re.findall(r"\w+", (text or "").lower()) if w not in {"timer", "reminder", "the", "my", "for", "to"}}
        hit = lambda i: i["id"] == id or (words and words & set(re.findall(r"\w+", i["text"].lower())))
        self.items = [i for i in self.items if not hit(i)]
        self.save()
        return {"cancelled": before - len(self.items)}

    async def run(self):
        while True:
            now = time.time()
            due = [i for i in self.items if i["at"] <= now]
            if due:
                self.items = [i for i in self.items if i["at"] > now]
                self.save()
                for i in due:
                    late = now - i["at"] > 120
                    msg = (f"Your {i['text']} timer is done." if i["kind"] == "timer"
                           else f"Reminder{' (I missed it while I was offline)' if late else ''}: {i['text']}.")
                    await self.announce(msg, "reminder")
            await asyncio.sleep(1)


# ---------------------------------------------------------------- notes (Markdown folder / Obsidian vault)
async def take_note(text, title=None):
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    day = NOTES_DIR / f"{datetime.now():%Y-%m-%d}.md"
    new = not day.exists()
    with day.open("a", encoding="utf-8") as f:
        if new:
            f.write(f"# {datetime.now():%A, %B %-d, %Y}\n\n")
        f.write(f"- {datetime.now():%-I:%M %p}{' **' + title + '**' if title else ''} — {text}\n")
    return {"saved_to": str(day.relative_to(VAULT))}


async def search_notes(query):
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]

    def go():
        hits = []
        for f in VAULT.rglob("*.md") if VAULT.exists() else []:
            if ".obsidian" in f.parts or ".trash" in f.parts:
                continue
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            low = txt.lower()
            score = sum(low.count(w) for w in words) + 5 * sum(w in f.stem.lower() for w in words)
            if score:
                i = min((low.find(w) for w in words if w in low), default=0)
                hits.append((score, str(f.relative_to(VAULT)), txt[max(0, i - 200):i + 600]))
        hits.sort(reverse=True)
        return [{"note": h[1], "excerpt": h[2]} for h in hits[:4]]
    found = await asyncio.to_thread(go)
    return {"matches": found or "nothing found"}


# ---------------------------------------------------------------- clipboard & screen
async def clipboard_read():
    return {"clipboard": run("wl-paste", "--no-newline", timeout=3)[:6000]}


async def clipboard_write(text):
    p = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE, start_new_session=True)
    p.communicate(text.encode())
    return {"copied": text[:80]}


def screenshot_cmd(scope, path):
    """Command for the configured screenshot tool (desktop.screenshot), or the first one installed."""
    want = CFG.get("desktop", {}).get("screenshot", "auto")
    tool = want if want != "auto" else next((t for t in SCREENSHOT_TOOLS if shutil.which(t)), None)
    if not tool or not shutil.which(tool):
        return None
    if tool == "spectacle":
        return ["spectacle", "-b", "-n", {"window": "-a", "screen": "-m", "everything": "-f"}.get(scope, "-a"), "-o", path]
    if tool == "gnome-screenshot":
        return ["gnome-screenshot", *(["-w"] if scope == "window" else []), "-f", path]
    return [tool, path]   # grim (and anything else): the whole desktop


async def look_at_screen(question, scope, ask_vision):
    from PIL import Image
    path = Path(tempfile.gettempdir()) / f"foidbot-{uuid.uuid4().hex[:6]}.png"
    cmd = screenshot_cmd(scope, str(path))
    if not cmd:
        return {"error": "no screenshot tool found; install spectacle, grim or gnome-screenshot "
                         "(or set desktop.screenshot in config/foidbot.toml)"}
    await asyncio.to_thread(subprocess.run, cmd, capture_output=True, timeout=10)
    for _ in range(20):
        if path.exists() and path.stat().st_size:
            break
        await asyncio.sleep(0.2)
    if not path.exists():
        return {"error": "screenshot failed"}
    im = Image.open(path).convert("RGB")
    im.thumbnail((1800, 1800))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    path.unlink(missing_ok=True)
    answer = await ask_vision(question, base64.b64encode(buf.getvalue()).decode())
    return {"what_i_see": answer}
