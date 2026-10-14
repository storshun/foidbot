"""foidbot's corner widget: avatar + speech bubble. Talks to the daemon at ws://127.0.0.1:<daemon.port>."""
import html
import json
import math
import random
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, QTimer, QUrl, QFileSystemWatcher
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QPainter, QPainterPath, QPixmap, QRadialGradient, QRegion, QTransform
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWebSockets import QWebSocket
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QLineEdit, QMenu, QVBoxLayout, QWidget

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "daemon"))
from config import load  # noqa: E402

CFG = load()
NAME = CFG["assistant"]["name"]
PORT = CFG["daemon"].get("port", 8770)
AVATAR = ROOT / "assets/avatar"      # avatar.toml says which expression goes with each state/mood
FRAMES = AVATAR / "frames"           # built by bin/foidbot-build-avatar
MOUTH_LEVEL = {"u": "small", "e": "mid", "o": "mid", "a": "wide"}   # viseme -> lip-sync overlay; m/rest = closed
UI =tomllib.loads((ROOT / "config/widget.toml").read_text()) if (ROOT / "config/widget.toml").exists() else {}
AV_W, AV_H = UI.get("avatar_width", 240), UI.get("avatar_height", 320)
BUBBLE_W = 380
WIN_W, WIN_H = BUBBLE_W + AV_W, max(AV_H, 300)
CORNER = UI.get("corner", "bottom-right")
MARGIN = UI.get("margin", 12)

STATE_COLORS = {"idle": "#9b8cff", "listening": "#5fd4ff", "thinking": "#c9a2ff", "working": "#ffd36e",
                "speaking": "#ff9ed8", "sleep": "#5a5a7a"}
MOOD_COLORS = {"alert": "#ff5a5a", "concerned": "#ffb35a", "happy": "#ffd0f0", "playful": "#ff7ab8",
               "wink": "#ff7ab8", "flustered": "#ff8fa0", "sultry": "#d45aa8"}


class Bubble(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("bubble")
        self.setStyleSheet("""
            #bubble { background: rgba(24, 20, 38, 232); border: 1px solid rgba(190, 170, 255, 90); border-radius: 14px; }
            QLabel { color: #f2eeff; font-size: 14px; }
            QLabel#you { color: #a99fcc; font-size: 12px; }
            QLineEdit { background: rgba(255,255,255,18); color: #fff; border: 1px solid rgba(190,170,255,80);
                        border-radius: 8px; padding: 6px; font-size: 13px; }
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(6)
        self.you = QLabel(objectName="you", wordWrap=True)
        self.text = QLabel(wordWrap=True, textFormat=Qt.RichText, openExternalLinks=True)
        self.text.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.input = QLineEdit(placeholderText=f"Type to {NAME}…  (Enter to send, Esc to close)")
        for w in (self.you, self.text, self.input):
            lay.addWidget(w)
        self.input.hide()
        self.reply, self.sources = "", []

    def render(self):
        body = html.escape(self.reply).replace("\n", "<br>")
        if self.sources:
            links = " · ".join(f'<a style="color:#b9a6ff" href="{html.escape(s["url"])}">{html.escape((s.get("title") or s["url"])[:48])}</a>'
                               for s in self.sources)
            body += f'<div style="margin-top:8px; font-size:11px; color:#9a90c0">{links}</div>'
        self.text.setText(body)
        self.text.setVisible(bool(self.reply or self.sources))
        self.adjustSize()


class Widget(QWidget):
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle(NAME)
        self.resize(WIN_W, WIN_H)
        self.state, self.mood, self.viseme = "idle", "calm", "rest"
        self.private, self.mode_setting, self.muted = False, "auto", False
        self.cc_label = None   # Claude Code session she's relaying to
        self.last_reply_at = 0.0
        self.blink_until, self.next_blink = 0.0, time.time() + 3
        self.connected = False
        self.frames, self.manifest, self.no_blink = {}, {}, set()
        self.cur_key, self.base, self.intro, self.intro_until = None, None, None, 0.0
        self.shown, self.prev, self.fade_start, self.fade_len, self.next_drift = None, None, 0.0, 0.15, 0.0
        self.mouth_v, self.mouth_t = "rest", 0.0
        self.load_avatar()
        self.reload = QTimer(self, singleShot=True, interval=1500, timeout=self.load_avatar)
        self.watch = QFileSystemWatcher([str(p) for p in (AVATAR, FRAMES, AVATAR / "avatar.toml") if p.exists()])
        self.watch.directoryChanged.connect(lambda _: self.reload.start())
        self.watch.fileChanged.connect(lambda _: self.reload.start())

        self.bubble = Bubble(self)
        self.bubble.setFixedWidth(BUBBLE_W - 16)
        self.bubble.hide()
        self.bubble.input.returnPressed.connect(self.send_typed)
        self.bubble.input.installEventFilter(self)
        self.hide_timer = QTimer(self, singleShot=True, timeout=self.maybe_hide_bubble)

        self.chime = QSoundEffect(self)
        self.chime.setSource(QUrl.fromLocalFile(str(ROOT / "assets/chime.wav")))
        self.chime.setVolume(0.4)

        self.ws = QWebSocket()
        self.ws.connected.connect(self.on_connected)
        self.ws.disconnected.connect(self.on_disconnected)
        self.ws.textMessageReceived.connect(self.on_message)
        self.reconnect = QTimer(self, interval=2000, timeout=self.try_connect)
        self.reconnect.start()
        self.try_connect()

        self.anim = QTimer(self, interval=33, timeout=self.update)
        self.anim.start()
        self.update_mask()

    # ---- avatar: expression image + lip-sync / blink overlays -------------------
    def load_avatar(self):
        try:
            self.manifest = tomllib.loads((AVATAR / "avatar.toml").read_text())
        except (OSError, tomllib.TOMLDecodeError) as e:
            print("avatar.toml:", e, file=sys.stderr)
            self.manifest = {}
        self.no_blink = set(self.manifest.get("no_blink", []))
        frames, flip = {}, CORNER.endswith("left")
        for p in FRAMES.glob("*.png"):
            pm = QPixmap(str(p))
            if pm.isNull():
                continue
            pm = pm.scaled(AV_W, AV_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if flip:
                pm = pm.transformed(QTransform().scale(-1, 1))
            frames[p.stem] = pm
        self.frames = frames
        self.cur_key = self.shown = self.prev = None
        if hasattr(self, "watch"):   # editors replace files and build-avatar may create frames/, which drops the watch
            watched = self.watch.files() + self.watch.directories()
            for p in (AVATAR, FRAMES, AVATAR / "avatar.toml"):
                if p.exists() and str(p) not in watched:
                    self.watch.addPath(str(p))

    def entry_key(self, now):
        if self.state in ("sleep", "listening", "thinking", "working"):
            return ("state", self.state)
        recent = now - self.last_reply_at < 20
        if self.state == "speaking" or (recent and self.mood != "calm"):
            return ("mood", self.mood)
        return ("state", "idle")

    def entry(self, key):
        m = self.manifest
        return m.get(key[0], {}).get(key[1]) or m.get("mood", {}).get("calm") or m.get("state", {}).get("idle") or {}

    def pick_layers(self, now):
        """Returns (expression name, mouth overlay name or None, blink overlay or None)."""
        key = self.entry_key(now)
        e = self.entry(key)
        if key != self.cur_key:
            self.cur_key = key
            names = [n for n in e.get("base", []) if n in self.frames]
            self.base = random.choice(names) if names else None
            self.intro = e.get("intro") if e.get("intro") in self.frames else None
            self.intro_until = now + e.get("intro_secs", 1.0) if self.intro else 0.0
            self.next_drift = now + random.uniform(25, 60)
        elif key == ("state", "idle") and now > self.next_drift:
            drift = [n for n in e.get("variants", []) if n in self.frames and n != self.base]
            if drift:
                self.base = random.choice(drift)
            self.next_drift = now + random.uniform(25, 60)
        in_intro = now < self.intro_until
        want = self.intro if in_intro else self.base
        if want != self.shown:   # crossfade; slow when she drifts or settles back to idle
            self.fade_len = 0.8 if key == ("state", "idle") and self.shown else 0.15
            self.prev, self.shown, self.fade_start = self.shown, want, now
        # lip sync: hold each mouth shape ~70 ms so fast viseme changes don't strobe
        if self.viseme != self.mouth_v and now - self.mouth_t > 0.07:
            self.mouth_v, self.mouth_t = self.viseme, now
        mouth = None
        fam = e.get("mouth", "neutral")
        if self.state == "speaking" and fam != "none" and not in_intro and self.mouth_v in MOUTH_LEVEL:
            mouth = f"mouth_{fam}_{MOUTH_LEVEL[self.mouth_v]}"
        blink = "blink" if now < self.blink_until and self.shown not in self.no_blink and self.state != "sleep" else None
        return want, mouth, blink

    # ---- painting ------------------------------------------------------------
    def avatar_rect(self):
        x = 0 if CORNER.endswith("left") else WIN_W - AV_W
        return QRect(x, WIN_H - AV_H, AV_W, AV_H)

    def paintEvent(self, _):
        now = time.time()
        if now > self.next_blink:
            self.blink_until = now + 0.14
            self.next_blink = now + random.uniform(2.5, 6.0)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.avatar_rect()
        bob = 0 if self.state == "sleep" else math.sin(now * 2 * math.pi / 4.2) * 2.0
        name, mouth, blink = self.pick_layers(now) if self.frames else (None, None, None)
        if name in self.frames:
            pm = self.frames[name]
            at = QPoint(r.x() + (AV_W - pm.width()) // 2, int(r.bottom() - pm.height() + bob))
            fade = (now - self.fade_start) / self.fade_len
            if fade < 1 and self.prev in self.frames:
                p.drawPixmap(at, self.frames[self.prev])
                p.setOpacity(max(fade, 0.0))
            p.drawPixmap(at, pm)
            p.setOpacity(1.0)
            for overlay in (mouth, blink):
                if overlay in self.frames:
                    p.drawPixmap(at, self.frames[overlay])
        else:
            self.paint_orb(p, r, bob, now)
        if self.cc_label:
            p.setPen(QColor("#d9a066"))
            p.setFont(QFont("sans", 8))
            p.drawText(r.adjusted(0, 4, 0, 0), Qt.AlignHCenter | Qt.AlignTop, "Claude Code: " + self.cc_label[:28])
        if not self.connected:
            p.setPen(QColor("#ff8080"))
            p.setFont(QFont("sans", 9))
            p.drawText(r.adjusted(0, 0, 0, -4), Qt.AlignHCenter | Qt.AlignBottom, "brain offline")

    def paint_orb(self, p, r, bob, now):
        color = QColor(MOOD_COLORS.get(self.mood) if self.state == "speaking" and self.mood in MOOD_COLORS
                       else STATE_COLORS.get(self.state, "#9b8cff"))
        pulse = {"listening": 0.10 * math.sin(now * 6), "thinking": 0.05 * math.sin(now * 3),
                 "working": 0.06 * math.sin(now * 8)}.get(self.state, 0.02 * math.sin(now * 1.5))
        rad = AV_W * 0.30 * (1 + pulse)
        c = QPoint(r.center().x(), int(r.bottom() - AV_W * 0.45 + bob))
        g = QRadialGradient(c, rad * 1.8)
        glow = QColor(color)
        glow.setAlpha(110)
        g.setColorAt(0, glow)
        g.setColorAt(1, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(g)
        p.drawEllipse(c, rad * 1.8, rad * 1.8)
        g2 = QRadialGradient(c - QPoint(int(rad * .3), int(rad * .3)), rad * 1.3)
        g2.setColorAt(0, color.lighter(150))
        g2.setColorAt(1, color.darker(160))
        p.setBrush(g2)
        p.drawEllipse(c, rad, rad)
        # face
        p.setBrush(QColor(30, 20, 50, 220))
        closed = self.state == "sleep" or now < self.blink_until
        for dx in (-rad * .33, rad * .33):
            if closed:
                p.drawRoundedRect(QRectF(c.x() + dx - 7, c.y() - rad * .15, 14, 3), 1.5, 1.5)
            else:
                p.drawEllipse(QPoint(int(c.x() + dx), int(c.y() - rad * .12)), 5, 7)
        open_h = {"rest": 2, "m": 2, "u": 6, "e": 6, "o": 11, "a": 14}[self.viseme]
        open_w = {"rest": 16, "m": 18, "u": 9, "e": 20, "o": 12, "a": 16}[self.viseme]
        p.drawRoundedRect(QRectF(c.x() - open_w / 2, c.y() + rad * .32 - open_h / 2, open_w, open_h), 4, 4)

    # ---- layout / input region -----------------------------------------------
    def place_bubble(self):
        self.bubble.adjustSize()
        h = min(self.bubble.sizeHint().height(), WIN_H - 8)
        self.bubble.resize(BUBBLE_W - 16, h)
        x = AV_W + 8 if CORNER.endswith("left") else 8
        self.bubble.move(x, WIN_H - h - 40)
        self.update_mask()

    def update_mask(self):
        reg = QRegion(self.avatar_rect().adjusted(AV_W // 8, AV_H // 6, -AV_W // 8, 0))
        if self.bubble.isVisible():
            reg = reg.united(QRegion(self.bubble.geometry()))
        self.setMask(reg)

    def show_bubble(self):
        self.bubble.show()
        self.place_bubble()
        self.hide_timer.stop()

    def maybe_hide_bubble(self):
        if self.bubble.underMouse() or self.bubble.input.hasFocus() or self.state in ("speaking", "thinking", "working"):
            self.hide_timer.start(4000)
            return
        self.bubble.hide()
        self.update_mask()

    # ---- daemon link -----------------------------------------------------------
    def try_connect(self):
        if not self.connected:
            self.ws.open(QUrl(f"ws://127.0.0.1:{PORT}"))

    def on_connected(self):
        self.connected = True

    def on_disconnected(self):
        self.connected = False

    def send(self, **msg):
        if self.connected:
            self.ws.sendTextMessage(json.dumps(msg))

    def on_message(self, raw):
        m = json.loads(raw)
        t = m["t"]
        b = self.bubble
        if t == "state":
            self.state = m["state"]
            if self.state == "listening":
                self.mood = "calm"
        elif t == "chime":
            self.chime.play()
        elif t == "mood":
            self.mood = m["mood"]
        elif t == "viseme":
            self.viseme = m["v"]
        elif t == "user":
            b.you.setText("You: " + m["text"])
            b.you.show()
            b.reply, b.sources = "", []
            b.render()
            self.show_bubble()
        elif t == "reply_start":
            b.reply, b.sources = "", []
            self.show_bubble()
        elif t == "reply":
            b.reply += m["delta"]
            b.render()
            self.place_bubble()
        elif t == "reply_end":
            if m.get("sources"):
                b.sources = m["sources"]
                b.render()
                self.place_bubble()
            self.last_reply_at = time.time()
            self.hide_timer.start(15000 + 40 * len(b.reply))
        elif t == "notice":
            b.you.hide()
            b.reply, b.sources = m["text"], []
            b.render()
            self.mood = m.get("level", "concerned")
            self.last_reply_at = time.time()
            self.show_bubble()
            self.hide_timer.start(60000)
        elif t == "mode":
            self.private, self.mode_setting = m["private"], m["setting"]
        elif t == "muted":
            self.muted = m["on"]
        elif t == "cc":
            self.cc_label = m.get("title") or m.get("session")
            self.update()

    # ---- mouse / keyboard ----------------------------------------------------
    def mousePressEvent(self, e):
        if not self.avatar_rect().contains(e.position().toPoint()):
            return
        if e.button() == Qt.LeftButton:
            self.send(t="listen")
        elif e.button() == Qt.RightButton:
            self.menu(e.globalPosition().toPoint())

    def mouseDoubleClickEvent(self, e):
        if self.avatar_rect().contains(e.position().toPoint()):
            self.open_input()

    def open_input(self):
        self.bubble.input.show()
        self.show_bubble()
        self.activateWindow()
        self.bubble.input.setFocus()

    def send_typed(self):
        text = self.bubble.input.text().strip()
        if text:
            self.send(t="text", text=text)
        self.bubble.input.clear()
        self.bubble.input.hide()
        self.place_bubble()

    def eventFilter(self, obj, ev):
        if obj is self.bubble.input and ev.type() == ev.Type.KeyPress and ev.key() == Qt.Key_Escape:
            self.bubble.input.hide()
            self.place_bubble()
            self.hide_timer.start(3000)
            return True
        return False

    def menu(self, pos):
        m = QMenu(self)
        m.addAction("Talk", lambda: self.send(t="listen"))
        m.addAction("Type a message…", self.open_input)
        m.addAction("Stop talking", lambda: self.send(t="stop"))
        if self.cc_label:
            m.addAction("Leave Claude Code mode", lambda: self.send(t="text", text="exit Claude Code mode"))
        elif CFG["claude_code"].get("enabled", True):
            m.addAction("Talk to Claude Code…", lambda: self.send(t="text", text="talk to Claude Code"))
        m.addSeparator()
        modes = m.addMenu(f"Company: {'just us' if self.private else 'others around'}")
        grp = QActionGroup(m)
        for key, label in (("auto", "Detect automatically"), ("private", "Just us"), ("company", "Others are around")):
            a = QAction(label, m, checkable=True, checked=self.mode_setting == key)
            a.triggered.connect(lambda _=False, k=key: self.send(t="mode", setting=k))
            grp.addAction(a)
            modes.addAction(a)
        mute = QAction("Ignore wake word", m, checkable=True, checked=self.muted)
        mute.triggered.connect(lambda on: self.send(t="mute", on=on))
        m.addAction(mute)
        m.addSeparator()
        m.addAction("Hide bubble", lambda: (self.bubble.hide(), self.update_mask()))
        m.addAction("Close widget", QApplication.quit)
        m.exec(pos)


def install_kwin_rule(app, win):
    """Pin the window to a screen corner, above other windows, out of the taskbar (KWin rule, Wayland-safe)."""
    if not shutil.which("kwriteconfig6"):   # not KDE Plasma 6: the window manager places the widget itself
        return
    geo = app.primaryScreen().availableGeometry()
    x = geo.left() + MARGIN if CORNER.endswith("left") else geo.right() - WIN_W - MARGIN + 1
    y = geo.top() + MARGIN if CORNER.startswith("top") else geo.bottom() - WIN_H - MARGIN + 1
    rid = "foidbot-widget"
    group = ["--file", "kwinrulesrc", "--group", rid]
    kw = lambda k, v: subprocess.run(["kwriteconfig6", *group, "--key", k, str(v)], check=False)
    for k, v in {"Description": "foidbot widget", "wmclass": "foidbot", "wmclassmatch": 1, "wmclasscomplete": "false",
                 "position": f"{x},{y}", "positionrule": 2, "above": "true", "aboverule": 2,
                 "skiptaskbar": "true", "skiptaskbarrule": 2, "skippager": "true", "skippagerrule": 2,
                 "skipswitcher": "true", "skipswitcherrule": 2, "noborder": "true", "noborderrule": 2,
                 "desktops": "\\0", "desktopsrule": 2, "acceptfocus": "true"}.items():
        kw(k, v)
    rules = subprocess.run(["kreadconfig6", "--file", "kwinrulesrc", "--group", "General", "--key", "rules"],
                           capture_output=True, text=True).stdout.strip()
    ids = [r for r in rules.split(",") if r]
    if rid not in ids:
        ids.append(rid)
        subprocess.run(["kwriteconfig6", "--file", "kwinrulesrc", "--group", "General", "--key", "rules", ",".join(ids)])
        subprocess.run(["kwriteconfig6", "--file", "kwinrulesrc", "--group", "General", "--key", "count", str(len(ids))])
    subprocess.run(["dbus-send", "--session", "--type=method_call", "--dest=org.kde.KWin", "/KWin", "org.kde.KWin.reconfigure"], capture_output=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setDesktopFileName("foidbot")
    app.setApplicationName("foidbot")
    app.setQuitOnLastWindowClosed(True)
    w = Widget()
    install_kwin_rule(app, w)
    w.show()
    sys.exit(app.exec())
