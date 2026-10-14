"""The assistant's own browser: a separate Chromium profile driven through Playwright.

The model sees pages as numbered elements ([12] button "Add to cart") and acts by number.
Risky actions (buy, send, post, delete, sign in, ...) never run straight away: they become a pending
action that only runs after the user says yes in their *next* message. The check is in code, not left to the model.
"""
import asyncio
import os
import re
import time
from pathlib import Path

from playwright.async_api import async_playwright

from config import load, pronouns

CFG = load()
USER, P = CFG["user"]["name"], pronouns(CFG)

PROFILE = Path(__file__).resolve().parent.parent / "data/browser-profile"

RISKY = re.compile(
    r"\b(buy|purchase|order|checkout|check out|pay|payment|place|subscribe|donate|send|post|publish|reply|tweet|"
    r"comment|submit|delete|remove|cancel (my|subscription|order|account)|confirm|transfer|book|reserve|apply|"
    r"sign up|register|unsubscribe|accept|agree|log ?in|sign ?in|share|upload|save changes|update account)\b", re.I)
YES = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok|okay|go ahead|do it|confirm(ed)?|please do|affirmative|absolutely|"
                 r"that's right|correct|proceed)\b", re.I)

SNAPSHOT_JS = r"""
() => {
  document.querySelectorAll('[data-foidbot]').forEach(e => e.removeAttribute('data-foidbot'));
  const sel = 'a[href], button, input:not([type=hidden]), textarea, select, [role=button], [role=link], [role=tab], ' +
              '[role=menuitem], [role=checkbox], [role=radio], [role=combobox], [role=searchbox], [contenteditable=true], summary';
  const vis = e => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);
    return r.width > 2 && r.height > 2 && s.visibility !== 'hidden' && s.display !== 'none' &&
           r.bottom > -200 && r.top < innerHeight + 1200; };
  const out = []; let n = 0;
  for (const e of document.querySelectorAll(sel)) {
    if (!vis(e) || n >= 80) continue;
    n++; e.setAttribute('data-foidbot', n);
    const tag = e.tagName.toLowerCase();
    const role = e.getAttribute('role') || (tag === 'a' ? 'link' : tag === 'input' ? (e.type || 'text') : tag);
    let name = (e.getAttribute('aria-label') || e.innerText || e.value || e.placeholder || e.title || e.alt || '').trim();
    name = name.replace(/\s+/g, ' ').slice(0, 80);
    let extra = '';
    if (e.type === 'checkbox' || e.type === 'radio') extra = e.checked ? ' [CHECKED]' : ' [not checked]';
    else if (tag === 'select') extra = ` selected="${(e.selectedOptions[0] || {}).text || ''}"`;
    else if ((tag === 'input' || tag === 'textarea') && e.value) extra = ` value="${String(e.value).slice(0, 40)}"`;
    if (!name && e.labels && e.labels[0]) name = e.labels[0].innerText.trim().slice(0, 60);
    out.push(`[${n}] ${role} "${name}"${extra}`);
  }
  return out.join('\n');
}
"""


class Browser:
    def __init__(self):
        self.pw = self.ctx = self.page = None
        self.pending = None   # {"desc", "run": coroutine factory, "at": time}
        self.lock = asyncio.Lock()
        self.last_user = ("", 0.0)  # set by the daemon each turn
        self.prev_ask = ""          # the assistant's previous reply
        self.risky_done = False     # a gated action actually ran this turn
        self.task_risky_done = False

    async def ensure(self):
        if self.ctx:
            try:
                if self.page and not self.page.is_closed():
                    return self.page
                self.page = self.ctx.pages[-1] if self.ctx.pages else await self.ctx.new_page()
                return self.page
            except Exception:
                self.ctx = None
        PROFILE.mkdir(parents=True, exist_ok=True)
        self.pw = self.pw or await async_playwright().start()
        self.ctx = await self.pw.chromium.launch_persistent_context(
            str(PROFILE), headless=bool(os.environ.get("FOIDBOT_HEADLESS")), viewport=None, args=["--window-size=1400,1000", "--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"])
        self.ctx.on("close", lambda *_: setattr(self, "ctx", None))
        self.ctx.set_default_timeout(6000)
        self.page = self.ctx.pages[0] if self.ctx.pages else await self.ctx.new_page()
        self.ctx.on("page", self._adopt)  # follow new tabs
        self._adopt(self.page)
        return self.page

    def _adopt(self, page):
        self.page = page
        page.on("crash", lambda *_: setattr(self, "crashed", True))

    async def recover(self):
        """Replace a crashed tab with a fresh one at the same address."""
        url = self.page.url if self.page else ""
        try:
            await self.page.close()
        except Exception:
            pass
        self.crashed = False
        self.page = await self.ctx.new_page()
        self._adopt(self.page)
        if url.startswith("http"):
            try:
                await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception:
                pass

    async def snapshot(self, text_chars=2500):
        if getattr(self, "crashed", False):
            await self.recover()
        pg = self.page
        try:
            await pg.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            pass
        await asyncio.sleep(0.6)
        for _ in range(8):  # bot-check interstitials redirect by themselves after a few seconds
            try:
                head = await pg.evaluate("() => (document.title + ' ' + (document.body ? document.body.innerText.slice(0, 400) : ''))")
            except Exception:
                break
            if not re.search(r"checking your browser|just a moment|pardon our interruption|please wait while we verify|ddos protection", head, re.I):
                break
            await asyncio.sleep(1.5)
        try:
            elements = await pg.evaluate(SNAPSHOT_JS)
        except Exception as e:
            if "crashed" not in str(e).lower() and "closed" not in str(e).lower():
                raise
            await self.recover()
            pg = self.page
            elements = await pg.evaluate(SNAPSHOT_JS)
        text = await pg.evaluate("() => document.body ? document.body.innerText : ''")
        text = re.sub(r"\n\s*\n+", "\n", text)[:text_chars]
        snap = {"url": pg.url, "title": await pg.title(), "elements": elements, "page_text": text}
        if re.search(r"captcha|are you a robot|verify you are human|unusual traffic|select all (squares|images)|press and hold|"
                     r"checking your browser|pardon our interruption|just a moment\.\.\.",
                     text[:3000] + elements[:2000], re.I):
            snap["captcha"] = True
        return snap

    async def with_effect(self, action):
        """Run an action and report what actually changed, so the model can't assume success."""
        before = await self.snapshot(0) if self.page else {"url": "", "elements": ""}
        after = await action()
        if after.get("needs_confirmation") or after.get("error"):
            return after
        if after["url"] != before["url"]:
            effect = f"page changed to {after['url']}"
        else:
            old, new = set(before["elements"].splitlines()), set(after["elements"].splitlines())
            changed = [l for l in after["elements"].splitlines() if l not in old][:6]
            effect = ("changed: " + "; ".join(changed)) if changed else "nothing visible changed"
        return {"effect": effect, **after}

    def _el(self, ref):
        return self.page.locator(f'[data-foidbot="{int(ref)}"]').first

    async def exists(self, ref):
        return await self.page.locator(f'[data-foidbot="{int(ref)}"]').count() > 0

    async def describe(self, ref):
        try:
            return (await self._el(ref).evaluate(
                "e => (e.getAttribute('aria-label') || e.innerText || (e.labels && e.labels[0] && e.labels[0].innerText) || e.placeholder || e.name || e.value || '').trim().slice(0,80)"))
        except Exception:
            return ""

    def already_agreed(self, name):
        """The user's latest message is a yes, and the assistant's message just before it asked about this same action."""
        text, _ = self.last_user
        if not YES.search(text or "") or not self.prev_ask.rstrip().endswith("?"):
            return False
        words = {w.lower() for w in RISKY.findall(name)} | {w.lower() for w in re.findall(r"\w{4,}", name)}
        return any(w in self.prev_ask.lower() for w in words)

    def gate(self, desc, run, name=""):
        if self.already_agreed(name):
            self.risky_done = True
            return None
        self.pending = {"desc": desc, "run": run, "at": time.time()}
        return {"needs_confirmation": True, "action": desc,
                "instruction": f"Everything before this step is done. Ask {USER} to confirm this exact action in one "
                               f"short sentence and stop. If {USER} says yes, call browser_confirm."}

    # ---- tools ---------------------------------------------------------------------------
    async def open(self, url):
        async with self.lock:
            if not re.match(r"^\w+://", url):
                url = "https://" + url
            pg = await self.ensure()
            try:
                await pg.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                await pg.goto("about:blank")
                return {"error": f"couldn't open {url}: {str(e).splitlines()[0][:120]}", "url": "about:blank",
                        "title": "", "elements": "(no page open)", "page_text": ""}
            return await self.snapshot()

    async def look(self):
        async with self.lock:
            await self.ensure()
            return await self.snapshot(6000)

    async def click(self, ref):
        async with self.lock:
            await self.ensure()
            if not await self.exists(ref):
                return {"error": f"there is no element [{ref}] on this page; use a number from the list", **(await self.snapshot())}
            name = await self.describe(ref)

            async def run():
                async def act():
                    await self._el(ref).click(timeout=8000)
                    return await self.snapshot()
                return await self.with_effect(act)
            if RISKY.search(name) and (g := self.gate(f'click "{name}" on {self.page.url}', run, name)):
                return g
            return await run()

    async def type(self, ref, text, submit=False):
        async with self.lock:
            await self.ensure()
            if not await self.exists(ref):
                return {"error": f"there is no element [{ref}] on this page; use a number from the list", **(await self.snapshot())}
            el = self._el(ref)
            kind = (await el.evaluate("e => (e.type || e.getAttribute('role') || e.tagName).toLowerCase()")) or ""
            name = await self.describe(ref)

            async def run():
                async def act():
                    await el.fill(text, timeout=8000)
                    if submit:
                        await el.press("Enter")
                    return await self.snapshot()
                return await self.with_effect(act)
            is_search = "search" in kind or re.search(r"search|find|query|where|zip|location", name, re.I)
            if "password" in kind:
                return {"error": f"I don't type passwords. {USER} can sign in in my browser window."}
            if submit and not is_search:
                submit = False   # forms are submitted with their button, so the confirmation can name it
                res = await run()
                res["note"] = "Typed it without pressing Enter. To send the form, click its submit button."
                return res
            return await run()

    async def scroll(self, direction="down"):
        async with self.lock:
            await self.ensure()
            await self.page.mouse.wheel(0, 900 if direction == "down" else -900)
            return await self.snapshot()

    async def back(self):
        async with self.lock:
            await self.ensure()
            await self.page.go_back()
            return await self.snapshot()

    async def confirm(self, last_user_text, last_user_at):
        p = self.pending
        if not p or time.time() - p["at"] > 180:
            self.pending = None
            return {"error": "nothing is waiting for confirmation"}
        if last_user_at <= p["at"] or not YES.search(last_user_text or ""):
            return {"error": f"{USER} has not said yes to this yet. Ask {P['obj']} and wait for the answer."}
        self.pending = None
        self.risky_done = True
        async with self.lock:
            return {"done": p["desc"], **(await p["run"]())}
