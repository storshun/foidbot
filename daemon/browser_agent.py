"""Website tasks as a code-driven loop. Each step the model fills one JSON form (enforced by the server's grammar):
what it sees, then exactly one action. Code runs the action, reports the real effect, and decides when the task
may end. A gated step (buy/send/submit...) stops the loop and the user is asked with a question the code writes.
"""
import json
import re
from urllib.parse import urlparse

from config import load

CFG = load()
USER, ME = CFG["user"]["name"], CFG["assistant"]["name"]

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "seen": {"type": "string", "description": "the facts on the page that matter for the task, briefly"},
        "action": {"type": "string", "enum": ["search", "open", "click", "type", "scroll_down", "scroll_up", "back", "finish", "ask"]},
        "url": {"type": "string"},
        "ref": {"type": "integer"},
        "text": {"type": "string"},
        "submit": {"type": "boolean"},
        "message": {"type": "string", "description": f"for finish/ask: what you say to {USER}, as {ME}: warm, plain, one or two short spoken sentences, no technical jargon"},
    },
    "required": ["seen", "action"],
}

RULES = f"""You operate a web browser for {USER}, one step at a time. Each turn you get the current page as numbered
elements, plus the real effect of your last action. Reply with one JSON step.

- Act only on numbers that appear in the current element list. Checkboxes and radios show [CHECKED] or [not checked].
- One action per step. Do not describe plans; just take the next step.
- search: find pages with a web search (put the query in text); you get result links back. Prefer this to using a
  search engine's own page. open: go to a url (url is required). click: press element ref (also for radios and checkboxes). type: fill element ref with text;
  submit=true presses Enter and is only for search boxes.
- Fill in every part of the request first (all fields, sizes, options, toppings), checking the [CHECKED] marks,
  and click the form's submit/order/send button last.
- finish: the task is complete, or there is nothing more you can do. Your message must describe only what the
  page actually shows. Never say something happened unless the effects or the page prove it.
- ask: you need information only {USER} has (a choice, a detail). Put the question in message.
- Never type passwords or payment card details; use ask so {USER} can enter them.
- Only press buttons that buy, order, send, post, delete or submit when the task explicitly asks for that. {USER} is
  asked to confirm automatically when you do. If the task is to find or report something, never press them.
- As soon as the page shows the answer to the task, finish and give it.
- Never repeat the same action with the same input; if something didn't work, try a different approach."""

RISKY_TASK = re.compile(r"\b(submit|order|buy|purchase|send|post|book|reserve|sign up|subscribe|checkout|check out|pay|delete)\b", re.I)
CLAIM = re.compile(r"\b(submitted|placed|sent|ordered|posted|purchased|bought|booked|went through|signed up|completed the order)\b", re.I)


class BrowserAgent:
    def __init__(self, browser, llm_post, log, pick_url=None, search=None):
        self.b, self.post, self.log, self.pick_url, self.search = browser, llm_post, log, pick_url, search
        self.awaiting = False     # True after an "ask" or a confirmation question: the next message answers it
        self.url = None           # which brain drives this task (heavy 27B when the GPU has room)
        self.task = None          # the task in progress (survives a confirmation pause)
        self.steps = []           # condensed history of this task

    async def step_llm(self, page):
        hist = "\n".join(self.steps[-12:]) or "(no steps yet)"
        user = (f"Task: {self.task}\n\nSteps so far:\n{hist}\n\nCurrent page: {page.get('url', 'none')} "
                f"— {page.get('title', '')}\nElements:\n{page.get('elements', '(no page open)')}\n\n"
                f"Page text (start):\n{page.get('page_text', '')[:1800]}")
        body = {"model": "foidbot", "temperature": 0.2, "max_tokens": 400,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "step", "schema": STEP_SCHEMA}},
                "messages": [{"role": "system", "content": RULES}, {"role": "user", "content": user}]}
        r = await self.post(body, self.url)
        return json.loads(r["choices"][0]["message"]["content"])

    async def run(self, task=None, set_state=None):
        """Returns (message_for_user, finished)."""
        if task:
            self.task, self.steps = task, []
            self.b.task_risky_done = False
        if self.pick_url and (task or not self.url):
            self.url = await self.pick_url()
            self.log("  web brain:", "27B" if self.url else "9B")
        page = {}
        about_current = not task or re.search(r"\b(this|current|the) (page|site|tab|form)\b|\bhere\b", task, re.I)
        if about_current and self.b.ctx and self.b.page:
            try:
                page = await self.b.look()
            except Exception:
                page = {}
        for _ in range(16):
            try:
                st = await self.step_llm(page)
            except Exception as e:
                self.log("browser agent:", e)
                return "Something went wrong while I was working in the browser.", True
            act = st.get("action")
            sig = (act, st.get("url"), st.get("ref"), st.get("text"))
            if act not in ("finish", "ask") and sig == getattr(self, "_last_sig", None):
                self.steps.append(f"- {act}: you just did exactly this; choose a different action")
                self._last_sig = None
                continue
            self._last_sig = sig
            self.log("  web step:", act, {k: st[k] for k in ("url", "ref", "text") if st.get(k) is not None}, "|", st.get("seen", "")[:80])
            if act in ("finish", "ask"):
                msg = (st.get("message") or st.get("seen") or "").strip()
                if (act == "finish" and RISKY_TASK.search(self.task) and not self.b.task_risky_done
                        and CLAIM.search(msg) and not getattr(self, "_corrected", False)):
                    self._corrected = True
                    self.steps.append("CHECK: you said the task went through, but no submit/order/send action has run. "
                                      "Continue the task, or finish with an honest message.")
                    continue
                self._corrected = False
                if act == "finish":
                    self.task = None
                self.awaiting = act == "ask"
                return msg or "I'm done.", act == "finish"
            try:
                if act == "search":
                    found = await self.search(st.get("text") or self.task)
                    self.steps.append(f"- search {st.get('text')!r} -> " + "; ".join(
                        f"{r['title']} <{r['url']}>" for r in found.get("results", [])[:6]))
                    continue
                if act == "open" and not st.get("url"):
                    self.steps.append("- open failed: open needs a url (use search to find one)")
                    continue
                if act == "open":
                    res = await self.b.open(st.get("url") or "")
                elif act == "click":
                    res = await self.b.click(st.get("ref", 0))
                elif act == "type":
                    res = await self.b.type(st.get("ref", 0), st.get("text", ""), bool(st.get("submit")))
                elif act in ("scroll_down", "scroll_up"):
                    res = await self.b.scroll("down" if act == "scroll_down" else "up")
                else:
                    res = await self.b.back()
            except Exception as e:
                res = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            if res.get("captcha"):
                self.steps.append(f"- {act} -> the page is showing a captcha; waiting for {USER} to solve it")
                self.awaiting = True
                host = urlparse(res.get("url", "")).netloc.replace("www.", "")
                return (f"{host or 'The site'} wants a captcha before it'll let me continue. Could you solve it in my "
                        "browser window and then tell me to keep going?"), False
            if res.get("needs_confirmation"):
                host = urlparse(self.b.page.url).netloc.replace("www.", "")
                self.steps.append(f"{act} {st.get('ref')}: waiting for {USER} to confirm ({res['action']})")
                what = re.sub(r" on https?://\S+$", "", res["action"])
                self.awaiting = True
                return f"Before I {what} on {host}, I want your okay. Should I go ahead?", False
            desc = f"{act} {st.get('url') or st.get('ref') or ''} {repr(st.get('text')) if st.get('text') else ''}".strip()
            outcome = res.get("error") or res.get("effect") or f"now on {res.get('url')}"
            self.steps.append(f"- {desc} -> {outcome}")
            page = res if "elements" in res else await self.b.look()
        return "This is taking more steps than I expected. I've stopped where things are; you can see the page in my window.", True

    async def confirmed(self, yes: bool, last_user):
        """The user answered a confirmation question."""
        if not yes:
            self.b.pending = None
            self.task = None
            return "Okay, I won't do that.", True
        res = await self.b.confirm(*last_user)
        if res.get("error"):
            return res["error"], True
        self.b.task_risky_done = True
        self.steps.append(f"- confirmed by {USER}: {res.get('done')} -> {res.get('effect', 'done')}")
        return await self.run()
