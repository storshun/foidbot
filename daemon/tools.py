"""Tools the assistant can call. Every tool returns a dict; `sources` entries end up as links in her bubble."""
import asyncio
import re
from datetime import datetime, timedelta, timezone

import time
from calendar import timegm
from urllib.parse import quote_plus

import desktop
import feedparser
import httpx
import trafilatura
from config import load, pronouns

CFG = load()
USER, P = CFG["user"]["name"], pronouns(CFG)
CITY = CFG["location"]["name"]
UA = {"User-Agent": "foidbot (+https://github.com/storshun/foidbot)"}
http = httpx.AsyncClient(headers=UA, timeout=15, follow_redirects=True)
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0"}

NEWS_FEEDS = {
    "national": {"NPR": "https://feeds.npr.org/1001/rss.xml", "CBS News": "https://www.cbsnews.com/latest/rss/main",
                 "NBC News": "https://feeds.nbcnews.com/nbcnews/public/news", "ABC News": "https://feeds.abcnews.com/abcnews/topstories"},
    "world": {"BBC": "https://feeds.bbci.co.uk/news/world/rss.xml", "NPR World": "https://feeds.npr.org/1004/rss.xml"},
    # local news: Google News search for the home town (set during onboarding)
    "local": {"Google News": f"https://news.google.com/rss/search?q={quote_plus(CITY)}&hl=en-US&gl=US&ceid=US:en"},
    "business": {"CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"},
    "tech": {"Ars Technica": "https://feeds.arstechnica.com/arstechnica/index", "The Verge": "https://www.theverge.com/rss/index.xml"},
    "sports": {"ESPN": "https://www.espn.com/espn/rss/news"},
}


async def page_text(url, limit=7000):
    r = await http.get(url, headers=BROWSER_UA, timeout=8)
    text = await asyncio.to_thread(trafilatura.extract, r.text, include_tables=True) or ""
    return text[:limit]


class Tools:
    def __init__(self, cfg):
        self.cfg = cfg
        self.searx = cfg["search"]["searxng"]
        self.lat, self.lon = cfg["location"]["lat"], cfg["location"]["lon"]
        self._nws_point = None
        self._home_nws = None    # does the NWS cover home? (None = not known yet)
        self.browser = None      # set by the daemon
        self.timers = None
        self.ask_vision = None
        self.last_user = ("", 0.0)  # (text, time) of the user's latest message, for confirmations

    # ---- schema the model sees ------------------------------------------------
    SCHEMA = [
        {"name": "web_search", "description": "Search the web. Use for anything current or that you need to verify.",
         "parameters": {"type": "object", "properties": {
             "query": {"type": "string"},
             "category": {"type": "string", "enum": ["general", "news"], "description": "news for current events"},
             "time_range": {"type": "string", "enum": ["any", "day", "week", "month"]}},
             "required": ["query"]}},
        {"name": "news", "description": "Latest headlines (last 36 hours) from major outlets. Use for current events and 'what's going on'.",
         "parameters": {"type": "object", "properties": {
             "section": {"type": "string", "enum": ["national", "world", "local", "business", "tech", "sports"],
                         "description": f"local = around {CITY}"},
             "topic": {"type": "string", "description": "optional filter, e.g. 'hurricane' or 'Astros'"}}}},
        {"name": "read_page", "description": "Read the main text of a web page, to confirm details from a search result.",
         "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
        {"name": "encyclopedia", "description": "Look up a topic on Wikipedia (summary of the article).",
         "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}},
        {"name": "stock_quote", "description": "Live stock, ETF, index, crypto or forex quotes by ticker symbol (e.g. AAPL, ^GSPC, BTC-USD).",
         "parameters": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"}}},
                        "required": ["symbols"]}},
        {"name": "business_hours", "description": "Find a business or place and its opening hours, address and phone.",
         "parameters": {"type": "object", "properties": {
             "name": {"type": "string"},
             "near": {"type": "string", "description": "town or area; defaults to home"}},
             "required": ["name"]}},
        {"name": "weather", "description": "Local weather (National Weather Service in the US, Open-Meteo elsewhere).",
         "parameters": {"type": "object", "properties": {
             "kind": {"type": "string", "enum": ["current", "forecast", "hourly", "alerts", "discussion"]},
             "place": {"type": "string", "description": "only if asking about somewhere other than home"}},
             "required": ["kind"]}},
        # ---- the assistant's own browser (separate profile; the user can watch it)
        {"name": "browse_open", "description": "Open a web page in YOUR browser window to interact with it (forms, buttons, sites that need clicking). Returns numbered elements.",
         "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
        {"name": "browse_look", "description": "Re-read the current page in your browser: numbered elements and text.",
         "parameters": {"type": "object", "properties": {}}},
        {"name": "browse_click", "description": "Click a numbered element in your browser.",
         "parameters": {"type": "object", "properties": {"ref": {"type": "integer"}}, "required": ["ref"]}},
        {"name": "browse_type", "description": "Type into a numbered field in your browser; submit=true presses Enter.",
         "parameters": {"type": "object", "properties": {"ref": {"type": "integer"}, "text": {"type": "string"},
                                                         "submit": {"type": "boolean"}}, "required": ["ref", "text"]}},
        {"name": "browse_scroll", "description": "Scroll your browser page.",
         "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["down", "up"]}}}},
        {"name": "browse_back", "description": "Go back a page in your browser.", "parameters": {"type": "object", "properties": {}}},
        {"name": "browser_confirm", "description": f"Carry out the browser action {USER} just said yes to. Only after {USER} clearly agreed.",
         "parameters": {"type": "object", "properties": {}}},
        # ---- desktop
        {"name": "open_app", "description": f"Launch an application on {USER}'s computer.",
         "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
        {"name": "open_link", "description": f"Open a URL in {USER}'s own browser so {P['subj']} can see it.",
         "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
        {"name": "media", "description": "Control whatever music or video is playing, or see what's playing.",
         "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["status", "play", "pause", "toggle", "next", "previous", "stop"]}},
                        "required": ["action"]}},
        {"name": "volume", "description": "Get or set the computer's speaker volume.",
         "parameters": {"type": "object", "properties": {"level": {"type": "integer", "description": "0-100"},
                                                         "change": {"type": "integer", "description": "e.g. 10 or -10"},
                                                         "mute": {"type": "boolean"}}}},
        {"name": "set_timer", "description": "Start a countdown timer.",
         "parameters": {"type": "object", "properties": {"minutes": {"type": "number"}, "label": {"type": "string", "description": "what it's for, e.g. laundry"}},
                        "required": ["minutes", "label"]}},
        {"name": "set_reminder", "description": f"Remind {USER} of something later. Give minutes from now, or at (a local time like '5:30 pm' or ISO date-time).",
         "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "minutes": {"type": "number"},
                                                         "at": {"type": "string"}}, "required": ["text"]}},
        {"name": "list_reminders", "description": "List pending timers and reminders.", "parameters": {"type": "object", "properties": {}}},
        {"name": "cancel_reminder", "description": "Cancel a timer or reminder by id or by words in its text.",
         "parameters": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}}}},
        {"name": "take_note", "description": f"Save a note to {USER}'s notes (today's note). Keep {P['poss']} wording, including names.",
         "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "title": {"type": "string"}}, "required": ["text"]}},
        {"name": "search_notes", "description": f"Search {USER}'s notes.",
         "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        {"name": "clipboard", "description": f"Read {USER}'s clipboard, or copy text to it.",
         "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["read", "write"]}, "text": {"type": "string"}},
                        "required": ["action"]}},
        {"name": "look_at_screen", "description": f"Look at {USER}'s screen to answer a question about what's on it.",
         "parameters": {"type": "object", "properties": {"question": {"type": "string"},
                                                         "scope": {"type": "string", "enum": ["window", "screen", "everything"],
                                                                   "description": f"window = {P['poss']} active window (default)"}},
                        "required": ["question"]}},
    ]

    GROUPS = {
        "lookup": ["web_search", "news", "read_page", "encyclopedia", "stock_quote", "business_hours", "weather"],
        "browser": ["browse_open", "browse_look", "browse_click", "browse_type", "browse_scroll", "browse_back",
                    "browser_confirm", "web_search"],
        "computer": ["open_app", "open_link", "media", "volume", "set_timer", "set_reminder", "list_reminders",
                     "cancel_reminder", "take_note", "search_notes", "clipboard", "look_at_screen"],
    }

    def schema(self, group=None):
        names = self.GROUPS.get(group)
        return [{"type": "function", "function": f} for f in self.SCHEMA if names is None or f["name"] in names]

    async def call(self, name, args):
        fn = getattr(self, f"t_{name}", None)
        if not fn:
            return {"error": f"unknown tool {name}"}
        try:
            return await fn(**args)
        except Exception as e:  # tools must never crash the conversation
            return {"error": f"{type(e).__name__}: {e}"}

    # ---- tools ---------------------------------------------------------------
    async def t_web_search(self, query, category="general", time_range="any"):
        params = {"q": query, "format": "json", "categories": category}
        if time_range and time_range != "any":
            params["time_range"] = time_range
        r = await http.get(f"{self.searx}/search", params=params)
        res = r.json().get("results", [])[:6]
        items = [{"title": x.get("title"), "url": x.get("url"), "snippet": (x.get("content") or "")[:300],
                  "date": x.get("publishedDate")} for x in res]
        # read the top two pages so answers come from real text, not snippets
        texts = await asyncio.gather(*(page_text(i["url"], 2500) for i in items[:2]), return_exceptions=True)
        for i, t in zip(items, texts):
            if isinstance(t, str) and t:
                i["page_text"] = t
        return {"results": items, "sources": [{"title": i["title"], "url": i["url"]} for i in items[:3]]}

    async def t_read_page(self, url):
        text = await page_text(url)
        return {"url": url, "text": text or "(no readable text)", "sources": [{"title": url, "url": url}]}

    async def t_news(self, section="national", topic=None):
        feeds = NEWS_FEEDS.get(section, NEWS_FEEDS["national"])

        async def grab(src, url):
            try:
                d = feedparser.parse((await http.get(url, headers=BROWSER_UA, timeout=8)).content)
            except Exception:
                return []
            out = []
            for e in d.entries[:15]:
                pp = e.get("published_parsed") or e.get("updated_parsed")
                ts = timegm(pp) if pp else 0
                out.append({"source": src, "title": e.get("title"), "url": e.get("link"), "ts": ts,
                            "summary": re.sub(r"<[^>]+>", "", e.get("summary", ""))[:250]})
            return out
        items = [x for lst in await asyncio.gather(*(grab(k, u) for k, u in feeds.items())) for x in lst]
        cutoff = time.time() - 36 * 3600
        items = [x for x in items if x["ts"] == 0 or x["ts"] > cutoff]
        if topic:
            words = [w for w in re.findall(r"\w+", topic.lower()) if len(w) > 2]
            items = [x for x in items if any(w in (x["title"] + x["summary"]).lower() for w in words)]
        items.sort(key=lambda x: -x["ts"])
        for x in items:
            ts = x.pop("ts")
            x["age"] = f"{(time.time() - ts) / 3600:.0f}h ago" if ts else "?"
        if not items and topic:
            return await self.t_web_search(topic, "news", "week")
        return {"headlines": items[:14], "note": "Stories carried by several outlets are the biggest.",
                "sources": [{"title": f'{x["source"]}: {x["title"]}', "url": x["url"]} for x in items[:3]]}

    async def t_encyclopedia(self, topic):
        s = await http.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": topic, "format": "json", "srlimit": 1})
        hits = s.json()["query"]["search"]
        if not hits:
            return {"error": "no article found"}
        title = hits[0]["title"]
        r = await http.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "prop": "extracts", "explaintext": 1, "exintro": 0, "exchars": 6000,
            "titles": title, "format": "json", "redirects": 1})
        page = next(iter(r.json()["query"]["pages"].values()))
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        return {"title": title, "text": page.get("extract", ""), "sources": [{"title": f"Wikipedia: {title}", "url": url}]}

    async def t_stock_quote(self, symbols):
        import yfinance as yf

        def fetch():
            out = []
            for sym in symbols[:6]:
                t = yf.Ticker(sym)
                fi = t.fast_info
                prev, last = fi.get("previousClose"), fi.get("lastPrice")
                info = {}
                try:
                    info = t.get_info()
                except Exception:
                    pass
                out.append({
                    "symbol": sym.upper(), "name": info.get("shortName"), "price": round(last, 2) if last else None,
                    "change": round(last - prev, 2) if last and prev else None,
                    "change_pct": round((last - prev) / prev * 100, 2) if last and prev else None,
                    "day_low": fi.get("dayLow"), "day_high": fi.get("dayHigh"),
                    "currency": fi.get("currency"), "market_state": info.get("marketState"),
                })
            return out
        quotes = await asyncio.to_thread(fetch)
        return {"quotes": quotes, "as_of": datetime.now().strftime("%-I:%M %p"),
                "sources": [{"title": f"Yahoo Finance: {q['symbol']}", "url": f"https://finance.yahoo.com/quote/{q['symbol']}"} for q in quotes]}

    async def t_business_hours(self, name, near=None):
        near = near or self.cfg["location"]["name"]
        r = await http.get("https://nominatim.openstreetmap.org/search", params={
            "q": f"{name}, {near}", "format": "jsonv2", "extratags": 1, "addressdetails": 1, "limit": 3})
        places = [{"name": p.get("name") or p["display_name"].split(",")[0], "address": p["display_name"],
                   "opening_hours": (p.get("extratags") or {}).get("opening_hours"),
                   "phone": (p.get("extratags") or {}).get("phone"),
                   "website": (p.get("extratags") or {}).get("website")} for p in r.json()]
        web = await self.t_web_search(f"{name} {near} hours")
        return {"openstreetmap": places, "web": web["results"][:4],
                "note": "OpenStreetMap hours can be stale; compare with the web results.",
                "sources": web["sources"][:2]}

    # ---- browser & desktop -------------------------------------------------------------
    async def t_browse_open(self, url):
        return await self.browser.open(url)

    async def t_browse_look(self):
        return await self.browser.look()

    async def t_browse_click(self, ref):
        return await self.browser.click(ref)

    async def t_browse_type(self, ref, text, submit=False):
        return await self.browser.type(ref, text, submit)

    async def t_browse_scroll(self, direction="down"):
        return await self.browser.scroll(direction)

    async def t_browse_back(self):
        return await self.browser.back()

    async def t_browser_confirm(self):
        return await self.browser.confirm(*self.last_user)

    async def t_open_app(self, name):
        return await desktop.open_app(name)

    async def t_open_link(self, url):
        return await desktop.open_link(url)

    async def t_media(self, action="status"):
        return await desktop.media(action)

    async def t_volume(self, level=None, change=None, mute=None):
        return await desktop.volume(level, change, mute)

    async def t_set_timer(self, minutes, label="timer"):
        return await self.timers.add(label, minutes=minutes, kind="timer")

    async def t_set_reminder(self, text, minutes=None, at=None):
        return await self.timers.add(text, minutes=minutes, at=at)

    async def t_list_reminders(self):
        return await self.timers.list()

    async def t_cancel_reminder(self, id=None, text=None):
        return await self.timers.cancel(id, text)

    async def t_take_note(self, text, title=None):
        return await desktop.take_note(text, title)

    async def t_search_notes(self, query):
        return await desktop.search_notes(query)

    async def t_clipboard(self, action, text=""):
        return await (desktop.clipboard_write(text) if action == "write" else desktop.clipboard_read())

    async def t_look_at_screen(self, question, scope="window"):
        return await desktop.look_at_screen(question, scope, self.ask_vision)

    # ---- weather (NWS in the US, Open-Meteo elsewhere) -------------------------
    async def nws_point(self, lat=None, lon=None):
        """NWS grid info for a point, or None when the NWS doesn't cover it (outside the US)."""
        if lat is None and self._nws_point:
            return self._nws_point
        lat, lon = lat or self.lat, lon or self.lon
        r = await http.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")
        if r.status_code in (400, 404):
            if lat == self.lat:
                self._home_nws = False
            return None
        r.raise_for_status()
        p = r.json()["properties"]
        if lat == self.lat:
            self._nws_point, self._home_nws = p, True
        return p

    async def home_nws(self):
        """True/False once known; None if the NWS couldn't be reached (try again later)."""
        if self._home_nws is None:
            try:
                await self.nws_point()
            except Exception:
                return None
        return self._home_nws

    async def open_meteo(self, lat, lon, days=4):
        r = await http.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": days,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,wind_gusts_10m",
            "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m,wind_gusts_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,"
                     "wind_gusts_10m_max"})
        r.raise_for_status()
        return r.json()

    async def weather_open_meteo(self, kind, lat, lon):
        src = [{"title": "Open-Meteo", "url": f"https://open-meteo.com/en/docs#latitude={lat}&longitude={lon}"}]
        if kind in ("alerts", "discussion"):
            return {kind: "not available here: official alerts and forecast discussions come from the US National "
                          "Weather Service, which only covers the US", "sources": src}
        d = await self.open_meteo(lat, lon)
        if kind == "current":
            c = d["current"]
            return {"conditions": WMO.get(c["weather_code"], "unknown"), "temp_c": c["temperature_2m"],
                    "feels_like_c": c["apparent_temperature"], "humidity_pct": c["relative_humidity_2m"],
                    "wind_kmh": c["wind_speed_10m"], "gust_kmh": c["wind_gusts_10m"], "observed": c["time"], "sources": src}
        if kind == "hourly":
            h, now = d["hourly"], d["current"]["time"][:13]
            start = next((i for i, t in enumerate(h["time"]) if t[:13] >= now), 0)
            return {"periods": [{"time": h["time"][i][11:16], "temp_c": h["temperature_2m"][i],
                                 "short": WMO.get(h["weather_code"][i], ""), "rain_pct": h["precipitation_probability"][i],
                                 "wind_kmh": h["wind_speed_10m"][i]} for i in range(start, min(start + 18, len(h["time"])))],
                    "sources": src}
        dl = d["daily"]
        return {"periods": [{"day": datetime.fromisoformat(dl["time"][i]).strftime("%A"),
                             "detail": WMO.get(dl["weather_code"][i], ""), "high_c": dl["temperature_2m_max"][i],
                             "low_c": dl["temperature_2m_min"][i], "rain_pct": dl["precipitation_probability_max"][i],
                             "rain_mm": dl["precipitation_sum"][i], "gusts_kmh": dl["wind_gusts_10m_max"][i]}
                            for i in range(len(dl["time"]))], "sources": src}

    async def geocode(self, place):
        r = await http.get("https://nominatim.openstreetmap.org/search", params={"q": place, "format": "jsonv2", "limit": 1})
        j = r.json()
        return (float(j[0]["lat"]), float(j[0]["lon"])) if j else (None, None)

    async def t_weather(self, kind="current", place=None):
        lat, lon = (await self.geocode(place)) if place else (self.lat, self.lon)
        if lat is None:
            return {"error": f"could not find {place}"}
        try:
            p = await self.nws_point(lat, lon) if place else await self.nws_point()
        except Exception:
            p = None   # NWS down or unreachable: Open-Meteo still answers
        if p is None:
            return await self.weather_open_meteo(kind, lat, lon)
        src = [{"title": "National Weather Service", "url": f"https://forecast.weather.gov/MapClick.php?lat={lat}&lon={lon}"}]
        if kind == "alerts":
            r = await http.get("https://api.weather.gov/alerts/active", params={"point": f"{lat:.4f},{lon:.4f}"})
            alerts = [{"event": a["properties"]["event"], "severity": a["properties"]["severity"],
                       "headline": a["properties"]["headline"], "ends": a["properties"].get("ends") or a["properties"].get("expires"),
                       "instruction": (a["properties"].get("instruction") or "")[:400]} for a in r.json()["features"]]
            return {"alerts": alerts or "none active", "sources": src}
        if kind == "discussion":
            office = p["gridId"]
            r = await http.get(f"https://api.weather.gov/products/types/AFD/locations/{office}")
            latest = r.json()["@graph"][0]["@id"]
            text = (await http.get(latest)).json()["productText"]
            return {"discussion": text[:6000], "sources": src}
        if kind == "current":
            st = (await http.get(p["observationStations"])).json()["features"][0]["properties"]["stationIdentifier"]
            o = (await http.get(f"https://api.weather.gov/stations/{st}/observations/latest")).json()["properties"]
            c2f = lambda c: None if c is None else round(c * 9 / 5 + 32)
            kmh2mph = lambda k: None if k is None else round(k * 0.621)
            return {"station": st, "conditions": o.get("textDescription"), "temp_f": c2f(o["temperature"]["value"]),
                    "feels_like_f": c2f((o.get("heatIndex") or {}).get("value") or (o.get("windChill") or {}).get("value")),
                    "humidity_pct": round(o["relativeHumidity"]["value"] or 0),
                    "wind_mph": kmh2mph(o["windSpeed"]["value"]), "gust_mph": kmh2mph(o["windGust"]["value"]),
                    "observed": o["timestamp"], "sources": src}
        url = p["forecastHourly"] if kind == "hourly" else p["forecast"]
        periods = (await http.get(url)).json()["properties"]["periods"]
        if kind == "hourly":
            periods = [{"time": x["startTime"][11:16], "temp_f": x["temperature"], "short": x["shortForecast"],
                        "rain_pct": (x.get("probabilityOfPrecipitation") or {}).get("value"), "wind": x["windSpeed"]}
                       for x in periods[:18]]
        else:
            periods = [{"name": x["name"], "detail": x["detailedForecast"]} for x in periods[:6]]
        return {"periods": periods, "sources": src}


# WMO weather codes (Open-Meteo)
WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "heavy freezing drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "heavy freezing rain",
       71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
       82: "violent showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
       96: "thunderstorms with hail", 99: "thunderstorms with heavy hail"}


# ---- the always-on weather watcher (the YallBot part) ----------------------------
WARNINGS = {"Tornado Warning", "Tornado Emergency", "Severe Thunderstorm Warning", "Flash Flood Warning",
            "Flash Flood Emergency", "Hurricane Warning", "Storm Surge Warning", "Extreme Wind Warning",
            "Tropical Storm Warning"}
WATCHES = {"Tornado Watch", "Severe Thunderstorm Watch", "Flood Watch", "Flash Flood Watch", "Hurricane Watch",
           "Tropical Storm Watch", "Storm Surge Watch", "Excessive Heat Warning", "Extreme Heat Warning",
           "Heat Advisory", "Freeze Warning", "Hard Freeze Warning", "Winter Storm Warning", "Ice Storm Warning",
           "Dense Fog Advisory", "Air Quality Alert", "Wind Advisory", "High Wind Warning", "Coastal Flood Warning"}


class WeatherWatch:
    """Polls NWS alerts every 3 minutes and scans the next 24 h of the hourly forecast for anything unusual.

    Outside the US (no NWS) there are no official alerts; the forecast scan uses Open-Meteo instead."""

    def __init__(self, tools: Tools, announce):
        self.tools, self.announce = tools, announce  # announce(text, level) is a coroutine
        self.seen_file = __import__("pathlib").Path(__file__).resolve().parent.parent / "data/alerts_seen.txt"
        self.seen = set(self.seen_file.read_text().split()) if self.seen_file.exists() else set()
        self.brief = ""
        self._last_scan = datetime.min.replace(tzinfo=timezone.utc)

    async def run(self):
        while True:
            try:
                nws = await self.tools.home_nws()
                if nws:
                    await self.check_alerts()
                if nws is not None and datetime.now(timezone.utc) - self._last_scan > timedelta(hours=1):
                    await (self.scan_forecast() if nws else self.scan_open_meteo())
            except Exception as e:
                print("weather watch:", e, flush=True)
            await asyncio.sleep(180)

    async def check_alerts(self):
        t = self.tools
        r = await http.get("https://api.weather.gov/alerts/active", params={"point": f"{t.lat:.4f},{t.lon:.4f}"})
        for a in r.json()["features"]:
            pr = a["properties"]
            if pr["id"] in self.seen:
                continue
            self.seen.add(pr["id"])
            self.seen_file.write_text("\n".join(sorted(self.seen)[-200:]))
            ev = pr["event"]
            until = pr.get("ends") or pr.get("expires")
            until_s = datetime.fromisoformat(until).astimezone().strftime("%-I:%M %p") if until else "further notice"
            first_instr = re.split(r"(?<=\.)\s", (pr.get("instruction") or "").strip())[0]
            if ev in WARNINGS:
                await self.announce(f"{self.tools.cfg['user']['name']}, the Weather Service just issued {'an' if ev[0] in 'AEIOU' else 'a'} {ev} for our area "
                                    f"until {until_s}. {first_instr}", "alert")
            elif ev in WATCHES:
                article = "an" if ev[0] in "AEIOU" else "a"
                await self.announce(f"Heads up: there's {article} {ev} for our area until {until_s}.", "concerned")

    async def scan_forecast(self):
        self._last_scan = datetime.now(timezone.utc)
        p = await self.tools.nws_point()
        periods = (await http.get(p["forecastHourly"])).json()["properties"]["periods"][:24]
        grid = (await http.get(p["forecastGridData"])).json()["properties"]
        notes = []
        storms = [x for x in periods if "thunder" in x["shortForecast"].lower()
                  and ((x.get("probabilityOfPrecipitation") or {}).get("value") or 0) >= 40]
        if storms:
            s = storms[0]
            when = datetime.fromisoformat(s["startTime"]).strftime("%-I %p %A")
            notes.append(f"thunderstorms likely starting around {when}")
        if any("severe" in x["shortForecast"].lower() for x in periods):
            notes.append("the forecast mentions severe storms")
        hi = max(x["temperature"] for x in periods)
        lo = min(x["temperature"] for x in periods)
        if hi >= 100:
            notes.append(f"a high near {hi} degrees")
        if lo <= 36:
            notes.append(f"a low near {lo} degrees")
        heat = [v["value"] for v in grid.get("heatIndex", {}).get("values", [])[:24] if v["value"] is not None]
        if heat and max(heat) * 9 / 5 + 32 >= 108:
            notes.append(f"heat index up to {round(max(heat) * 9 / 5 + 32)}")
        gusts = [v["value"] for v in grid.get("windGust", {}).get("values", [])[:24] if v["value"] is not None]
        if gusts and max(gusts) * 0.621 >= 35:
            notes.append(f"wind gusts up to {round(max(gusts) * 0.621)} mph")
        qpf = sum(v["value"] or 0 for v in grid.get("quantitativePrecipitation", {}).get("values", [])[:12]) / 25.4
        if qpf >= 2:
            notes.append(f"heavy rain, about {qpf:.1f} inches")
        await self.report(notes)

    async def scan_open_meteo(self):
        self._last_scan = datetime.now(timezone.utc)
        t = self.tools
        d = await t.open_meteo(t.lat, t.lon, days=2)
        h, now = d["hourly"], d["current"]["time"][:13]   # times are local to the place
        i0 = next((i for i, x in enumerate(h["time"]) if x[:13] >= now), 0)
        rng = range(i0, min(i0 + 24, len(h["time"])))
        notes = []
        storms = [i for i in rng if h["weather_code"][i] >= 95 and (h["precipitation_probability"][i] or 0) >= 40]
        if storms:
            notes.append(f"thunderstorms likely starting around {datetime.fromisoformat(h['time'][storms[0]]).strftime('%-I %p %A')}")
        temps = [h["temperature_2m"][i] for i in rng if h["temperature_2m"][i] is not None]
        if temps and max(temps) >= 38:
            notes.append(f"a high near {round(max(temps))} degrees Celsius")
        if temps and min(temps) <= 2:
            notes.append(f"a low near {round(min(temps))} degrees Celsius")
        gusts = [h["wind_gusts_10m"][i] for i in rng if h["wind_gusts_10m"][i] is not None]
        if gusts and max(gusts) >= 56:
            notes.append(f"wind gusts up to {round(max(gusts))} kilometers an hour")
        rain = sum(h["precipitation"][i] or 0 for i in list(rng)[:12])
        if rain >= 50:
            notes.append(f"heavy rain, about {round(rain)} millimeters")
        await self.report(notes)

    async def report(self, notes):
        brief = "; ".join(notes)
        if brief and brief != self.brief:
            await self.announce(f"Something to keep an eye on in the next day: {brief}.", "concerned")
        self.brief = brief
