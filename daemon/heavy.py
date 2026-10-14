"""Optional bigger brain for website tasks ([heavy] in config/foidbot.toml, service foidbot-llm-heavy).

Starts only when it is enabled and the GPU has room; stops after 10 idle minutes.
"""
import asyncio
import subprocess
import time
from urllib.parse import urlsplit

UNIT = "foidbot-llm-heavy"
IDLE_S = 600


def free_vram_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.split()[0])
    except Exception:
        return 0


class Heavy:
    def __init__(self, http, log, cfg):
        self.http, self.log = http, log
        h = cfg.get("heavy", {})
        self.enabled = h.get("enabled", False)
        self.url = h.get("url", "http://127.0.0.1:8092/v1/chat/completions")
        u = urlsplit(self.url)
        self.health = f"{u.scheme}://{u.netloc}/health"
        self.needs_mib = int(h.get("min_free_vram_gb", 20.5) * 1000)
        self.last_used = 0.0
        self.lock = asyncio.Lock()

    async def running(self):
        try:
            return (await self.http.get(self.health, timeout=2)).status_code == 200
        except Exception:
            return False

    async def acquire(self):
        """URL of the heavy model if it is (or can be made) available, else None."""
        async with self.lock:
            if not self.enabled:
                return None
            if await self.running():
                self.last_used = time.time()
                return self.url
            free = await asyncio.to_thread(free_vram_mib)
            if free < self.needs_mib:
                self.log(f"heavy brain skipped: {free} MiB free, needs {self.needs_mib}")
                return None
            self.log("starting heavy brain…")
            subprocess.run(["systemctl", "--user", "start", UNIT])
            for _ in range(90):
                if await self.running():
                    self.last_used = time.time()
                    return self.url
                await asyncio.sleep(1)
            subprocess.run(["systemctl", "--user", "stop", UNIT])
            return None

    async def reaper(self):
        while True:
            await asyncio.sleep(60)
            if self.last_used and time.time() - self.last_used > IDLE_S and await self.running():
                self.log("stopping idle heavy brain")
                subprocess.run(["systemctl", "--user", "stop", UNIT])
                self.last_used = 0.0
