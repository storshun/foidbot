"""Settings shared by the daemon, the widget and the bin/ tools: config/foidbot.toml (written by foidbot-onboard)."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config/foidbot.toml"
EXAMPLE = ROOT / "config/foidbot.example.toml"

PRONOUNS = {
    "he": {"subj": "he", "obj": "him", "poss": "his", "refl": "himself"},
    "she": {"subj": "she", "obj": "her", "poss": "her", "refl": "herself"},
    "they": {"subj": "they", "obj": "them", "poss": "their", "refl": "themselves"},
}


def load():
    """The user's config, with any key they don't have filled in from the example."""
    base = tomllib.loads(EXAMPLE.read_text())
    if CONFIG.exists():
        merge(base, tomllib.loads(CONFIG.read_text()))
    return base


def merge(into, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(into.get(k), dict):
            merge(into[k], v)
        else:
            into[k] = v
    return into


def path(p):
    """A config path: ~ expanded, relative paths taken from the repo root."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else ROOT / p


def pronouns(cfg):
    return PRONOUNS.get(cfg["user"].get("pronouns", "they"), PRONOUNS["they"])
