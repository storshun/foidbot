#!/usr/bin/env bash
# Stops foidbot and removes what install.sh put outside this folder (services, autostart, command links).
#   ./uninstall.sh          keep this folder (models, your settings and recordings) so you can reinstall later
#   ./uninstall.sh --all    also delete the downloaded models, environments, training data and your data
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pkill -f "$ROOT/widget/widget.py"
for u in foidbot foidbot-train-wake foidbot-llm foidbot-llm-heavy foidbot-aec foidbot-searxng; do
  systemctl --user disable --now "$u" 2>/dev/null
done
rm -f ~/.config/systemd/user/foidbot{,-llm,-llm-heavy,-aec,-searxng}.service \
      ~/.config/containers/systemd/foidbot-searxng.container \
      ~/.config/autostart/foidbot.desktop ~/.local/share/applications/foidbot.desktop
for b in foidbot-ctl foidbot-onboard foidbot-enroll foidbot-train-wake foidbot-build-avatar; do
  [[ -L ~/.local/bin/$b ]] && rm -f ~/.local/bin/$b
done
systemctl --user daemon-reload
python3 - <<'PY'   # drop the widget's KWin window rule (KDE Plasma)
import configparser, pathlib, subprocess
f = pathlib.Path.home() / ".config/kwinrulesrc"
if f.exists():
    c = configparser.RawConfigParser(strict=False); c.optionxform = str; c.read(f)
    if c.has_section("foidbot-widget"):
        c.remove_section("foidbot-widget")
        ids = [r for r in c.get("General", "rules", fallback="").split(",") if r and r != "foidbot-widget"]
        c.set("General", "rules", ",".join(ids)); c.set("General", "count", str(len(ids)))
        with open(f, "w") as fh:
            c.write(fh, space_around_delimiters=False)
        subprocess.run(["dbus-send", "--session", "--type=method_call", "--dest=org.kde.KWin", "/KWin",
                        "org.kde.KWin.reconfigure"], capture_output=True)
PY
if [[ ${1:-} == --all ]]; then
  read -r -p "Delete models, environments, training data AND your settings/recordings in $ROOT? [y/N] " a
  [[ ${a,,} == y* ]] && rm -rf "$ROOT"/{.venv,models,vendor,data,assets/avatar/frames,config/foidbot.toml} \
    "$ROOT"/wakeword/{.venv,openWakeWord,piper-sample-generator,data,work}
fi
echo "foidbot is uninstalled."
