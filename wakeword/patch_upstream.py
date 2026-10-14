"""Small fixes to the pinned openWakeWord and piper-sample-generator checkouts (run by install.sh).

Exact string replacements so a changed upstream fails loudly instead of half-applying. Safe to run twice.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

FIXES = {
    "openWakeWord/openwakeword/data.py": [
        # DeepPhonemizer's model download is gone; foidbot-train-wake passes the name's phonemes instead
        ("    input_text_phones = [pronouncing.phones_for_word(i) for i in input_text.split()]",
         "    known = __import__('json').loads(__import__('os').environ.get('OWW_EXTRA_PHONES', '{}'))\n"
         "    input_text_phones = [known.get(i.lower()) or pronouncing.phones_for_word(i) for i in input_text.split()]"),
    ],
    "openWakeWord/openwakeword/train.py": [
        # extra feature sets named positive_* (Kokoro clips, the user's own recordings) are positives too
        ('            if key == "positive":', '            if key.startswith("positive"):'),
        # the flag defaults to a truthy string; only convert to tflite when asked
        ("        if args.convert_to_tflite:", "        if args.convert_to_tflite is True:"),
    ],
    "piper-sample-generator/generate_samples.py": [
        # torch >= 2.6 defaults to weights_only=True, which can't load the piper checkpoint
        ("    model = torch.load(model_path)", "    model = torch.load(model_path, weights_only=False)"),
    ],
}


def main():
    for rel, pairs in FIXES.items():
        p = HERE / rel
        text = p.read_bytes().decode()
        for old, new in pairs:
            if new.splitlines()[-1] in text:
                continue
            # some upstream files have CRLF line endings
            hits = [(o, n) for o, n in ((old, new), (old + "\r", new.replace("\n", "\r\n") + "\r")) if o in text]
            if not hits:
                sys.exit(f"patch_upstream: can't find this line in {rel}:\n{old}")
            o, n = hits[-1]
            text = text.replace(o, n, 1)
        p.write_bytes(text.encode())
    print("upstream patches applied")


if __name__ == "__main__":
    main()
