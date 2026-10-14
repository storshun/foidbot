"""Turning the assistant's name into what wake-word training needs.

- arpabet(): CMU ARPAbet phonemes for any name ("Schala" -> "SH AA1 L AH0"), from espeak-ng's IPA.
  openWakeWord uses them to invent sound-alike negatives; its own fallback phonemizer's download is dead.
- sound_alikes(): real words that sound close to the name ("scholar", "sheila"...) for "hey <word>" negatives.
Runs in both venvs: espeak-ng comes from espeakng_loader (runtime venv) or piper_phonemize (training venv).
"""
import re

VOWELS = {
    # longest first is handled by the tokenizer below
    "aɪə": "AY ER", "aʊə": "AW ER", "iəɹ": "IH R", "ɛəɹ": "EH R", "ʊəɹ": "UH R",
    "aɪ": "AY", "aʊ": "AW", "eɪ": "EY", "oʊ": "OW", "ɔɪ": "OY", "əʊ": "OW", "iə": "IY AH", "ɛə": "EH", "ʊə": "UH",
    "ɑː": "AA", "ɑ": "AA", "ɒ": "AA", "æ": "AE", "ʌ": "AH", "ə": "AH", "ɐ": "AH", "ɔː": "AO", "ɔ": "AO",
    "ɛ": "EH", "e": "EH", "ɜː": "ER", "ɝ": "ER", "ɚ": "ER", "ɪ": "IH", "ᵻ": "IH", "iː": "IY", "i": "IY",
    "ʊ": "UH", "uː": "UW", "u": "UW", "o": "OW", "a": "AA",
}
CONSONANTS = {
    "tʃ": "CH", "dʒ": "JH", "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G", "g": "G", "f": "F", "v": "V",
    "θ": "TH", "ð": "DH", "s": "S", "z": "Z", "ʃ": "SH", "ʒ": "ZH", "h": "HH", "m": "M", "n": "N", "ŋ": "NG",
    "l": "L", "ɹ": "R", "r": "R", "j": "Y", "w": "W", "ɾ": "T", "x": "K", "ç": "HH", "ɬ": "L", "ʔ": "",
}
TOKENS = sorted({**VOWELS, **CONSONANTS}, key=len, reverse=True)


def ipa(text):
    """espeak-ng IPA with stress marks, from whichever binding this venv has."""
    try:
        import espeakng_loader
        from phonemizer.backend import EspeakBackend
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        return EspeakBackend("en-us", with_stress=True).phonemize([text])[0].strip()
    except ImportError:
        from piper_phonemize import phonemize_espeak
        return " ".join("".join(s) for s in phonemize_espeak(text, "en-us"))


def ipa_to_arpabet(s):
    out, stress, i = [], "0", 0
    while i < len(s):
        c = s[i]
        if c == "ˈ":
            stress, i = "1", i + 1
            continue
        if c == "ˌ":
            stress, i = "2", i + 1
            continue
        tok = next((t for t in TOKENS if s.startswith(t, i)), None)
        if tok is None:
            i += 1              # spaces, length marks on consonants, anything unknown
            continue
        if tok in VOWELS:
            phones = VOWELS[tok].split()
            out.append(phones[0] + stress)
            out += [p + "0" if p in ("AH", "ER") else p for p in phones[1:]]
            stress = "0"
        elif CONSONANTS[tok]:
            out.append(CONSONANTS[tok])
        i += len(tok)
    return " ".join(out)


def arpabet(word):
    """'Schala' -> 'SH AA1 L AH0'. A dictionary word keeps its CMUdict pronunciation when one is available."""
    try:
        import pronouncing
        known = pronouncing.phones_for_word(word.lower())
        if known:
            return known[0]
    except ImportError:
        pass
    return ipa_to_arpabet(ipa(word))


def syllables(phones):
    return sum(p[-1].isdigit() for p in phones.split())


def plain(phones):
    return [re.sub(r"\d", "", p) for p in phones.split()]


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sound_alikes(phones, n=14):
    """Dictionary words one or two phonemes away from the name (not homophones), closest first."""
    import pronouncing
    pronouncing.init_cmu()
    target = plain(phones)
    seen, hits = set(), []
    for word, ph in pronouncing.pronunciations:
        w = re.sub(r"\(\d+\)$", "", word)
        if w in seen or not w.isalpha():
            continue
        p = plain(ph)
        if abs(len(p) - len(target)) > 1:
            continue
        d = edit_distance(p, target)
        if 1 <= d <= 2:
            seen.add(w)
            hits.append((d, len(w), w))
    hits.sort()
    return [w for _, _, w in hits[:n]]


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        ph = arpabet(name)
        print(name, "|", ph, "|", syllables(ph), "syllables")
        try:
            print("  sounds like:", ", ".join(sound_alikes(ph)))
        except ImportError:
            pass
