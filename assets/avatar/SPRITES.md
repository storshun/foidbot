# The avatar

Layered 2D avatar built from the default expression art (the blonde girl in the green-and-black robe).

```
src/<expression>.png      1408x1408 RGBA originals, identical framing
frames/<expression>.png   bust crop 768x1024 (built)
frames/mouth_<family>_<level>.png   lower-face overlays for lip sync (built)
frames/blink.png          closed-eye overlay cut from sleep (built)
avatar.toml               which expression each state / mood shows, lip-sync family, intros, idle drift
```

Rebuild after changing art: `bin/foidbot-build-avatar` (or `bin/foidbot-build-avatar <dir>` to re-import a folder).
The widget hot-reloads `avatar.toml` and `frames/`.

## How it animates

- **Expression** = one full image, chosen from the mood tag the LLM starts each reply with (`[happy]`, `[concerned]`…)
  or from her state (listening, thinking, working, sleep). Alternates in `base` are picked at random for variety.
  Changes crossfade (0.15 s; 0.8 s for idle drift) because every image re-renders the hair slightly.
- **Talking** never swaps the whole image: a lower-face overlay (lips + jaw) goes on top, driven by the daemon's
  visemes. `u` → small, `e`/`o` → mid, `a` → wide, `m`/silence → the expression's own closed mouth.
  Families: neutral (look_away / horrified / shocked), smile (grin_2 / grin / laugh), angry (look_away / rage / yell).
- **Blink** = eyes from `sleep` over the current face, skipped for expressions in `no_blink`.
- **Intros**: amused opens with a laugh, surprised/horrified with shocked, sleep with a yawn.

## Adding art

New images must keep the exact same framing and camera. Drop them in `src/` (snake_case name), reference them
in `avatar.toml`, rebuild. A new mouth family needs open-mouth versions of the same face; add them to
`MOUTHS` in `bin/foidbot-build-avatar`. If the mouth or eyes ever move, adjust `LOWER_FACE` / `EYES` there.
