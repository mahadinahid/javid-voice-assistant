# Theme fixtures

- `presets_2026-10-26.json`: what `themes.derive` worked out for the first four presets (Ember, Graphite, Ocean,
  Orchid), their CSS variables and contrast checks, **made** by the code as it was before palette roles and light
  themes existed (commit 99d49a8). `regression.py` checks those four still come out exactly the same, so adding roles
  and light backgrounds can't have changed a theme that uses neither.
