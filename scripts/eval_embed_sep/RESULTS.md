# Embedding separation: granite vs bekko

Measured on shared JA (and one EN) query–doc pairs. Higher `margin = related − unrelated` is better separation.

## Per-pair margins

| model | tag | related | unrelated | margin |
|-------|-----|--------:|----------:|-------:|
| granite-97m | named_return | 0.8212 | 0.7192 | +0.1020 |
| granite-97m | continue_subway | 0.8085 | 0.7618 | +0.0467 |
| granite-97m | continue_curry | 0.8018 | 0.7043 | +0.0975 |
| granite-97m | new_vs_topics | 0.6855 | 0.7322 | -0.0467 |
| granite-97m | new_baseball | 0.7128 | 0.7138 | -0.0011 |
| granite-97m | new_tax | 0.7323 | 0.7028 | +0.0295 |
| granite-97m | ambiguous_two_past | 0.7404 | 0.7470 | -0.0066 |
| granite-97m | rank_subway_en | 0.8919 | 0.6877 | +0.2042 |
| bekko-a8m | named_return | 0.3500 | 0.1379 | +0.2121 |
| bekko-a8m | continue_subway | 0.2838 | 0.1787 | +0.1051 |
| bekko-a8m | continue_curry | 0.3173 | 0.0987 | +0.2186 |
| bekko-a8m | new_vs_topics | 0.0709 | 0.1647 | -0.0938 |
| bekko-a8m | new_baseball | 0.0435 | -0.0082 | +0.0518 |
| bekko-a8m | new_tax | 0.0240 | 0.1046 | -0.0806 |
| bekko-a8m | ambiguous_two_past | 0.0507 | 0.1761 | -0.1254 |
| bekko-a8m | rank_subway_en | 0.7459 | 0.0861 | +0.6598 |
| bekko-a25m | named_return | 0.3424 | 0.1165 | +0.2259 |
| bekko-a25m | continue_subway | 0.2861 | 0.2361 | +0.0500 |
| bekko-a25m | continue_curry | 0.2228 | 0.2043 | +0.0185 |
| bekko-a25m | new_vs_topics | 0.0238 | 0.1597 | -0.1359 |
| bekko-a25m | new_baseball | 0.1201 | 0.0510 | +0.0692 |
| bekko-a25m | new_tax | 0.0414 | 0.0763 | -0.0350 |
| bekko-a25m | ambiguous_two_past | 0.0007 | 0.2175 | -0.2168 |
| bekko-a25m | rank_subway_en | 0.7348 | -0.0058 | +0.7406 |

## Summary (mean margin)

| model | positive-ish tags (continue/named/rank) | new_* tags |
|-------|----------------------------------------:|-----------:|
| granite-97m | +0.1126 | -0.0061 |
| bekko-a8m | +0.2989 | -0.0409 |
| bekko-a25m | +0.2587 | -0.0339 |

## new_* max(similarity to either topic) — lower is better for New

| model | tag | max(sim) |
|-------|-----|---------:|
| granite-97m | new_vs_topics | 0.7322 |
| granite-97m | new_baseball | 0.7138 |
| granite-97m | new_tax | 0.7323 |
| bekko-a8m | new_vs_topics | 0.1647 |
| bekko-a8m | new_baseball | 0.0435 |
| bekko-a8m | new_tax | 0.1046 |
| bekko-a25m | new_vs_topics | 0.1597 |
| bekko-a25m | new_baseball | 0.1201 |
| bekko-a25m | new_tax | 0.0763 |

## Product decision (2026-09-06)

- Adopted **bekko-a8m** as the S1 embedder (replaced granite-97m).
- Rationale: related−unrelated margins and New max-sim separation; ONNX S1 suite 14/14 with `S1Thresholds::bekko_calibrated` (0.26 / 0.26 / 0.20).
- a25m not selected (weaker continue margins on this suite; larger).
