# S1 / deixis evaluation fixtures (bekko ONNX)

Text→embed cases for real `bekko-a8m`. Requires:

- `loco-embed` feature `ort`
- Cached weights (`loco download bekko-a8m` or `LOCO_BEKKO_DIR`)

Pattern coverage (with logic suites): see `../s1/PATTERNS.md`.

## Run

```bash
cargo test -p loco-embed --features ort --test s1_eval_onnx -- --nocapture
# or
mise run eval-s1-onnx
```

Skips when the model is not cached.

## Case kinds

| `kind` | Checks |
|--------|--------|
| `resolve_text` | expand (optional) → embed → `resolve_topic` |
| `embed_rank` | `cos(anchor,closer) > cos(anchor,farther)` (+ optional margin) |

Suites: `ja-bekko-resolve.json` (seed), `ja-bekko-coverage.json` (fill to ≥3/pattern),
`bekko-en-zh-resolve.json` (EN/ZH resolve_text + embed_rank smoke).

Texts and thresholds were calibrated against measured bekko-a8m scores
(`S1Thresholds::bekko_calibrated`).
