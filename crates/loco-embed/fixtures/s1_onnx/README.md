# S1 / deixis evaluation fixtures (granite ONNX)

Text→embed cases for real `granite-97m`. Requires:

- `loco-embed` feature `ort`
- Cached weights (`loco download granite-97m` or `LOCO_GRANITE_DIR`)

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

Texts were chosen against measured granite scores with default S1 thresholds.
