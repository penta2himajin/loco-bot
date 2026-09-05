# S1 / deixis evaluation fixtures (logic)

Declarative cases for the logic layer (no ONNX, no LiteRT-LM).

## Run

```bash
cargo test -p loco-embed --test s1_eval
# or
mise run eval-s1
```

## Case kinds

| `kind` | Checks |
|--------|--------|
| `deixis` | `classify_deixis(user)` |
| `expand` | `expand_query(user, previous_user)` |
| `resolve` | `resolve_topic` with synthetic embeddings |
| `clarify_match` | `match_clarification` (`expect_no_match` allowed) |

Embeddings may be `"unit": [dim, hot]` or `"vec": [...], "normalize": true`.

Real granite text suites live in `../s1_onnx/` (`mise run eval-s1-onnx`).
