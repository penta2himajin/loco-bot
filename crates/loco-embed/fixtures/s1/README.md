# S1 / deixis evaluation fixtures

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
| `clarify_match` | `match_clarification` |

Embeddings may be `"unit": [dim, hot]` or `"vec": [...], "normalize": true`.

Optional ONNX text suites are out of scope for this directory; add later under `fixtures/s1_onnx/` if needed.
