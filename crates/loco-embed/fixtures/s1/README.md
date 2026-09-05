# S1 / deixis evaluation fixtures (logic)

Declarative cases for the logic layer (no ONNX, no LiteRT-LM).

Pattern taxonomy and ≥3-case coverage map: [`PATTERNS.md`](./PATTERNS.md), [`coverage_map.json`](./coverage_map.json).

## Run

```bash
cargo test -p loco-embed --test s1_eval
cargo test -p loco-embed --test s1_coverage
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

Suites: `ja-deixis-s1-logic.json`, `ja-deixis-s1-edge.json`, `ja-s1-coverage.json`,
`en-zh-deixis-expand.json` (EN/ZH deixis + expand + RU resolve).

Real bekko text suites live in `../s1_onnx/` (`mise run eval-s1-onnx`).
