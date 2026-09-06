# S1 evaluation patterns

Target: **≥3 distinct cases per pattern** (logic + ONNX).
Machine-readable map: `coverage_map.json`.

| Pattern | Cases |
|---------|------:|
| `clarify.empty_no_match` | 3 |
| `clarify.keyword_ReturnTo` | 3 |
| `clarify.no_match` | 3 |
| `clarify.number_ContinueCurrent` | 3 |
| `clarify.number_ReturnTo` | 3 |
| `clarify.partial_ReturnTo` | 3 |
| `deixis.ContinueHint` | 5 |
| `deixis.Plain` | 3 |
| `deixis.ReturnNamed` | 3 |
| `deixis.ReturnUnspecified` | 5 |
| `deixis.empty_Plain` | 3 |
| `expand.ContinueHint_no_prior` | 3 |
| `expand.ContinueHint_prepend` | 3 |
| `expand.Plain_unchanged` | 3 |
| `expand.ReturnNamed_clean` | 3 |
| `expand.ReturnUnspecified_clean` | 3 |
| `onnx.embed_rank` | 3 |
| `onnx.resolve_text.Continue` | 3 |
| `onnx.resolve_text.Continue_expand` | 3 |
| `onnx.resolve_text.New_empty_past` | 3 |
| `onnx.resolve_text.New_multipast` | 3 |
| `onnx.resolve_text.New_single_past` | 3 |
| `onnx.resolve_text.RU_multipast_Clarify` | 3 |
| `onnx.resolve_text.RU_previous_Return` | 3 |
| `onnx.resolve_text.Return_named` | 3 |
| `resolve.RU.empty_past_current_Continue` | 3 |
| `resolve.RU.empty_past_no_current_New` | 3 |
| `resolve.RU.multipast_Clarify` | 3 |
| `resolve.RU.single_past_Return` | 3 |
| `resolve.RU.with_previous_Return` | 4 |
| `resolve.S1.Continue` | 3 |
| `resolve.S1.ContinueHint` | 3 |
| `resolve.S1.New_no_current` | 3 |
| `resolve.S1.New_weak` | 3 |
| `resolve.S1.Return_named` | 3 |
| `resolve.S1.Return_plain` | 3 |
| `resolve.S1.ambiguous_past_Clarify` | 3 |
| `resolve.S1.gray_S2_Clarify` | 3 |
| `resolve.S1.gray_S2_New` | 3 |

Total patterns: **39**.
Logic fill suite: `ja-s1-coverage.json` (59 cases).
ONNX fill suite: `../s1_onnx/ja-bekko-coverage.json` (13 cases).
