# S1 resolve through-check: bekko a8m / a25m

Fixture: `crates/loco-embed/fixtures/s1_onnx/ja-bekko-resolve.json`

## bekko-a8m

Proposed thresholds: `Th(continue_min=0.264, return_min=0.264, new_max=0.199, clarify_min=0.232, ambiguity_delta=0.05)`

Bands:
- continue_subway: rel=0.2838 unrel=0.1787 max=0.2838
- continue_curry: rel=0.3173 unrel=0.0987 max=0.3173
- named_return: rel=0.3500 unrel=0.1379 max=0.3500
- new_quantum: rel=0.0709 unrel=0.1647 max=0.1647
- new_baseball: rel=0.0435 unrel=-0.0082 max=0.0435
- ambiguous: rel=0.0507 unrel=0.1761 max=0.1761

**Score: 14/14**

| id | status | detail |
|----|--------|--------|
| rank_subway_over_banana | ok | `{"margin": 0.6597763001918793, "expect": "rank"}` |
| rank_curry_over_subway | ok | `{"margin": 0.22490032017230988, "expect": "rank"}` |
| onnx_named_return_subway | ok | `{"deixis": "ReturnNamed", "past_ranked": [[0, 0.3500242829322815]], "current_sim": 0.13791728019714355, "got": {"outc…` |
| onnx_continue_subway | ok | `{"deixis": "Plain", "past_ranked": [], "current_sim": 0.28377705812454224, "got": {"outcome": "Continue", "chunk_inde…` |
| onnx_continue_curry | ok | `{"deixis": "Plain", "past_ranked": [], "current_sim": 0.31734412908554077, "got": {"outcome": "Continue", "chunk_inde…` |
| onnx_continue_deixis_expand | ok | `{"deixis": "ContinueHint", "past_ranked": [], "current_sim": 0.8597021698951721, "got": {"outcome": "Continue", "chun…` |
| onnx_unspecified_stack | ok | `{"deixis": "ReturnUnspecified", "got": {"outcome": "Return", "chunk_index": 0}, "expect": {"outcome": "Return", "chun…` |
| onnx_new_with_past | ok | `{"deixis": "Plain", "past_ranked": [[1, 0.16467204689979553]], "current_sim": 0.07089996337890625, "got": {"outcome":…` |
| onnx_new_baseball_with_past | ok | `{"deixis": "Plain", "past_ranked": [[1, -0.008230462670326233]], "current_sim": 0.043522678315639496, "got": {"outcom…` |
| onnx_new_tax_with_past | ok | `{"deixis": "Plain", "past_ranked": [[0, 0.10456901788711548]], "current_sim": 0.02401411533355713, "got": {"outcome":…` |
| onnx_new_with_two_pasts | ok | `{"deixis": "Plain", "past_ranked": [[1, 0.16467204689979553], [2, 0.11615841835737228]], "current_sim": 0.07089996337…` |
| onnx_new_baseball_two_pasts | ok | `{"deixis": "Plain", "past_ranked": [[0, 0.043522678315639496], [1, -0.008230462670326233]], "current_sim": -0.0057829…` |
| onnx_new_empty_past | ok | `{"deixis": "Plain", "past_ranked": [], "current_sim": 0.07089996337890625, "got": {"outcome": "New", "chunk_index": n…` |
| onnx_ambiguous_past_clarify | ok | `{"deixis": "ReturnUnspecified", "got": {"outcome": "NeedsClarification", "chunk_index": null}, "expect": {"outcome": …` |

## bekko-a25m

Proposed thresholds: `Th(continue_min=0.239, return_min=0.239, new_max=0.219, clarify_min=0.229, ambiguity_delta=0.05)`

Bands:
- continue_subway: rel=0.2861 unrel=0.2361 max=0.2861
- continue_curry: rel=0.2228 unrel=0.2043 max=0.2228
- named_return: rel=0.3424 unrel=0.1165 max=0.3424
- new_quantum: rel=0.0238 unrel=0.1597 max=0.1597
- new_baseball: rel=0.1201 unrel=0.0510 max=0.1201
- ambiguous: rel=0.0007 unrel=0.2175 max=0.2175

**Score: 13/14**

| id | status | detail |
|----|--------|--------|
| rank_subway_over_banana | ok | `{"margin": 0.7405970450490713, "expect": "rank"}` |
| rank_curry_over_subway | ok | `{"margin": 0.11567804217338562, "expect": "rank"}` |
| onnx_named_return_subway | ok | `{"deixis": "ReturnNamed", "past_ranked": [[0, 0.34239476919174194]], "current_sim": 0.1165420264005661, "got": {"outc…` |
| onnx_continue_subway | ok | `{"deixis": "Plain", "past_ranked": [], "current_sim": 0.2860749065876007, "got": {"outcome": "Continue", "chunk_index…` |
| onnx_continue_curry | fail | `{"deixis": "Plain", "past_ranked": [[0, 0.2043268382549286]], "current_sim": 0.22283431887626648, "got": {"outcome": …` |
| onnx_continue_deixis_expand | ok | `{"deixis": "ContinueHint", "past_ranked": [], "current_sim": 0.8399531841278076, "got": {"outcome": "Continue", "chun…` |
| onnx_unspecified_stack | ok | `{"deixis": "ReturnUnspecified", "got": {"outcome": "Return", "chunk_index": 0}, "expect": {"outcome": "Return", "chun…` |
| onnx_new_with_past | ok | `{"deixis": "Plain", "past_ranked": [[1, 0.1596728265285492]], "current_sim": 0.023800237104296684, "got": {"outcome":…` |
| onnx_new_baseball_with_past | ok | `{"deixis": "Plain", "past_ranked": [[1, 0.05097677558660507]], "current_sim": 0.12014716863632202, "got": {"outcome":…` |
| onnx_new_tax_with_past | ok | `{"deixis": "Plain", "past_ranked": [[0, 0.0763496458530426]], "current_sim": 0.04138454794883728, "got": {"outcome": …` |
| onnx_new_with_two_pasts | ok | `{"deixis": "Plain", "past_ranked": [[1, 0.1596728265285492], [2, 0.1264365017414093]], "current_sim": 0.0238002371042…` |
| onnx_new_baseball_two_pasts | ok | `{"deixis": "Plain", "past_ranked": [[0, 0.12014716863632202], [1, 0.05097677558660507]], "current_sim": -0.0039176922…` |
| onnx_new_empty_past | ok | `{"deixis": "Plain", "past_ranked": [], "current_sim": 0.023800237104296684, "got": {"outcome": "New", "chunk_index": …` |
| onnx_ambiguous_past_clarify | ok | `{"deixis": "ReturnUnspecified", "got": {"outcome": "NeedsClarification", "chunk_index": null}, "expect": {"outcome": …` |

Grid best: 14/14 with `Th(continue_min=0.16999999999999998, return_min=0.16999999999999998, new_max=0.05, clarify_min=0.13, ambiguity_delta=0.05)`
