# S2 LFM spike results (2026-09-06)

Harness: `cases_ja.json` (6 JA Continue/New/Return/Clarify cases).
Floor remains Rust `GraySafetyS2` (no LFM in product path yet).

## LFM2.5-Encoder-350M-Prompt-Router

| Lanes | Score |
|-------|-------|
| English lane text | **2/6** |
| Japanese lane text | **2/6** |

Observed bias: **over-selects Return** (and sometimes Clarify). True Continue/New are weak.
Named / unspecified Return often work; multipast New does not.

## LFM2.5-350M (generative JSON)

| Prompt | Score |
|--------|-------|
| Short EN system | **2/6** (Continue bias) |
| Few-shot EN examples | **2/6** (New/Clarify bias; named Return → Clarify) |

JSON parse mostly OK; decision quality not usable for S2 as-is.

## Conclusion

Neither off-the-shelf LFM router nor LFM-350M instruct is ready for loco S2 on these JA fixtures without fine-tuning or different labeling. Next options (not run here): fine-tune Prompt-Router lanes on loco cases, or keep heuristic S2 + raise S1 floors only.
