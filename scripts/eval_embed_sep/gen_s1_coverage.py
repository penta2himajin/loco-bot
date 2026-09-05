#!/usr/bin/env python3
"""Generate S1 coverage fixtures so each pattern has ≥3 distinct cases.

Writes:
  crates/loco-embed/fixtures/s1/coverage_map.json
  crates/loco-embed/fixtures/s1/ja-s1-coverage.json
  crates/loco-embed/fixtures/s1_onnx/ja-bekko-coverage.json
  crates/loco-embed/fixtures/s1/PATTERNS.md
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S1 = ROOT / "crates/loco-embed/fixtures/s1"
ONNX = ROOT / "crates/loco-embed/fixtures/s1_onnx"

# Existing case ids (from logic + edge + onnx) mapped into the taxonomy.
EXISTING: dict[str, list[str]] = {
    "deixis.Plain": ["deixis_plain_curry", "deixis_plain_subway"],
    "deixis.ContinueHint": [
        "deixis_continue_それ",
        "deixis_continue_なぜ",
        "deixis_continue_詳しく",
        "deixis_あれについて",
        "deixis_bare_続き",
    ],
    "deixis.ReturnNamed": [
        "deixis_named_return",
        "deixis_en_go_back",
        "deixis_named_cooking",
    ],
    "deixis.ReturnUnspecified": [
        "deixis_unspecified_short",
        "deixis_unspecified_戻って",
        "deixis_unspecified_前の話",
        "deixis_en_earlier_unspecified",
        "deixis_前のやつ",
    ],
    "deixis.empty_Plain": [],
    "expand.Plain_unchanged": ["expand_plain_unchanged"],
    "expand.ContinueHint_prepend": [
        "expand_continue_prepends",
        "expand_詳しく_prepends",
    ],
    "expand.ContinueHint_no_prior": ["expand_continue_no_prior"],
    "expand.ReturnUnspecified_clean": ["expand_return_clean"],
    "expand.ReturnNamed_clean": ["expand_named_return_clean"],
    "resolve.RU.with_previous_Return": ["resolve_unspecified_stack_return"],
    "resolve.RU.empty_past_current_Continue": ["resolve_unspecified_no_past_continue"],
    "resolve.RU.empty_past_no_current_New": [],
    "resolve.RU.single_past_Return": ["resolve_unspecified_single_past"],
    "resolve.RU.multipast_Clarify": ["resolve_unspecified_no_stack_clarify"],
    "resolve.S1.Continue": ["resolve_continue_strong_current"],
    "resolve.S1.Return_named": ["resolve_named_return_via_s1"],
    "resolve.S1.Return_plain": ["resolve_return_beats_weak_current"],
    "resolve.S1.ContinueHint": [],
    "resolve.S1.New_weak": ["resolve_new_when_weak"],
    "resolve.S1.New_no_current": ["resolve_no_current_new"],
    "resolve.S1.ambiguous_past_Clarify": ["resolve_ambiguous_past_clarify"],
    "resolve.S1.gray_S2_New": [],
    "resolve.S1.gray_S2_Clarify": [],
    "clarify.number_ReturnTo": ["clarify_by_number"],
    "clarify.number_ContinueCurrent": ["clarify_stay_current"],
    "clarify.keyword_ReturnTo": ["clarify_by_keyword"],
    "clarify.partial_ReturnTo": ["clarify_partial_label"],
    "clarify.no_match": ["clarify_no_match_garbage"],
    "clarify.empty_no_match": [],
    "onnx.embed_rank": ["rank_subway_over_banana", "rank_curry_over_subway"],
    "onnx.resolve_text.Continue": ["onnx_continue_subway", "onnx_continue_curry"],
    "onnx.resolve_text.Continue_expand": ["onnx_continue_deixis_expand"],
    "onnx.resolve_text.New_single_past": [
        "onnx_new_with_past",
        "onnx_new_baseball_with_past",
        "onnx_new_tax_with_past",
    ],
    "onnx.resolve_text.New_multipast": [
        "onnx_new_with_two_pasts",
        "onnx_new_baseball_two_pasts",
    ],
    "onnx.resolve_text.New_empty_past": ["onnx_new_empty_past"],
    "onnx.resolve_text.Return_named": ["onnx_named_return_subway"],
    "onnx.resolve_text.RU_previous_Return": ["onnx_unspecified_stack"],
    "onnx.resolve_text.RU_multipast_Clarify": ["onnx_ambiguous_past_clarify"],
}


def need(pattern: str) -> int:
    return max(0, 3 - len(EXISTING.get(pattern, [])))


LOGIC_NEW: list[dict] = []
ONNX_NEW: list[dict] = []
MAP: dict[str, list[str]] = {k: list(v) for k, v in EXISTING.items()}


def add_logic(pattern: str, case: dict) -> None:
    cid = case["id"]
    LOGIC_NEW.append(case)
    MAP.setdefault(pattern, []).append(cid)


def add_onnx(pattern: str, case: dict) -> None:
    cid = case["id"]
    ONNX_NEW.append(case)
    MAP.setdefault(pattern, []).append(cid)


# --- deixis ---
add_logic(
    "deixis.Plain",
    {
        "kind": "deixis",
        "id": "deixis_plain_weather",
        "user": "明日の天気を一言で",
        "expect_deixis": "Plain",
    },
)
for i, user in enumerate(["   ", "\t", ""], start=1):
    add_logic(
        "deixis.empty_Plain",
        {
            "kind": "deixis",
            "id": f"deixis_empty_plain_{i}",
            "user": user,
            "expect_deixis": "Plain",
        },
    )

# --- expand ---
add_logic(
    "expand.Plain_unchanged",
    {
        "kind": "expand",
        "id": "expand_plain_weather_unchanged",
        "user": "明日の天気を一言で",
        "previous_user": "カレーの作り方を一言",
        "expect_unchanged": True,
    },
)
add_logic(
    "expand.Plain_unchanged",
    {
        "kind": "expand",
        "id": "expand_plain_tax_unchanged",
        "user": "確定申告の基礎を一言",
        "previous_user": "地下鉄について一言",
        "expect_unchanged": True,
    },
)
add_logic(
    "expand.ContinueHint_prepend",
    {
        "kind": "expand",
        "id": "expand_それ_prepends",
        "user": "それについて",
        "previous_user": "明日の天気予報を教えて",
        "expect_expanded_contains": "天気",
    },
)
add_logic(
    "expand.ContinueHint_no_prior",
    {
        "kind": "expand",
        "id": "expand_詳しく_no_prior",
        "user": "もっと詳しく",
        "expect_unchanged": True,
    },
)
add_logic(
    "expand.ContinueHint_no_prior",
    {
        "kind": "expand",
        "id": "expand_それ_no_prior",
        "user": "それについて",
        "expect_unchanged": True,
    },
)
add_logic(
    "expand.ReturnUnspecified_clean",
    {
        "kind": "expand",
        "id": "expand_ru_戻って_clean",
        "user": "さっきの話に戻って",
        "previous_user": "量子コンピュータの基礎を一言で",
        "expect_unchanged": True,
    },
)
add_logic(
    "expand.ReturnUnspecified_clean",
    {
        "kind": "expand",
        "id": "expand_ru_前の話_clean",
        "user": "前の話なんだけど",
        "previous_user": "プロ野球の指名打者制度とは",
        "expect_unchanged": True,
    },
)
add_logic(
    "expand.ReturnNamed_clean",
    {
        "kind": "expand",
        "id": "expand_rn_curry_clean",
        "user": "さっきのカレーの話に戻って",
        "previous_user": "量子コンピュータの基礎を一言で",
        "expect_unchanged": True,
    },
)
add_logic(
    "expand.ReturnNamed_clean",
    {
        "kind": "expand",
        "id": "expand_rn_weather_clean",
        "user": "さっきの天気の話に戻って",
        "previous_user": "カレーの作り方を一言",
        "expect_unchanged": True,
    },
)

# --- resolve RU ---
for i, (user, prev) in enumerate(
    [
        ("さっきの話", 1),
        ("さっきの話に戻って", 0),
        ("前の話なんだけど", 1),
    ],
    start=1,
):
    if need("resolve.RU.with_previous_Return") <= 0 and i > 1:
        # still add until 3 total including existing
        pass
    add_logic(
        "resolve.RU.with_previous_Return",
        {
            "kind": "resolve",
            "id": f"resolve_ru_stack_return_{i}",
            "user": user,
            "query_emb": {"unit": [4, 0]},
            "current": {"unit": [4, 2]},
            "past": [{"index": 0, "unit": [4, 0]}, {"index": 1, "unit": [4, 1]}],
            "previous_chunk": prev,
            "chunk_labels": ["a", "b", "c"],
            "expect": {"outcome": "Return", "chunk_index": prev},
        },
    )
# trim extras if we overshot — keep map accurate later

add_logic(
    "resolve.RU.empty_past_current_Continue",
    {
        "kind": "resolve",
        "id": "resolve_ru_empty_past_continue_2",
        "user": "さっきの話に戻って",
        "query_emb": {"unit": [4, 0]},
        "current": {"unit": [4, 1]},
        "past": [],
        "chunk_labels": ["cur"],
        "expect": {"outcome": "Continue"},
    },
)
add_logic(
    "resolve.RU.empty_past_current_Continue",
    {
        "kind": "resolve",
        "id": "resolve_ru_empty_past_continue_3",
        "user": "前の話なんだけど",
        "query_emb": {"unit": [4, 0]},
        "current": {"unit": [4, 3]},
        "past": [],
        "chunk_labels": ["cur"],
        "expect": {"outcome": "Continue"},
    },
)
for i, user in enumerate(["さっきの話", "さっきの話に戻って", "earlier"], start=1):
    add_logic(
        "resolve.RU.empty_past_no_current_New",
        {
            "kind": "resolve",
            "id": f"resolve_ru_empty_no_current_new_{i}",
            "user": user,
            "query_emb": {"unit": [4, 0]},
            "past": [],
            "chunk_labels": [],
            "expect": {"outcome": "New"},
        },
    )
add_logic(
    "resolve.RU.single_past_Return",
    {
        "kind": "resolve",
        "id": "resolve_ru_single_past_2",
        "user": "さっきの話に戻って",
        "query_emb": {"unit": [4, 1]},
        "current": {"unit": [4, 2]},
        "past": [{"index": 1, "unit": [4, 1]}],
        "chunk_labels": ["x", "only", "now"],
        "expect": {"outcome": "Return", "chunk_index": 1},
    },
)
add_logic(
    "resolve.RU.single_past_Return",
    {
        "kind": "resolve",
        "id": "resolve_ru_single_past_3",
        "user": "前の話なんだけど",
        "query_emb": {"unit": [4, 0]},
        "current": {"unit": [4, 3]},
        "past": [{"index": 0, "unit": [4, 0]}],
        "chunk_labels": ["past0", "x", "y", "now"],
        "expect": {"outcome": "Return", "chunk_index": 0},
    },
)
add_logic(
    "resolve.RU.multipast_Clarify",
    {
        "kind": "resolve",
        "id": "resolve_ru_multipast_clarify_2",
        "user": "さっきの話",
        "query_emb": {"unit": [4, 0]},
        "current": {"unit": [4, 2]},
        "past": [{"index": 0, "unit": [4, 0]}, {"index": 1, "unit": [4, 1]}],
        "chunk_labels": ["alpha", "beta", "now"],
        "expect": {
            "outcome": "NeedsClarification",
            "min_candidates": 3,
            "question_contains": "どの話",
        },
    },
)
add_logic(
    "resolve.RU.multipast_Clarify",
    {
        "kind": "resolve",
        "id": "resolve_ru_multipast_clarify_3",
        "user": "earlier",
        "query_emb": {"unit": [4, 0]},
        "current": {"unit": [4, 2]},
        "past": [
            {"index": 0, "unit": [4, 0]},
            {"index": 1, "unit": [4, 1]},
            {"index": 3, "unit": [4, 3]},
        ],
        "chunk_labels": ["a", "b", "c", "d"],
        "expect": {
            "outcome": "NeedsClarification",
            "min_candidates": 4,
            "question_contains": "どの話",
        },
    },
)

# --- resolve S1 ---
add_logic(
    "resolve.S1.Continue",
    {
        "kind": "resolve",
        "id": "resolve_s1_continue_2",
        "user": "カレーのルーについて",
        "query_emb": {"unit": [4, 2]},
        "current": {"unit": [4, 2]},
        "past": [{"index": 0, "unit": [4, 0]}],
        "chunk_labels": ["old", "x", "curry"],
        "expect": {"outcome": "Continue"},
    },
)
add_logic(
    "resolve.S1.Continue",
    {
        "kind": "resolve",
        "id": "resolve_s1_continue_3",
        "user": "天気の傘について",
        "query_emb": {"unit": [4, 0]},
        "current": {"unit": [4, 0]},
        "past": [{"index": 1, "unit": [4, 1]}],
        "chunk_labels": ["weather", "other"],
        "expect": {"outcome": "Continue"},
    },
)
add_logic(
    "resolve.S1.Return_named",
    {
        "kind": "resolve",
        "id": "resolve_s1_return_named_2",
        "user": "さっきのカレーの話に戻って",
        "query_emb": {"unit": [4, 1]},
        "current": {"unit": [4, 0]},
        "past": [{"index": 1, "unit": [4, 1]}],
        "previous_chunk": 0,
        "chunk_labels": ["weather", "curry"],
        "thresholds": {"continue_min": 0.95, "return_min": 0.9, "new_max": 0.2},
        "expect": {"outcome": "Return", "chunk_index": 1},
    },
)
add_logic(
    "resolve.S1.Return_named",
    {
        "kind": "resolve",
        "id": "resolve_s1_return_named_3",
        "user": "さっきの天気の話に戻って",
        "query_emb": {"unit": [4, 2]},
        "current": {"unit": [4, 0]},
        "past": [{"index": 2, "unit": [4, 2]}],
        "chunk_labels": ["a", "b", "weather"],
        "thresholds": {"continue_min": 0.95, "return_min": 0.9, "new_max": 0.2},
        "expect": {"outcome": "Return", "chunk_index": 2},
    },
)
add_logic(
    "resolve.S1.Return_plain",
    {
        "kind": "resolve",
        "id": "resolve_s1_return_plain_2",
        "user": "地下鉄の切符の話",
        "query_emb": {"unit": [4, 0]},
        "current": {"unit": [4, 3]},
        "past": [{"index": 0, "unit": [4, 0]}],
        "chunk_labels": ["subway", "x", "y", "now"],
        "thresholds": {"continue_min": 0.95, "return_min": 0.9, "new_max": 0.2},
        "expect": {"outcome": "Return", "chunk_index": 0},
    },
)
add_logic(
    "resolve.S1.Return_plain",
    {
        "kind": "resolve",
        "id": "resolve_s1_return_plain_3",
        "user": "カレーのスパイスの話",
        "query_emb": {"unit": [4, 1]},
        "current": {"unit": [4, 3]},
        "past": [{"index": 1, "unit": [4, 1]}],
        "chunk_labels": ["x", "curry", "y", "now"],
        "thresholds": {"continue_min": 0.95, "return_min": 0.9, "new_max": 0.2},
        "expect": {"outcome": "Return", "chunk_index": 1},
    },
)
for i, user in enumerate(["それについて", "なぜ？", "もっと詳しく"], start=1):
    add_logic(
        "resolve.S1.ContinueHint",
        {
            "kind": "resolve",
            "id": f"resolve_s1_continue_hint_{i}",
            "user": user,
            "query_emb": {"unit": [4, 2]},
            "current": {"unit": [4, 2]},
            "past": [{"index": 0, "unit": [4, 0]}],
            "chunk_labels": ["old", "x", "cur"],
            "expect": {"outcome": "Continue"},
        },
    )
add_logic(
    "resolve.S1.New_weak",
    {
        "kind": "resolve",
        "id": "resolve_s1_new_weak_2",
        "user": "量子コンピュータの基礎",
        "query_emb": {"unit": [4, 3]},
        "current": {"unit": [4, 0]},
        "past": [{"index": 1, "unit": [4, 1]}],
        "chunk_labels": ["a", "b", "c", "q"],
        "expect": {"outcome": "New"},
    },
)
add_logic(
    "resolve.S1.New_weak",
    {
        "kind": "resolve",
        "id": "resolve_s1_new_weak_3",
        "user": "確定申告の話",
        "query_emb": {"unit": [4, 2]},
        "current": {"unit": [4, 0]},
        "past": [{"index": 1, "unit": [4, 1]}],
        "chunk_labels": ["a", "b", "tax"],
        "expect": {"outcome": "New"},
    },
)
add_logic(
    "resolve.S1.New_no_current",
    {
        "kind": "resolve",
        "id": "resolve_s1_new_no_current_2",
        "user": "カレーの作り方を一言",
        "query_emb": {"unit": [4, 1]},
        "past": [],
        "chunk_labels": [],
        "expect": {"outcome": "New"},
    },
)
add_logic(
    "resolve.S1.New_no_current",
    {
        "kind": "resolve",
        "id": "resolve_s1_new_no_current_3",
        "user": "天気を教えて",
        "query_emb": {"unit": [4, 0]},
        "past": [{"index": 1, "unit": [4, 1]}],
        "chunk_labels": ["x", "old"],
        "expect": {"outcome": "New"},
    },
)
add_logic(
    "resolve.S1.ambiguous_past_Clarify",
    {
        "kind": "resolve",
        "id": "resolve_s1_ambiguous_2",
        "user": "関連する話に戻したい",
        "query_emb": {"vec": [0.72, 0.69, 0.0, 0.0], "normalize": True},
        "current": {"unit": [4, 3]},
        "past": [
            {"index": 0, "vec": [1.0, 0.1, 0.0, 0.0], "normalize": True},
            {"index": 1, "vec": [0.1, 1.0, 0.0, 0.0], "normalize": True},
        ],
        "chunk_labels": ["alpha", "beta", "gamma"],
        "ambiguity_delta": 0.05,
        "thresholds": {"continue_min": 0.95, "return_min": 0.70, "new_max": 0.20},
        "expect": {"outcome": "NeedsClarification"},
    },
)
add_logic(
    "resolve.S1.ambiguous_past_Clarify",
    {
        "kind": "resolve",
        "id": "resolve_s1_ambiguous_3",
        "user": "前のどれかに戻りたい話",
        "query_emb": {"vec": [0.72, 0.69, 0.0, 0.0], "normalize": True},
        "current": {"unit": [4, 3]},
        "past": [
            {"index": 0, "vec": [1.0, 0.1, 0.0, 0.0], "normalize": True},
            {"index": 1, "vec": [0.1, 1.0, 0.0, 0.0], "normalize": True},
        ],
        "chunk_labels": ["one", "two", "now"],
        "ambiguity_delta": 0.05,
        "thresholds": {"continue_min": 0.95, "return_min": 0.70, "new_max": 0.20},
        "expect": {"outcome": "NeedsClarification"},
    },
)

# gray band: use wide thresholds; GraySafetyS2.clarify_min stays 0.23 (absolute).
# Mid scores ~0.70 → Gray; single past → S2 New; dual close pasts → S2 Clarify.
WIDE_TH = {"continue_min": 0.90, "return_min": 0.90, "new_max": 0.50}
for i in range(1, 4):
    add_logic(
        "resolve.S1.gray_S2_New",
        {
            "kind": "resolve",
            "id": f"resolve_s1_gray_s2_new_{i}",
            "user": f"すこし近い別話題{i}",
            "query_emb": {"vec": [0.7, 0.7, 0.0, 0.0], "normalize": True},
            "current": {"unit": [4, 3]},
            "past": [
                {
                    "index": 0,
                    "vec": [1.0, 0.05, 0.0, 0.0],
                    "normalize": True,
                }
            ],
            "chunk_labels": ["past", "x", "y", "now"],
            "thresholds": WIDE_TH,
            "expect": {"outcome": "New"},
        },
    )

for i in range(1, 4):
    add_logic(
        "resolve.S1.gray_S2_Clarify",
        {
            "kind": "resolve",
            "id": f"resolve_s1_gray_s2_clarify_{i}",
            "user": f"どちらにも少し近い話{i}",
            "query_emb": {"vec": [0.7, 0.7, 0.0, 0.0], "normalize": True},
            "current": {"unit": [4, 3]},
            "past": [
                {"index": 0, "vec": [1.0, 0.05, 0.0, 0.0], "normalize": True},
                {"index": 1, "vec": [0.05, 1.0, 0.0, 0.0], "normalize": True},
            ],
            "chunk_labels": ["p0", "p1", "x", "now"],
            "thresholds": WIDE_TH,
            "ambiguity_delta": 0.05,
            "expect": {"outcome": "NeedsClarification", "question_contains": "どの話"},
        },
    )

# --- clarify ---
cands = [
    {"label": "今の話題のまま", "action": {"kind": "ContinueCurrent"}},
    {"label": "地下鉄について一言", "action": {"kind": "ReturnTo", "chunk_index": 0}},
    {"label": "カレーの作り方", "action": {"kind": "ReturnTo", "chunk_index": 1}},
]
add_logic(
    "clarify.number_ReturnTo",
    {
        "kind": "clarify_match",
        "id": "clarify_by_number_3",
        "user": "3",
        "candidates": cands,
        "expect_action": {"kind": "ReturnTo", "chunk_index": 1},
    },
)
add_logic(
    "clarify.number_ReturnTo",
    {
        "kind": "clarify_match",
        "id": "clarify_by_number_curry",
        "user": "2",
        "candidates": [
            {"label": "天気", "action": {"kind": "ReturnTo", "chunk_index": 2}},
            {"label": "地下鉄", "action": {"kind": "ReturnTo", "chunk_index": 0}},
        ],
        "expect_action": {"kind": "ReturnTo", "chunk_index": 0},
    },
)
add_logic(
    "clarify.number_ContinueCurrent",
    {
        "kind": "clarify_match",
        "id": "clarify_stay_keyword",
        "user": "今の話題",
        "candidates": cands,
        "expect_action": {"kind": "ContinueCurrent"},
    },
)
add_logic(
    "clarify.number_ContinueCurrent",
    {
        "kind": "clarify_match",
        "id": "clarify_stay_number_again",
        "user": "1",
        "candidates": [
            {"label": "このまま", "action": {"kind": "ContinueCurrent"}},
            {"label": "地下鉄", "action": {"kind": "ReturnTo", "chunk_index": 0}},
        ],
        "expect_action": {"kind": "ContinueCurrent"},
    },
)
add_logic(
    "clarify.keyword_ReturnTo",
    {
        "kind": "clarify_match",
        "id": "clarify_by_keyword_curry",
        "user": "カレー",
        "candidates": cands,
        "expect_action": {"kind": "ReturnTo", "chunk_index": 1},
    },
)
add_logic(
    "clarify.keyword_ReturnTo",
    {
        "kind": "clarify_match",
        "id": "clarify_by_keyword_subway2",
        "user": "地下鉄について",
        "candidates": cands,
        "expect_action": {"kind": "ReturnTo", "chunk_index": 0},
    },
)
add_logic(
    "clarify.partial_ReturnTo",
    {
        "kind": "clarify_match",
        "id": "clarify_partial_2",
        "user": "地下",
        "candidates": cands,
        "expect_action": {"kind": "ReturnTo", "chunk_index": 0},
    },
)
add_logic(
    "clarify.partial_ReturnTo",
    {
        "kind": "clarify_match",
        "id": "clarify_partial_3",
        "user": "カレ",
        "candidates": cands,
        "expect_action": {"kind": "ReturnTo", "chunk_index": 1},
    },
)
add_logic(
    "clarify.no_match",
    {
        "kind": "clarify_match",
        "id": "clarify_no_match_zzz",
        "user": "zzzz",
        "candidates": cands,
        "expect_no_match": True,
    },
)
add_logic(
    "clarify.no_match",
    {
        "kind": "clarify_match",
        "id": "clarify_no_match_99",
        "user": "99",
        "candidates": cands,
        "expect_no_match": True,
    },
)
for i, user in enumerate(["", "  ", "\n"], start=1):
    add_logic(
        "clarify.empty_no_match",
        {
            "kind": "clarify_match",
            "id": f"clarify_empty_no_match_{i}",
            "user": user,
            "candidates": cands,
            "expect_no_match": True,
        },
    )

# Deduplicate overshoot for RU.with_previous — keep first 2 new only if existing had 1
# Rebuild MAP cleanly from EXISTING + added, but trim patterns that got >3 from our bulk add
# Actually audit only requires ≥3; >3 is fine.

# --- ONNX ---
add_onnx(
    "onnx.embed_rank",
    {
        "kind": "embed_rank",
        "id": "rank_weather_over_curry",
        "anchor": "明日の天気予報を教えて",
        "closer": "Will I need an umbrella tomorrow?",
        "farther": "カレーの作り方を詳しく教えて",
        "min_margin": 0.05,
    },
)
add_onnx(
    "onnx.resolve_text.Continue",
    {
        "kind": "resolve_text",
        "id": "onnx_continue_subway_home",
        "user": "発車ホームはどこ？",
        "current_text": "東京の地下鉄の切符の買い方",
        "past_texts": [{"index": 1, "text": "明日の天気予報を教えて"}],
        "previous_chunk": 1,
        "chunk_labels": ["地下鉄", "天気"],
        "expect": {"outcome": "Continue"},
    },
)
add_onnx(
    "onnx.resolve_text.Continue_expand",
    {
        "kind": "resolve_text",
        "id": "onnx_continue_expand_curry",
        "user": "なぜ？",
        "previous_user": "カレーの作り方を詳しく教えて",
        "use_expand": True,
        "current_text": "カレーの作り方を詳しく教えて",
        "past_texts": [{"index": 0, "text": "東京の地下鉄の切符の買い方"}],
        "previous_chunk": 0,
        "chunk_labels": ["地下鉄", "カレー"],
        "expect": {"outcome": "Continue"},
    },
)
add_onnx(
    "onnx.resolve_text.Continue_expand",
    {
        "kind": "resolve_text",
        "id": "onnx_continue_expand_weather",
        "user": "もっと詳しく",
        "previous_user": "明日の天気予報を教えて",
        "use_expand": True,
        "current_text": "明日の天気予報を教えて",
        "past_texts": [{"index": 1, "text": "カレーの作り方を詳しく教えて"}],
        "previous_chunk": 1,
        "chunk_labels": ["天気", "カレー"],
        "expect": {"outcome": "Continue"},
    },
)
add_onnx(
    "onnx.resolve_text.New_multipast",
    {
        "kind": "resolve_text",
        "id": "onnx_new_piano_two_pasts",
        "user": "ピアノの調律はどうする",
        "current_text": "明日の天気予報を教えて",
        "past_texts": [
            {"index": 0, "text": "東京の地下鉄の切符の買い方"},
            {"index": 1, "text": "カレーの作り方を詳しく教えて"},
        ],
        "previous_chunk": 0,
        "chunk_labels": ["地下鉄", "カレー", "天気"],
        "expect": {"outcome": "New"},
    },
)
add_onnx(
    "onnx.resolve_text.New_empty_past",
    {
        "kind": "resolve_text",
        "id": "onnx_new_empty_baseball",
        "user": "プロ野球の指名打者制度とは",
        "current_text": "カレーの作り方を詳しく教えて",
        "past_texts": [],
        "chunk_labels": ["カレー"],
        "expect": {"outcome": "New"},
    },
)
add_onnx(
    "onnx.resolve_text.New_empty_past",
    {
        "kind": "resolve_text",
        "id": "onnx_new_empty_iss",
        "user": "国際宇宙ステーションの高さ",
        "current_text": "東京の地下鉄の切符の買い方",
        "past_texts": [],
        "chunk_labels": ["地下鉄"],
        "expect": {"outcome": "New"},
    },
)
add_onnx(
    "onnx.resolve_text.Return_named",
    {
        "kind": "resolve_text",
        "id": "onnx_named_return_curry",
        "user": "さっきのカレーの話に戻って",
        "current_text": "東京の地下鉄の切符の買い方",
        "past_texts": [{"index": 1, "text": "カレーの作り方を詳しく教えて"}],
        "previous_chunk": 0,
        "chunk_labels": ["地下鉄", "カレー"],
        "expect": {"outcome": "Return", "chunk_index": 1},
    },
)
add_onnx(
    "onnx.resolve_text.Return_named",
    {
        "kind": "resolve_text",
        "id": "onnx_named_return_weather",
        "user": "さっきの天気の話に戻って",
        "current_text": "カレーの作り方を詳しく教えて",
        "past_texts": [{"index": 0, "text": "明日の天気予報を教えて"}],
        "previous_chunk": 1,
        "chunk_labels": ["天気", "カレー"],
        "expect": {"outcome": "Return", "chunk_index": 0},
    },
)
add_onnx(
    "onnx.resolve_text.RU_previous_Return",
    {
        "kind": "resolve_text",
        "id": "onnx_ru_stack_return_2",
        "user": "さっきの話に戻って",
        "current_text": "カレーの作り方を詳しく教えて",
        "past_texts": [{"index": 0, "text": "東京の地下鉄の切符の買い方"}],
        "previous_chunk": 0,
        "chunk_labels": ["地下鉄", "カレー"],
        "expect": {"outcome": "Return", "chunk_index": 0},
    },
)
add_onnx(
    "onnx.resolve_text.RU_previous_Return",
    {
        "kind": "resolve_text",
        "id": "onnx_ru_stack_return_3",
        "user": "前の話なんだけど",
        "current_text": "明日の天気予報を教えて",
        "past_texts": [{"index": 1, "text": "カレーの作り方を詳しく教えて"}],
        "previous_chunk": 1,
        "chunk_labels": ["天気", "カレー"],
        "expect": {"outcome": "Return", "chunk_index": 1},
    },
)
add_onnx(
    "onnx.resolve_text.RU_multipast_Clarify",
    {
        "kind": "resolve_text",
        "id": "onnx_ru_multipast_clarify_2",
        "user": "さっきの話",
        "current_text": "明日の天気予報を教えて",
        "past_texts": [
            {"index": 0, "text": "東京の地下鉄の切符の買い方"},
            {"index": 1, "text": "カレーの作り方を詳しく教えて"},
        ],
        "chunk_labels": ["地下鉄", "カレー", "天気"],
        "expect": {
            "outcome": "NeedsClarification",
            "min_candidates": 3,
            "question_contains": "どの話",
        },
    },
)
add_onnx(
    "onnx.resolve_text.RU_multipast_Clarify",
    {
        "kind": "resolve_text",
        "id": "onnx_ru_multipast_clarify_3",
        "user": "earlier",
        "current_text": "明日の天気予報を教えて",
        "past_texts": [
            {"index": 0, "text": "東京の地下鉄の切符の買い方"},
            {"index": 1, "text": "カレーの作り方を詳しく教えて"},
        ],
        "chunk_labels": ["地下鉄", "カレー", "天気"],
        "expect": {
            "outcome": "NeedsClarification",
            "min_candidates": 3,
            "question_contains": "どの話",
        },
    },
)

# Fix overshoot: resolve.RU.with_previous got existing+3; that's ok (≥3).

# Trim MAP for patterns where we added while existing already ≥3 — not needed.

# Verify ≥3
short = {k: v for k, v in MAP.items() if len(v) < 3}
if short:
    raise SystemExit(f"still short: { {k: len(v) for k, v in short.items()} }")

# Write files
logic_suite = {
    "version": 1,
    "name": "ja-s1-coverage",
    "cases": LOGIC_NEW,
}
onnx_suite = {
    "version": 1,
    "name": "ja-bekko-coverage",
    "cases": ONNX_NEW,
}
coverage_map = {
    "version": 1,
    "min_cases_per_pattern": 3,
    "patterns": {k: sorted(set(v)) for k, v in sorted(MAP.items())},
}

(S1 / "ja-s1-coverage.json").write_text(
    json.dumps(logic_suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(ONNX / "ja-bekko-coverage.json").write_text(
    json.dumps(onnx_suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(S1 / "coverage_map.json").write_text(
    json.dumps(coverage_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

lines = [
    "# S1 evaluation patterns",
    "",
    "Target: **≥3 distinct cases per pattern** (logic + ONNX).",
    "Machine-readable map: `coverage_map.json`.",
    "",
    "| Pattern | Cases |",
    "|---------|------:|",
]
for k, ids in coverage_map["patterns"].items():
    lines.append(f"| `{k}` | {len(ids)} |")
lines += [
    "",
    f"Total patterns: **{len(coverage_map['patterns'])}**.",
    f"Logic fill suite: `ja-s1-coverage.json` ({len(LOGIC_NEW)} cases).",
    f"ONNX fill suite: `../s1_onnx/ja-bekko-coverage.json` ({len(ONNX_NEW)} cases).",
    "",
]
(S1 / "PATTERNS.md").write_text("\n".join(lines), encoding="utf-8")
print(f"logic new={len(LOGIC_NEW)} onnx new={len(ONNX_NEW)} patterns={len(MAP)}")
print("all patterns ≥3")
