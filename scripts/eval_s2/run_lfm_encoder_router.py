#!/usr/bin/env python3
"""Spike: LFM2.5-Encoder-350M-Prompt-Router as S2 topic router (JA cases).

Usage:
  .tmp/venv-s2/bin/python scripts/eval_s2/run_lfm_encoder_router.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = Path(__file__).with_name("cases_ja.json")
MODEL_ID = "LiquidAI/LFM2.5-Encoder-350M-Prompt-Router"

CONTINUE = "今の話題を続ける"
NEW = "全く別の新しい話題を始める"
CLARIFY = "どれに戻るか不明なのでユーザーに確認する"
RETURN_FMT = "過去の話題に戻る: {label}"


def build_lanes(current_label: str, past_labels: list[str]) -> list[tuple[str, dict]]:
    """Return (lane_text, meta) pairs. meta drives expect matching."""
    lanes: list[tuple[str, dict]] = [
        (f"{CONTINUE}: {current_label}", {"kind": "Continue"}),
        (NEW, {"kind": "New"}),
        (CLARIFY, {"kind": "Clarify"}),
    ]
    for i, label in enumerate(past_labels):
        lanes.append(
            (
                RETURN_FMT.format(label=label),
                {"kind": "Return", "index": i},
            )
        )
    return lanes


def decide(ranked: list[dict], metas: list[dict]) -> dict:
    top = ranked[0]
    meta = metas[next(i for i, m in enumerate(metas) if m["lane"] == top["route"])]
    out = {"kind": meta["kind"], "score": top["score"], "route": top["route"]}
    if meta["kind"] == "Return":
        out["index"] = meta["index"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    suite = json.loads(args.cases.read_text())
    print(f"loading {args.model} on {args.device} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.to(args.device).eval()

    passed = failed = 0
    for case in suite["cases"]:
        lanes = build_lanes(case["current_label"], case.get("past_labels", []))
        route_texts = [t for t, _ in lanes]
        metas = [{"lane": t, **m} for t, m in lanes]
        ranked = model.route(case["user"], route_texts, tokenizer=tokenizer)
        # Attach lane text already in ranked["route"]
        got = decide(ranked, metas)
        expect = case["expect"]
        ok = got["kind"] == expect
        if ok and expect == "Return" and "expect_return_index" in case:
            ok = got.get("index") == case["expect_return_index"]
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"\n[{status}] {case['id']} expect={expect} got={got['kind']} p={got['score']:.3f}")
        print(f"  user: {case['user']}")
        for row in ranked[:5]:
            print(f"  {row['score']:.3f}  {row['route']}")

    print(f"\nsummary: {passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
