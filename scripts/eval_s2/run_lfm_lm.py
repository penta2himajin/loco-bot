#!/usr/bin/env python3
"""Fallback spike: LFM2.5-350M (generative) as structured S2 classifier.

Usage:
  .tmp/venv-s2/bin/python scripts/eval_s2/run_lfm_lm.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_CASES = Path(__file__).with_name("cases_ja.json")
MODEL_ID = "LiquidAI/LFM2.5-350M"

SYSTEM = """You classify chat topic switches. Output ONLY one JSON object:
{"decision":"Continue"|"New"|"Return"|"Clarify","return_index":null|number}

Meanings:
- Continue = same topic as current_topic
- New = unrelated new subject (not a return)
- Return = go back to one past_topics entry; set return_index to its [n]
- Clarify = wants to go back but which past topic is unclear

Examples:
current_topic: baking bread
past_topics: [0] Tokyo subway tickets
user: how long should I knead?
JSON: {"decision":"Continue","return_index":null}

current_topic: baking bread
past_topics: [0] Tokyo subway tickets
user: go back to the subway ticket question
JSON: {"decision":"Return","return_index":0}

current_topic: baking bread
past_topics: [0] Tokyo subway tickets
user: what is a qubit in one sentence?
JSON: {"decision":"New","return_index":null}

current_topic: weather tomorrow
past_topics: [0] Tokyo subway tickets
[1] curry recipe details
user: I want to go back to that earlier topic
JSON: {"decision":"Clarify","return_index":null}
"""


def build_user(case: dict) -> str:
    past = case.get("past_labels", [])
    past_lines = "\n".join(f"[{i}] {lab}" for i, lab in enumerate(past)) or "(none)"
    return (
        f"current_topic: {case['current_label']}\n"
        f"past_topics:\n{past_lines}\n"
        f"user: {case['user']}\n"
        "JSON:"
    )


def parse_decision(text: str) -> dict:
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        return {"decision": "PARSE_FAIL", "raw": text}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"decision": "PARSE_FAIL", "raw": text}
    return {
        "decision": obj.get("decision"),
        "return_index": obj.get("return_index"),
        "raw": text.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    suite = json.loads(args.cases.read_text())
    print(f"loading {args.model} on {args.device} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(args.device).eval()

    passed = failed = 0
    for case in suite["cases"]:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(case)},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        if hasattr(encoded, "input_ids"):
            input_ids = encoded["input_ids"]
        elif isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
        else:
            input_ids = encoded
        input_ids = input_ids.to(args.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.05,
            )
        gen = tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True)
        got = parse_decision(gen)
        expect = case["expect"]
        ok = got.get("decision") == expect
        if ok and expect == "Return" and "expect_return_index" in case:
            ok = got.get("return_index") == case["expect_return_index"]
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"\n[{status}] {case['id']} expect={expect} got={got.get('decision')}")
        print(f"  user: {case['user']}")
        print(f"  raw: {got.get('raw', '')[:200]}")

    print(f"\nsummary: {passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
