#!/usr/bin/env python3
"""Run s1_onnx fixture resolve cases with bekko a8m / a25m and calibrate thresholds.

Eval-only. Mirrors loco-embed resolve + GraySafetyS2 enough to score the suite.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "crates/loco-embed/fixtures/s1_onnx/ja-bekko-resolve.json"
OUT = Path(__file__).with_name("S1_RESOLVE_RESULTS.md")

MODELS = {
    "bekko-a8m": "hotchpotch/bekko-embedding-v1-a8m",
    "bekko-a25m": "hotchpotch/bekko-embedding-v1-a25m",
}


@dataclass
class Th:
    continue_min: float
    return_min: float
    new_max: float
    clarify_min: float
    ambiguity_delta: float = 0.05


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def classify_deixis(user: str) -> str:
    """Match crates/loco-embed/src/deixis.rs closely enough for fixtures."""
    u = user.strip().lower()
    return_markers = ("戻って", "さっきの", "前の話", "前のやつ", "go back", "back to", "earlier")
    if any(m in u for m in return_markers):
        residue = u
        for m in (
            "さっきの",
            "前の話",
            "前のやつ",
            "前の",
            "話に戻って",
            "に戻って",
            "戻って",
            "go back to",
            "go back",
            "back to",
            "earlier",
            "なんだけど",
            "だけど",
            "けれど",
            "の話",
            "話",
            "について",
            "教えて",
        ):
            residue = residue.replace(m, "")
        residue = re.sub(r"[\s、。？?!,.：:]", "", residue)
        fillers = {"けど", "なの", "です", "ます", "して", "どう", "なに", "何"}
        named = len(residue) >= 2 and residue not in fillers and any(
            ch.isalnum() or ("\u4e00" <= ch <= "\u9fff") or ("\u3040" <= ch <= "\u30ff")
            for ch in residue
        )
        return "ReturnNamed" if named else "ReturnUnspecified"
    if any(m in u for m in ("それについて", "それ", "これ", "あれ", "その", "この", "あの")):
        return "ContinueHint"
    return "Plain"


def expand_query(user: str, previous_user: str | None) -> str:
    if previous_user and classify_deixis(user) in {"ContinueHint", "Plain"}:
        # Only expand short deictic follow-ups in fixtures (use_expand flag).
        return f"{previous_user}\n{user}"
    return user


def decide_s1(q, cur, past, th: Th):
    if cur is None:
        return ("New", None, [])
    cs = cos(q, cur)
    if cs >= th.continue_min:
        return ("Continue", None, [])
    past_ranked = sorted(
        ((i, cos(q, e)) for i, e in past), key=lambda x: -x[1]
    )
    best_i, best_s = (past_ranked[0] if past_ranked else (None, float("-inf")))
    if best_i is not None and best_s >= th.return_min:
        return ("Return", best_i, past_ranked)
    best_overall = max(cs, best_s if np.isfinite(best_s) else 0.0)
    if best_overall < th.new_max:
        return ("New", None, past_ranked)
    return ("Gray", None, past_ranked)


def gray_s2(past_ranked, th: Th) -> str:
    if len(past_ranked) >= 2:
        (_, s0), (_, s1) = past_ranked[0], past_ranked[1]
        if s0 >= th.clarify_min and s1 >= th.clarify_min and (s0 - s1) < th.ambiguity_delta:
            return "NeedsClarification"
    return "New"


def ambiguous_past_return(q, cur, past, th: Th) -> bool:
    if len(past) < 2:
        return False
    scored = sorted(((i, cos(q, e)) for i, e in past), key=lambda x: -x[1])
    s0, s1 = scored[0][1], scored[1][1]
    if s0 < th.return_min:
        return False
    if s0 - s1 >= th.ambiguity_delta:
        return False
    if cur is not None and cos(q, cur) >= th.continue_min:
        return False
    return True


def resolve_case(embs: dict[str, np.ndarray], case: dict, th: Th) -> tuple[str, dict]:
    kind = case["kind"]
    if kind == "embed_rank":
        a = embs[case["anchor"]]
        c = embs[case["closer"]]
        f = embs[case["farther"]]
        margin = cos(a, c) - cos(a, f)
        ok = margin >= case.get("min_margin", 0.05)
        return ("ok" if ok else "fail", {"margin": margin, "expect": "rank"})

    user = case["user"]
    prev = case.get("previous_user")
    text = expand_query(user, prev) if case.get("use_expand") else user
    q = embs[text]
    cur_text = case.get("current_text")
    cur = embs[cur_text] if cur_text else None
    past = [(p["index"], embs[p["text"]]) for p in case.get("past_texts", [])]
    previous_chunk = case.get("previous_chunk")

    deixis = classify_deixis(user)
    detail = {"deixis": deixis}

    if deixis == "ReturnUnspecified":
        if previous_chunk is not None:
            out = ("Return", previous_chunk)
        elif len(past) == 0:
            out = ("Continue", None) if cur is not None else ("New", None)
        elif len(past) == 1:
            out = ("Return", past[0][0])
        else:
            out = ("NeedsClarification", None)
    else:
        if ambiguous_past_return(q, cur, past, th):
            out = ("NeedsClarification", None)
        else:
            s1, idx, ranked = decide_s1(q, cur, past, th)
            detail["past_ranked"] = [(i, float(s)) for i, s in ranked[:3]]
            if cur is not None:
                detail["current_sim"] = cos(q, cur)
            if s1 == "Gray":
                g = gray_s2(ranked, th)
                out = (g, None)
            elif s1 == "Return":
                out = ("Return", idx)
            else:
                out = (s1, idx)

    expect = case["expect"]
    got_outcome, got_idx = out
    exp_outcome = expect["outcome"]
    ok = got_outcome == exp_outcome
    if ok and "chunk_index" in expect:
        ok = got_idx == expect["chunk_index"]
    detail["got"] = {"outcome": got_outcome, "chunk_index": got_idx}
    detail["expect"] = expect
    return ("ok" if ok else "fail", detail)


def collect_texts(suite: dict) -> list[str]:
    texts: set[str] = set()
    for case in suite["cases"]:
        if case["kind"] == "embed_rank":
            texts.update([case["anchor"], case["closer"], case["farther"]])
            continue
        user = case["user"]
        prev = case.get("previous_user")
        if case.get("use_expand"):
            texts.add(expand_query(user, prev))
        else:
            texts.add(user)
        if case.get("current_text"):
            texts.add(case["current_text"])
        for p in case.get("past_texts", []):
            texts.add(p["text"])
    return sorted(texts)


def score_suite(model: SentenceTransformer, suite: dict, th: Th) -> tuple[int, int, list]:
    texts = collect_texts(suite)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embs = {t: np.asarray(v) for t, v in zip(texts, vecs)}
    ok_n = fail_n = 0
    rows = []
    for case in suite["cases"]:
        status, detail = resolve_case(embs, case, th)
        if status == "ok":
            ok_n += 1
        else:
            fail_n += 1
        rows.append((case["id"], status, detail))
    return ok_n, fail_n, rows


def measure_bands(model: SentenceTransformer) -> dict:
    """Measure continue / return / new bands used for calibration."""
    pairs = {
        "continue_subway": (
            "乗り換えのコツは？",
            "東京の地下鉄の切符の買い方",
            "カレーの作り方を詳しく教えて",
        ),
        "continue_curry": (
            "スパイスは何を使う？",
            "カレーの作り方を詳しく教えて",
            "東京の地下鉄の切符の買い方",
        ),
        "named_return": (
            "さっきの地下鉄の話に戻って",
            "東京の地下鉄の切符の買い方",
            "カレーの作り方を詳しく教えて",
        ),
        "new_quantum": (
            "量子コンピュータの基礎を一言で",
            "東京の地下鉄の切符の買い方",
            "カレーの作り方を詳しく教えて",
        ),
        "new_baseball": (
            "プロ野球の指名打者制度とは",
            "東京の地下鉄の切符の買い方",
            "カレーの作り方を詳しく教えて",
        ),
        "ambiguous": (
            "関連する話に戻したい",
            "東京の地下鉄の切符の買い方",
            "カレーの作り方を詳しく教えて",
        ),
    }
    out = {}
    for tag, (q, a, b) in pairs.items():
        qa, aa, bb = model.encode([q, a, b], normalize_embeddings=True, show_progress_bar=False)
        out[tag] = {
            "rel": float(np.dot(qa, aa)),
            "unrel": float(np.dot(qa, bb)),
            "max": float(max(np.dot(qa, aa), np.dot(qa, bb))),
        }
    return out


def propose_th(bands: dict) -> Th:
    cont = min(bands["continue_subway"]["rel"], bands["continue_curry"]["rel"])
    ret = bands["named_return"]["rel"]
    hard_neg = max(
        bands["new_quantum"]["max"],
        bands["new_baseball"]["max"],
        bands["continue_subway"]["unrel"],
        bands["continue_curry"]["unrel"],
    )
    # Floor continue/return just under weakest true positive; new_max just above hard neg.
    continue_min = round(cont - 0.02, 3)
    return_min = round(min(ret, cont) - 0.02, 3)
    new_max = round(hard_neg + 0.02, 3)
    # Keep ordering: new_max < continue_min
    if new_max >= continue_min:
        mid = (hard_neg + cont) / 2
        new_max = round(mid - 0.01, 3)
        continue_min = round(mid + 0.01, 3)
        return_min = continue_min
    clarify_min = round((new_max + return_min) / 2, 3)
    return Th(continue_min, return_min, new_max, clarify_min)


def main() -> None:
    suite = json.loads(FIXTURE.read_text())
    lines = [
        "# S1 resolve through-check: bekko a8m / a25m",
        "",
        f"Fixture: `{FIXTURE.relative_to(ROOT)}`",
        "",
    ]
    for name, repo in MODELS.items():
        print(f"loading {name} …")
        model = SentenceTransformer(repo)
        bands = measure_bands(model)
        th = propose_th(bands)
        print(f"  proposed th={th}")
        ok_n, fail_n, rows = score_suite(model, suite, th)
        print(f"  {ok_n}/{ok_n + fail_n}")
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"Proposed thresholds: `{th}`")
        lines.append("")
        lines.append("Bands:")
        for k, v in bands.items():
            lines.append(f"- {k}: rel={v['rel']:.4f} unrel={v['unrel']:.4f} max={v['max']:.4f}")
        lines.append("")
        lines.append(f"**Score: {ok_n}/{ok_n + fail_n}**")
        lines.append("")
        lines.append("| id | status | detail |")
        lines.append("|----|--------|--------|")
        for cid, status, detail in rows:
            brief = json.dumps(detail, ensure_ascii=False)
            if len(brief) > 120:
                brief = brief[:117] + "…"
            lines.append(f"| {cid} | {status} | `{brief}` |")
        lines.append("")

        # Also try a hand-tuned grid around proposal if failures.
        if fail_n:
            best = (ok_n, th)
            for cm in np.arange(0.15, 0.35, 0.02):
                for nm in np.arange(0.05, cm - 0.02, 0.02):
                    for cl in np.arange(nm, cm, 0.02):
                        cand = Th(float(cm), float(cm), float(nm), float(cl))
                        o, f, _ = score_suite(model, suite, cand)
                        if o > best[0]:
                            best = (o, cand)
            lines.append(f"Grid best: {best[0]}/{ok_n + fail_n} with `{best[1]}`")
            print(f"  grid best {best[0]} th={best[1]}")
            ok_n, fail_n, rows = score_suite(model, suite, best[1])
            for cid, status, detail in rows:
                if status == "fail":
                    print(f"  FAIL {cid}: {detail}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
