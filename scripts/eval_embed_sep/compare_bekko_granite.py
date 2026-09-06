#!/usr/bin/env python3
"""Compare embedding separation: granite-97m vs bekko a8m / a25m.

Measures related−unrelated cosine margins on the same JA pairs used for S1.
Eval-only; not a product runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("RESULTS.md")

GRANITE_ID = "ibm-granite/granite-embedding-97m-multilingual-r2"
BEKKO_A8M = "hotchpotch/bekko-embedding-v1-a8m"
BEKKO_A25M = "hotchpotch/bekko-embedding-v1-a25m"

# Pairs: (query, related_doc, unrelated_doc, tag)
PAIRS = [
    (
        "さっきの地下鉄の話に戻って",
        "東京の地下鉄の切符の買い方",
        "カレーの作り方を詳しく教えて",
        "named_return",
    ),
    (
        "乗り換えのコツは？",
        "東京の地下鉄の切符の買い方",
        "カレーの作り方を詳しく教えて",
        "continue_subway",
    ),
    (
        "スパイスは何を使う？",
        "カレーの作り方を詳しく教えて",
        "東京の地下鉄の切符の買い方",
        "continue_curry",
    ),
    (
        "量子コンピュータの基礎を一言で",
        "東京の地下鉄の切符の買い方",
        "カレーの作り方を詳しく教えて",
        "new_vs_topics",  # both should be weak; use related=subway as "current" proxy
    ),
    (
        "プロ野球の指名打者制度とは",
        "東京の地下鉄の切符の買い方",
        "カレーの作り方を詳しく教えて",
        "new_baseball",
    ),
    (
        "確定申告の基礎控除って何",
        "カレーの作り方を詳しく教えて",
        "東京の地下鉄の切符の買い方",
        "new_tax",
    ),
    (
        "関連する話に戻したい",
        "東京の地下鉄の切符の買い方",
        "カレーの作り方を詳しく教えて",
        "ambiguous_two_past",
    ),
    (
        "東京の地下鉄について教えて",
        "Tell me about the Tokyo subway",
        "バナナのスムージーの作り方",
        "rank_subway_en",
    ),
]


def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class GraniteCls:
    """Match loco-embed: CLS token + L2."""

    def __init__(self, model_id: str = GRANITE_ID):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        cls = out.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        return l2(cls)


class BekkoSt:
    def __init__(self, model_id: str):
        self.model = SentenceTransformer(model_id)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )


def eval_model(name: str, encode_fn) -> list[dict]:
    rows = []
    for q, rel, unrel, tag in PAIRS:
        embs = encode_fn([q, rel, unrel])
        r = cos(embs[0], embs[1])
        u = cos(embs[0], embs[2])
        rows.append(
            {
                "model": name,
                "tag": tag,
                "related": r,
                "unrelated": u,
                "margin": r - u,
            }
        )
        print(f"{name:12} {tag:20} rel={r:.4f} unrel={u:.4f} margin={r - u:+.4f}")
    return rows


def main() -> None:
    print("loading granite-97m (CLS) …", flush=True)
    granite = GraniteCls()
    print("loading bekko-a8m …", flush=True)
    a8m = BekkoSt(BEKKO_A8M)
    print("loading bekko-a25m …", flush=True)
    a25m = BekkoSt(BEKKO_A25M)

    all_rows: list[dict] = []
    all_rows += eval_model("granite-97m", granite.encode)
    print()
    all_rows += eval_model("bekko-a8m", a8m.encode)
    print()
    all_rows += eval_model("bekko-a25m", a25m.encode)

    # Summaries: mean margin on continue/return vs new tags
    def mean_margin(model: str, pred) -> float:
        xs = [r["margin"] for r in all_rows if r["model"] == model and pred(r["tag"])]
        return float(np.mean(xs)) if xs else float("nan")

    pos = lambda t: t.startswith("continue") or t.startswith("named") or t.startswith("rank")
    neg = lambda t: t.startswith("new_")

    lines = [
        "# Embedding separation: granite vs bekko",
        "",
        "Measured on shared JA (and one EN) query–doc pairs. Higher `margin = related − unrelated` is better separation.",
        "",
        "## Per-pair margins",
        "",
        "| model | tag | related | unrelated | margin |",
        "|-------|-----|--------:|----------:|-------:|",
    ]
    for r in all_rows:
        lines.append(
            f"| {r['model']} | {r['tag']} | {r['related']:.4f} | {r['unrelated']:.4f} | {r['margin']:+.4f} |"
        )

    lines += [
        "",
        "## Summary (mean margin)",
        "",
        "| model | positive-ish tags (continue/named/rank) | new_* tags |",
        "|-------|----------------------------------------:|-----------:|",
    ]
    for m in ("granite-97m", "bekko-a8m", "bekko-a25m"):
        lines.append(
            f"| {m} | {mean_margin(m, pos):+.4f} | {mean_margin(m, neg):+.4f} |"
        )

    # For new_* we want BOTH related and unrelated low; report max of the two scores
    lines += [
        "",
        "## new_* max(similarity to either topic) — lower is better for New",
        "",
        "| model | tag | max(sim) |",
        "|-------|-----|---------:|",
    ]
    for r in all_rows:
        if not r["tag"].startswith("new_"):
            continue
        mx = max(r["related"], r["unrelated"])
        lines.append(f"| {r['model']} | {r['tag']} | {mx:.4f} |")

    OUT.write_text("\n".join(lines) + "\n")
    raw = Path(__file__).with_name("results.json")
    raw.write_text(json.dumps(all_rows, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    print(f"wrote {raw}")


if __name__ == "__main__":
    main()
