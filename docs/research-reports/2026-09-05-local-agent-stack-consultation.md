# Local Agent Stack Consultation — LiteRT-LM, Gemma 4 E4B, Memory, Embeddings

- **Date**: 2026-09-05
- **Status**: P0–P5 landed (chat + memory + S1 + compiler + deixis + tools)
- **Scope**: Mobile/laptop small local agent (`loco-bot`)

## Goal

Build a small on-device agent for mobile and laptop that:

1. Runs **Gemma 4 E4B** via **LiteRT-LM**
2. Allows **first-run model download** (offline after that)
3. Adds a **chatstream-inspired memory layer** (ideas, not a hard dependency)
4. Uses a compact multilingual embedding model for **topic detection (S1)**

## Locked decisions

| Topic | Decision | Notes |
|-------|----------|-------|
| LLM runtime | LiteRT-LM | Native tool calling; Android / iOS / desktop / web |
| Chat model | `litert-community/gemma-4-E4B-it-litert-lm` | ~3.65 GB on disk; text decoder weights ~2.24 GB |
| Text-only pack | Defer | Official text-only `.litertlm` not published; vision/audio already lazy-loaded; unpack/repack later if needed |
| Model distribution | First-run download OK | Cache locally; no requirement to ship weights in the app binary |
| Memory approach | Chatstream *concepts*, not the crate | Hierarchical store + context compile + cascade; reimplement thin for loco-bot |
| Embedding (S1) | **`ibm-granite/granite-embedding-97m-multilingual-r2`** | See comparison below |
| Needle 2 | Optional later (v2+) | Tool router / structured extract / S2 worker — not the memory store |
| Language | **Rust-first** | Product binary is Rust; Python OK for eval / one-off scripts only |
| Toolchain | **mise** | Pin Rust (and later helpers) via repo `mise.toml` |
| First UI | CLI on laptop | Android / other shells after plain chat works |
| Memory timing | After plain chat | No memory middleware until E4B conversation path is green |
| LiteRT-LM Rust binding | `litertlm-rs` 0.16.x | Thin wrap in `loco-engine::ChatSession`; swap-friendly |

### P1 notes (2026-09-05)

- CLI: `loco chat [--backend cpu|gpu] [prompt]` (REPL if no prompt).
- Feature flag: `loco-cli`/`loco-engine` `inference` (default on for CLI).
- macOS quirk: community prebuilt `liblitert-lm.dylib` has install_name `@rpath/liblitert-lm.so`. Workspace `build.rs` scripts symlink `.so` → `.dylib` and set `@loader_path` rpath on the CLI binary.
- Build needs `LIBCLANG_PATH` for bindgen (set in `mise.toml` for macOS CLT).

### P2 notes (2026-09-05)

- Crate: `loco-memory` — durable turns, recent-N window, rolling text summary (no LLM).
- Persist path: `<cache>/memory/session.json`.
- CLI: `loco memory show|clear`; `loco chat` loads summary into system preamble and appends turns (disable with `--no-memory`).
- GPU smoke (Mac): one-shot `pong` in ~6.3s; CPU ~9.7s for similar prompt. Teardown warnings reduced by dropping Conversation before Engine.

### Why Rust (not Python-first)

- Final form is a cross-platform local agent; a single Rust core + thin CLIs/JNI later beats a Python→Rust rewrite.
- Official LiteRT-LM surface for non-mobile is the **C API**; community crates (`litertlm-rs` / `litertlm-sys`) wrap it. Python is not architecturally privileged.
- Memory (chatstream-shaped) and granite via ONNX (`ort`) fit Rust cleanly.
- Risk: Rust bindings are community-maintained — keep a thin FFI boundary so we can swap crates or bind `engine.h` directly if needed.
- Parallel smoke: official `litert-lm` CLI can validate models without blocking the Rust path.

## Delivery sequencing

| Phase | Deliverable |
|-------|-------------|
| **P0** | mise + Rust workspace + model download / doctor CLI |
| **P1** | LiteRT-LM + E4B plain chat in CLI (streaming) |
| **P2** | Thin memory (turns + recent-N + simple summary) |
| **P3** | granite-97m-r2 S1 topic detection — **landed** (`loco-embed` + session chunks) |
| **P4** | Context compiler → E4B — **landed** (`loco-memory::compile`, resident + dynamic on return) |
| **P5** | Small tool surface — **landed** (`loco-engine::tools` + agent loop) |
| **P6** | As-needed backlog (Needle / reranker / text-only); **P6.0** S1 eval harness first |

## Stack sketch

```
User input
  → Memory middleware (chatstream-inspired, simplified)
       S1: granite-97m-r2 cosine vs chunk/root embeddings
       S1b (optional later): cross-encoder rerank top-k only
       S2 (optional later): Needle 2 or E4B structured classify
       Compiler: resident (root + recent N) + dynamic (returned chunk)
  → LiteRT-LM + Gemma 4 E4B  (reply + tool calls)
  → Persist turns / summaries / facts into store
```

Embedding runs in a **separate** runtime from LiteRT-LM (e.g. ONNX Runtime / OpenVINO / ExecuTorch). That split is expected.

## Model notes

### Gemma 4 E4B + LiteRT-LM

- HF card: https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm
- Disk ~3.65 GB = text decoder (~2.24 GB) + embeddings (~0.67 GB, mmap-friendly) + multimodal parts loaded on demand
- Official issue (#1883): no planned text-only release; unpack/repack via LiteRT-LM builder if file size must shrink
- Agentic features: function calling in Python / Kotlin / Swift APIs; speculative decoding / MTP on supported backends
- Rough decode: mobile GPU ~20–50 tok/s (with speculative decoding on some tasks); Mac Metal much faster

### Chatstream memory (what to borrow)

Source: `~/repos/chatstream` (`docs/chatstream.md`, orchestrator / detector / compiler).

**Borrow**

- Never delete turns; hierarchy is an index, not compression-by-deletion
- Chunks with summary + keywords; root state summary
- Context compiler: resident window + dynamic chunk on topic return
- Cascade: embedding first (maximize instant decisions), escalate only on gray zone
- Lessons from chatstream S1 work: do **not** EMA-blend the query embedding; prefer additive temporal/keyword boosts; short/deictic inputs need query expansion from recent context

**Do not import wholesale**

- Cloudflare / cloud LLM providers as defaults
- Full S2/S3 and splitter complexity in v0
- Hard dependency on `chatstream-core` crate

### Needle 2 (Cactus)

- ~45M params, ~14 MB binary, ~28 MB RAM; tool calling + grammar-constrained extract
- “Memory” in docs = bounded KV / sliding window — **not** long-term conversational memory
- Contrastive head is for **tool catalogue retrieval**, not general passage embedding
- Fit: fact extraction, optional S2 topic classify, large-tool routing
- Cost: second runtime + dual tool-schema management — introduce only when E4B extract latency hurts

## Embedding comparison (topic detection / S1)

Use case: few–tens of topic chunks per session, score every turn, co-reside with multi-GB E4B.

| Model | Role | Rough retrieval (MMTEB ML) | Edge fit |
|-------|------|----------------------------|----------|
| Xenova / paraphrase-multilingual-MiniLM-L12-v2 | Older STS baseline | Weaker for retrieval/topic separation | Tiny, proven, good for spikes only |
| bekko-a8m / a25m | Ultra-compact multilingual retrieval | ~56–57.5 | Fastest/smallest; solid if mobile-first |
| **granite-embedding-97m-multilingual-r2** | Compact multilingual retrieval | **~60.3** | **Chosen** — quality/size balance |
| granite-embedding-311m-multilingual-r2 | Full-size multilingual retrieval | ~65.2 | Overkill for S1; keep as upgrade if Overlap stays high |
| Cross-encoder reranker | Precision on top-k | N/A | Optional S1b only; not every-chunk every turn |

**Why granite-97m over 311m for this product**

- Topic nav compares against a small candidate set; +5 retrieval points rarely beat better thresholds / query expansion
- Query embedding runs every turn; 97m is materially cheaper on CPU/MPS
- E4B already dominates RAM/disk; keep the memory sidecar lean
- IBM guidance: 97m for edge/latency; 311m when accuracy is top priority and budget allows

**Name note**: “bekko-a29m” was not found on Hub; public sizes are **a8m** and **a25m**.

## Open items (post-P0)

1. Persistence: SQLite vs files; sync story across devices (if any)
2. Tool surface for v1 (clock, notes, local search, …)
3. When (if ever) to add Needle 2 and/or a cross-encoder S1b
4. Japanese UX assumptions and evaluation set for S1 thresholds — **started** (`fixtures/s1/ja-deixis-s1-logic.json`)
5. Which LiteRT-LM Rust binding to standardize on for P1 (`litertlm-rs` vs alternatives)

## P3 implementation notes (2026-09-05)

- Crate `loco-embed`: cosine + short-query expansion + S1 cascade; optional `ort` feature loads `onnx/model.onnx` (CLS pool, L2 normalize, 384-d).
- Cache id `granite-97m` downloads `onnx/model.onnx` + `tokenizer.json` from `ibm-granite/granite-embedding-97m-multilingual-r2`.
- `SessionMemory` persists `chunks` / `current_chunk`; chat logs `[topic: …]` and includes active topic in the system preamble.
- Default thresholds: continue ≥ 0.75, return ≥ 0.65, new if best < 0.35; gray zone without S2 → New. Query expansion is deictic/bare-followup only (not every short topical line).
- Next: P4 context compiler (resident + dynamic chunk on return).

## P4 implementation notes (2026-09-05)

- `loco_memory::compile(switch)` builds resident (summary + active topic + optional recent turns) and, on `TopicSwitch::Return`, a dynamic snapshot of that chunk’s turns.
- CLI injects compiled notes in-band every turn (`[context: resident(+dynamic) N chars]`). Continue/New omit the recent-turn dump (Conversation already has it); Return includes dynamic + a short recent window.

## P4.1 deixis resolve (2026-09-05)

- Lexicon deixis classes: Plain / ContinueHint / ReturnNamed / ReturnUnspecified.
- Unspecified return ("さっきの話") uses `previous_chunk` (topic stack N-1); if missing and multiple past chunks → ask which topic.
- Named return still uses granite S1; close top-2 past scores → clarify.
- Continue deixis expands with the previous user turn only.
- Verified: `さっきの話` → return#0+dynamic; `それについて` → continue; seeded ambiguous → clarify → `2` ack.

## P5 implementation notes (2026-09-05)

- `loco-engine::tools`: OpenAI-style schemas via `ConversationConfig::set_tools`; parse `tool_calls`; built-ins `get_current_time`, `note_write` / `note_read` (`…/notes/notes.json`), `session_stats`.
- `ChatSession::reply_with_tools` non-streaming agent loop (cap 4 rounds); CLI default-on, `--no-tools` to disable.
- Next: measure before Needle / reranker / text-only; first step = S1/deixis eval harness.

## P6.0 S1/deixis eval harness (2026-09-06)

- `loco-embed::eval` loads declarative JSON suites; kinds: `deixis` / `expand` / `resolve` / `clarify_match`.
- Fixtures: `crates/loco-embed/fixtures/s1/` (synthetic embeddings — CI needs no ONNX).
- Run: `mise run eval-s1` or `cargo test -p loco-embed --test s1_eval`.
- Seed suite `ja-deixis-s1-logic.json` covers Japanese deixis phrases, stack return, ambiguous clarify, expand rules.
- Deferred: ONNX text suites, Needle, reranker, text-only pack (still as-needed).

## References

- LiteRT-LM: https://github.com/google-ai-edge/LiteRT-LM
- Gemma 4 + LiteRT-LM blog: https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/
- Granite 97m R2: https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2
- Granite R2 blog: https://huggingface.co/blog/ibm-granite/granite-embedding-multilingual-r2
- Bekko Embedding: https://huggingface.co/blog/hotchpotch/bekko-embedding
- Needle 2: https://huggingface.co/Cactus-Compute/needle2
- Chatstream design: `~/repos/chatstream/docs/chatstream.md`
