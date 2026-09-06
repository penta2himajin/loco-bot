# loco-bot

Small on-device agent for laptop/mobile: **LiteRT-LM + Gemma 4 E4B**, with chatstream-inspired session memory and bekko-a8m S1 topic detection.

See `docs/research-reports/2026-09-05-local-agent-stack-consultation.md` for stack decisions.

## Toolchain

This repo uses [mise](https://mise.jdx.dev/) for the Rust toolchain:

```bash
mise trust
mise install
```

Optional: `git config core.hooksPath git-hooks`

## Build & test

```bash
mise run build
mise run test
mise run lint
```

Or plain Cargo after `mise install`:

```bash
cargo build --workspace
cargo test --workspace
```

## CLI

Binary name: `loco`

```bash
cargo run -p loco-cli -- doctor
cargo run -p loco-cli -- models
cargo run -p loco-cli -- download gemma4-e4b          # ~3.7GB, first-run download
cargo run -p loco-cli -- download bekko-a8m           # ~165MB ONNX + tokenizer (S1)
cargo run -p loco-cli -- chat --backend cpu "Hello"   # one-shot reply (+ memory / S1 / tools)
cargo run -p loco-cli -- chat --backend gpu           # interactive REPL
cargo run -p loco-cli -- memory show
cargo run -p loco-cli -- memory clear
cargo test -p loco-embed --test s1_eval              # S1/deixis logic fixtures
cargo test -p loco-embed --features ort --test s1_eval_onnx  # real bekko (skip if uncached)
```

Session memory stores turns, a rolling summary, and S1 topic chunks under the cache
(`…/memory/session.json`). Each chat turn runs S1/deixis resolve (when bekko is cached), then a thin context
compiler injects resident notes and — on topic return — the returned chunk’s turns
(`[context: resident(+dynamic)]`). Underspecified returns like「さっきの話」use the
previous topic; ambiguous cases ask which topic. Use `--no-memory` or `--no-topic`
to disable.

Built-in tools (default on): `get_current_time`, `note_write` / `note_read`
(`…/notes/notes.json`), and `session_stats`. Disable with `--no-tools`.

Cache default: platform cache dir `/loco-bot/models/…` (override with `--cache-dir` or `LOCO_CACHE_DIR`).

Build notes: LiteRT-LM bindings need `libclang` (`LIBCLANG_PATH` is set in `mise.toml` for macOS CLT). First build downloads a native `liblitert-lm` prebuilt. The `embed` feature pulls ONNX Runtime via `ort`.

## Layout

```
crates/loco-cli/     # CLI entrypoint
crates/loco-engine/  # model catalog, cache, LiteRT-LM chat, tools
crates/loco-memory/  # session memory (turns + topic chunks)
crates/loco-embed/   # bekko-a8m ONNX + S1 cascade
docs/research-reports/
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

**Model weights are separate.** Third-party model licenses (Gemma 4, bekko, …)
apply in addition to this code license. See [`MODEL_LICENSES.md`](MODEL_LICENSES.md)
(verify upstream terms yourself; that file may lag).
