# loco-bot

Small on-device agent for laptop/mobile: **LiteRT-LM + Gemma 4 E4B**, with a chatstream-inspired memory layer planned after plain chat works.

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

## CLI (P0)

Binary name: `loco`

```bash
cargo run -p loco-cli -- doctor
cargo run -p loco-cli -- models
cargo run -p loco-cli -- download gemma4-e4b   # ~3.6GB, first-run download
```

Cache default: platform cache dir `/loco-bot/models/…` (override with `--cache-dir` or `LOCO_CACHE_DIR`).

Chat / LiteRT-LM inference lands in **P1**.

## Layout

```
crates/loco-cli/     # CLI entrypoint
crates/loco-engine/  # model catalog, cache paths, readiness
docs/research-reports/
```

## License

MIT. See `LICENSE`.
