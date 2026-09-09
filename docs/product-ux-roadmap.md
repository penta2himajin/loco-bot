# Product UX roadmap

Personas and phased surfaces for loco-bot. Engineering design stays in ADRs and research reports; this note is the product framing for P7–P10.

## Personas

- **Companion**: short asks from laptop or phone — time, notes, “what were we talking about?”, light search.
- **Work sidecar**: same runtime, permissioned tools for local folders and (later) mail/drive.
- **Glasses**: voice/UI client only; inference and memory stay on the phone (or laptop) host.

## Local-complete (honest scope)

| Capability | Offline | Needs network / account |
|------------|---------|-------------------------|
| Chat + S1 memory + notes/time | Yes | No |
| Local folder organize | Yes (FS permission) | No |
| Web search | No | Yes |
| Mail / Drive | No | Yes (OAuth) |

“Local-complete” means **model and memory stay on-device**. Network tools are optional and consent-gated.

## Phases

| Phase | Branch prefix | Deliverable |
|-------|---------------|-------------|
| **P7** | `claude/p7-…` | `loco-agent` library API + CLI/`loco serve` shells |
| **P8** | `claude/p8-…` | Tool ladder: FS (sandbox + dry-run) → web/mail/drive stubs |
| **P9** | `claude/p9-…` | Android companion **policy** ([`android-companion.md`](android-companion.md)); app lives outside this repo |
| **P10** | `claude/p10-…` | Glasses thin-client **policy** ([`glasses-client.md`](glasses-client.md)); app lives outside this repo |

Surface vs core boundary: [`surfaces.md`](surfaces.md).

## Agent API (in-process)

Surfaces call [`loco_agent::AgentSession`](../crates/loco-agent/src/lib.rs), not a public cloud API.

Typical turn events: `Clarify` → `Ack` / `Topic` → `Context` → `ToolRequest` / `ToolResult` → `Token` → `Done`.

- Interactive CLI: `loco chat`
- JSONL shell for thin UIs: `loco serve` (one `{"user":"…"}` line in → one `TurnOutcome` JSON line out)

## What not to do yet

- iOS packaging before Android companion works
- Unrestricted desktop “computer use”
- Bundling model weights in the app binary
- On-glass E4B
