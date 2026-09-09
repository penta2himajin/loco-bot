# Android companion (P9)

Phone hosts the loco-bot runtime (LiteRT-LM + bekko + Agent API). **App implementation
belongs in a separate frontend repository**; this note is the product/engineering policy
kept in loco-bot.

The product brain stays in Rust (`loco-agent`). The Android app is a surface: UI, OS
permissions, and process lifecycle around the same turn events (`Clarify`, `ToolRequest`,
`Done`, …).

## Goals

- Chat UI + model download progress + permission screens
- Session store under app sandbox
- Wire to Rust `loco-agent` via UniFFI / JNI or localhost IPC to `loco serve`

## Non-goals (for this phase)

- Shipping E4B weights inside the APK
- Glasses pairing (see [`glasses-client.md`](glasses-client.md))
- Full mail/drive OAuth UI until P8 providers exist
- Growing a Gradle tree inside the loco-bot monorepo

## Next engineering steps

1. Choose FFI path (UniFFI preferred if bindgen cost is acceptable).
2. Thin Kotlin shell that calls `AgentSession::turn` equivalents.
3. Map Android permission prompts onto `ToolConsent` / `ToolRisk`.

## History

Relocated from top-level `android/README.md` (introduced at `08d2615`). See
[`surfaces.md`](surfaces.md) for restore commands.
