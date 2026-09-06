# Android companion (P9)

Phone hosts the loco-bot runtime (LiteRT-LM + bekko + Agent API). This directory is a scaffold — no Gradle app yet.

## Goals

- Chat UI + model download progress + permission screens
- Session store under app sandbox
- Wire to Rust `loco-agent` via UniFFI / JNI or localhost IPC to `loco serve`

## Non-goals (for this phase)

- Shipping E4B weights inside the APK
- Glasses pairing (see P10)
- Full mail/drive OAuth UI until P8 providers exist

## Next engineering steps

1. Choose FFI path (UniFFI preferred if bindgen cost is acceptable).
2. Thin Kotlin shell that calls `AgentSession::turn` equivalents.
3. Map Android permission prompts onto `ToolConsent` / `ToolRisk`.
