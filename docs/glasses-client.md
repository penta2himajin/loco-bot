# Glasses client (P10)

Even G2–class glasses are a **thin voice/UI client**. Inference, memory, and tools run
on the phone (or laptop) runtime — never E4B on-glass.

**Client implementation belongs in a separate frontend repository** (typically alongside
or depending on the Android companion). This note is the policy kept in loco-bot.

## Session path

```
glasses mic/display  →  phone companion (P9)  →  loco-agent
```

Transport: BLE or local WebSocket to the phone app. Keep replies short (companion mode).

## Non-goals

- On-glass Gemma / LiteRT-LM
- Independent session memory on the glasses
- Unrestricted computer-use from the glasses
- Growing glasses app code inside the loco-bot monorepo

## Next engineering steps

1. Define a minimal event subset over the wire (user audio/text in → Clarify/Done out).
2. Phone app exposes a local Agent endpoint for the glasses shell.
3. Measure end-to-end latency before adding features.

## History

Relocated from top-level `glasses/README.md` (introduced at `5fed21b`). See
[`surfaces.md`](surfaces.md) for restore commands.
