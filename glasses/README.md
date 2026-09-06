# Glasses client (P10)

Even G2–class glasses are a **thin voice/UI client**. Inference, memory, and tools run on the phone (or laptop) runtime — never E4B on-glass.

## Session path

```
glasses mic/display  →  phone companion (P9)  →  loco-agent
```

Transport: BLE or local WebSocket to the phone app. Keep replies short (companion mode).

## Non-goals

- On-glass Gemma / LiteRT-LM
- Independent session memory on the glasses
- Unrestricted computer-use from the glasses

## Next engineering steps

1. Define a minimal event subset over the wire (user audio/text in → Clarify/Done out).
2. Phone app exposes a local Agent endpoint for the glasses shell.
3. Measure end-to-end latency before adding features.
