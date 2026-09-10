# macOS quick overlay (SoT)

Source of truth for the **laptop companion overlay**: a Spotlight-like floating
panel invoked by a global hotkey. Product framing lives in
[`product-ux-roadmap.md`](product-ux-roadmap.md); repo boundary in
[`surfaces.md`](surfaces.md).

**Status**: accepted direction (2026-09-10). Implementation may live outside
this repository; this document owns behaviour and contracts.

## Goal

From any app on macOS, press a global hotkey → a small overlay appears → type a
short question → stream the loco-bot reply → dismiss or pin.

Primary persona: **companion** short asks (and light work-sidecar later).

## Non-goals

- Embedding answers inside Apple **Spotlight** itself (no supported API).
- Replacing Spotlight’s `⌘Space` by default.
- Default hotkeys that use **`fn` / Globe** (unstable across keyboards and APIs).
- Shipping the Swift/UI binary inside the `loco-bot` git tree (see Surfaces).
- On-overlay E4B download UX beyond “runtime not ready” messaging (first-run
  remains `loco download` / doctor).

## Repo boundary

| Piece | Where |
|-------|--------|
| Brain (`loco-agent`, engine, memory, S1, tools) | **loco-bot** |
| Dev shells (`loco chat`, `loco serve`) | **loco-bot** |
| Overlay UI, global hotkey, menu-bar lifecycle | **Separate macOS surface** (Swift preferred) |
| This behaviour spec | **loco-bot** `docs/macos-overlay.md` (this file) |

The overlay is a thin client of the Agent API / IPC — same rule as Android and
glasses.

## Invocation

### Default hotkey

- **⌃⌘Space** (Control + Command + Space)

Rationale: registrable with ordinary global-hotkey APIs (modifiers + one key);
avoids Spotlight’s `⌘Space`; avoids flaky `fn`/Globe chords.

### Configurable

- User must be able to change the hotkey in the overlay app’s settings.
- Do **not** steal `⌘Space` unless the user explicitly reassigns it after
  freeing it in **System Settings → Keyboard → Keyboard Shortcuts**.
- Optional advanced chords (`fn+…`, three-key combos) are out of v1.

### Conflicts

On first launch (or when registration fails), show which shortcut is bound and
how to change it. Japanese IME / input-source shortcuts often sit near
`⌃Space` / `⌘Space`; document that in the settings UI copy.

## UX

### Appearance

- Centered (or upper-third) floating panel; dimmed desktop optional.
- Single composition: input field + streaming answer region.
- Not a full chat IDE. History can be minimal (last turn / session via memory).

### Flow

1. Hotkey → overlay focused, input empty (or prefilled from selection later).
2. User submits text (Return).
3. Show streaming tokens as they arrive (`AgentEvent::Token` when wired).
4. On `Clarify`: show question + numbered choices; accept number or label.
5. On `ToolRequest` (medium/high risk): inline allow/deny before continuing.
6. Escape or click-outside dismisses; optional “pin” keeps the panel open.

### Copy / language

- UI chrome may follow the user locale later; v1 may be Japanese-first or
  bilingual. Agent replies follow model + user language as today.

## Runtime integration

### Process model (v1)

- Menu-bar (or login-item) helper keeps a **warm** loco runtime:
  - Preferred: spawn/manage `loco serve --backend gpu` (JSONL stdio), or
  - Equivalent localhost IPC once added in loco-bot.
- Cold start of E4B is slow for overlay UX; **do not** load the model on every
  hotkey. Keep the runtime warm while the helper is running.

### Turn contract

Overlay sends user text; runtime returns the same event model as
`loco_agent::AgentEvent` / `TurnOutcome`:

`Clarify` → `Ack` / `Topic` → `Context` → `ToolRequest` / `ToolResult` →
`Token`* → `Done`

\* Today `Token` is reserved and `AgentSession::turn` may buffer the full
reply. **Overlay v1 should prefer streaming** once loco-bot exposes it (engine
already has `ChatSession::reply_stream`). Until then, show a waiting state then
the full `Done` text.

### Baseline latency (battery, M1 Max, measured 2026-09-10)

Informational targets — re-measure after wiring changes:

| Metric | GPU warm (bare chat stream) |
|--------|-----------------------------|
| TTFT (first text delta) | ~250–300 ms (first turn after open ~0.9 s) |
| Short answer E2E | ~0.3–0.5 s |
| Session open | ~0.7 s |

S1 embed is ~1–4 ms and is not the bottleneck.

Default inference backend for the overlay helper: **gpu** (Metal) on Apple
Silicon; fall back to cpu with a settings toggle.

## Permissions (macOS)

| Capability | When needed |
|------------|-------------|
| Global hotkey (⌃⌘Space) | Usually no special TCC for Carbon-style hotkeys |
| Accessibility / Input Monitoring | Only if using low-level event taps (avoid in v1) |
| Automation / Files | Only when FS tools are enabled with a sandbox root |
| Network | Only when high-risk tools (search/mail/drive) are enabled |

Request privileges lazily when a feature needs them, not at first launch.

## Settings (minimum)

- Hotkey (default ⌃⌘Space)
- Backend: gpu | cpu
- Start helper at login: on/off
- Tools: off / low-only / allow FS (with root) — mirror CLI consent ceiling
- Optional: clear or open session memory (delegate to `loco memory` semantics)

## Acceptance (v1)

1. ⌃⌘Space toggles or shows the overlay from another app.
2. Submitting a prompt yields a loco-bot reply without opening Terminal.
3. Runtime stays warm across multiple invocations in one login session.
4. Escape dismisses; hotkey can bring the overlay back.
5. If the model is missing, overlay shows a clear “run `loco doctor` /
   `loco download`” style message instead of hanging.

## Implementation notes for the next session

1. Keep UI in a **separate** macOS project; depend on `loco serve` or a thin IPC.
2. In loco-bot (if needed for polish): stream `Token` events through
   `AgentSession` / `loco serve` so TTFT is visible in the overlay.
3. Do not expand this SoT into Android/glasses UI code here.

## Open questions (non-blocking)

- Prefill input from current selection / clipboard on hotkey?
- Auto-dismiss after Done vs stay open until Escape?
- Branding name of the overlay app vs `loco` CLI?
