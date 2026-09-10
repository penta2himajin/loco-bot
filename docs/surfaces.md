# Surface boundary (loco-bot vs frontends)

loco-bot owns the **on-device agent brain** (`loco-agent`, engine, memory, embed) and thin
dev shells (`loco chat`, `loco serve`). Platform apps (Android, glasses UI) are **separate
surfaces**: they consume the Agent API / IPC contract; their Gradle/Swift UI code should
not grow inside this repository.

## Policy docs in this repo

| Surface | Doc |
|---------|-----|
| Product phases P7–P10 | [`product-ux-roadmap.md`](product-ux-roadmap.md) |
| macOS quick overlay (hotkey panel) | [`macos-overlay.md`](macos-overlay.md) |
| Android companion (P9) | [`android-companion.md`](android-companion.md) |
| Glasses thin client (P10) | [`glasses-client.md`](glasses-client.md) |

## Former top-level scaffolds (recoverable)

Before this consolidation, short READMEs lived at:

| Former path | Introduced | Tip commit that still had both |
|-------------|------------|--------------------------------|
| `android/README.md` | `08d2615` (P9) | `5fed21b` |
| `glasses/README.md` | `5fed21b` (P10) | `5fed21b` |

Restore a file from git history (example):

```bash
git show 5fed21b:android/README.md
git show 5fed21b:glasses/README.md
```

Or check out the pre-move tree:

```bash
git checkout 5fed21b -- android glasses
```

Content was folded into the policy docs above; the paths above remain in history forever.
