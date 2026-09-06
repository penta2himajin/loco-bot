//! Thin hierarchical-ready session memory for loco-bot.
//!
//! P2: durable turns, recent-N window, rolling text summary.
//! P3: topic chunks + S1 indices (embeddings filled by loco-embed).
//! P4: context compiler (resident + dynamic on topic return).
//! P4.1: previous-chunk stack + pending clarification for deixis.

mod chunk;
mod compile;
mod pending;
mod session;
mod summary;
mod turn;

pub use chunk::TopicChunk;
pub use compile::{
    compile, CompiledContext, CompilerConfig, DynamicChunk, ResidentContext, TopicSwitch,
};
pub use pending::{PendingCandidate, PendingClarification};
pub use session::{SessionMemory, SessionMemoryError, DEFAULT_RECENT_N};
pub use summary::simple_summary;
pub use turn::Turn;
