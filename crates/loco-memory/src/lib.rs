//! Thin hierarchical-ready session memory for loco-bot.
//!
//! P2: durable turns, recent-N window, rolling text summary.
//! P3: topic chunks + S1 indices (embeddings filled by loco-embed).
//! P4: context compiler (resident + dynamic on topic return).

mod chunk;
mod compile;
mod session;
mod summary;
mod turn;

pub use chunk::TopicChunk;
pub use compile::{
    compile, CompiledContext, CompilerConfig, DynamicChunk, ResidentContext, TopicSwitch,
};
pub use session::{SessionMemory, SessionMemoryError, DEFAULT_RECENT_N};
pub use summary::simple_summary;
pub use turn::Turn;
