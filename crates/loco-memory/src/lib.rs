//! Thin hierarchical-ready session memory for loco-bot.
//!
//! P2 scope: durable turns, a recent-N window, and a rolling text summary.
//! Topic detection / context compilation land in later phases.

mod session;
mod summary;
mod turn;

pub use session::{SessionMemory, SessionMemoryError, DEFAULT_RECENT_N};
pub use summary::simple_summary;
pub use turn::Turn;
