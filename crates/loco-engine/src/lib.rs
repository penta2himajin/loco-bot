//! Model catalog, cache layout, and readiness checks for loco-bot.
//!
//! LiteRT-LM inference wiring lands in a later phase; this crate owns paths
//! and model identity so the CLI can download and verify assets first.

mod catalog;
mod paths;
mod status;

pub use catalog::{ModelId, ModelSpec, GEMMA4_E4B_IT};
pub use paths::{default_cache_root, model_file_path, CacheLayout};
pub use status::{model_status, ModelStatus};
