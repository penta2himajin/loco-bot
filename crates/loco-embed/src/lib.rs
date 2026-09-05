//! Embedding helpers and S1 topic detection for loco-bot.
//!
//! Pure cosine / cascade logic is always available. The granite-97m ONNX
//! runtime sits behind the `ort` feature.

mod cosine;
mod expand;
mod s1;

#[cfg(feature = "ort")]
mod ort_granite;

pub use cosine::{cosine, l2_normalize};
pub use expand::expand_query;
pub use s1::{decide_s1, ChunkScore, S1Thresholds, TopicDecision};

#[cfg(feature = "ort")]
pub use ort_granite::{EmbedError, GraniteEmbedder};

/// Fixed embedding size for granite-embedding-97m-multilingual-r2.
pub const GRANITE_DIM: usize = 384;
