//! Topic chunks for S1 (embedding + turn range).

use serde::{Deserialize, Serialize};

/// One semantic topic span within a session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopicChunk {
    pub id: u64,
    /// Short text used for display / compiler (usually first user line).
    pub summary: String,
    /// L2-normalized embedding (bekko-a8m 384-d when using ort).
    pub embedding: Vec<f32>,
    /// Inclusive start index into [`crate::SessionMemory::turns`].
    pub turn_start: usize,
    /// Exclusive end index into turns (grows as turns append).
    pub turn_end: usize,
}

impl TopicChunk {
    pub fn new(
        id: u64,
        summary: impl Into<String>,
        embedding: Vec<f32>,
        turn_start: usize,
    ) -> Self {
        Self {
            id,
            summary: summary.into(),
            embedding,
            turn_start,
            turn_end: turn_start,
        }
    }
}
