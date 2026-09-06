//! Pending topic clarification (persisted with session memory).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingClarification {
    pub question: String,
    pub candidates: Vec<PendingCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingCandidate {
    pub label: String,
    /// `None` means stay on the current topic.
    pub return_chunk: Option<usize>,
}
