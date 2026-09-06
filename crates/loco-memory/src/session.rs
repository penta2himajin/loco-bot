//! In-memory + JSON file session store.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::chunk::TopicChunk;
use crate::pending::PendingClarification;
use crate::summary::simple_summary;
use crate::turn::Turn;

pub const DEFAULT_RECENT_N: usize = 8;

#[derive(Debug, Error)]
pub enum SessionMemoryError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionMemory {
    pub turns: Vec<Turn>,
    /// Rolling text summary of recent turns (no LLM).
    pub summary: String,
    /// How many turns to keep in the recent window / summary.
    pub recent_n: usize,
    /// Topic chunks indexed by S1 (P3+).
    #[serde(default)]
    pub chunks: Vec<TopicChunk>,
    /// Index into `chunks` for the active topic.
    #[serde(default)]
    pub current_chunk: Option<usize>,
    /// Previous current chunk (topic stack N-1) for unspecified "さっきの話".
    #[serde(default)]
    pub previous_chunk: Option<usize>,
    /// Waiting for the user to pick among ambiguous return targets.
    #[serde(default)]
    pub pending_clarify: Option<PendingClarification>,
}

impl Default for SessionMemory {
    fn default() -> Self {
        Self::new(DEFAULT_RECENT_N)
    }
}

impl SessionMemory {
    pub fn new(recent_n: usize) -> Self {
        Self {
            turns: Vec::new(),
            summary: String::new(),
            recent_n: recent_n.max(1),
            chunks: Vec::new(),
            current_chunk: None,
            previous_chunk: None,
            pending_clarify: None,
        }
    }

    pub fn append(&mut self, user: impl Into<String>, assistant: impl Into<String>) {
        self.turns.push(Turn::new(user, assistant));
        if let Some(i) = self.current_chunk {
            if let Some(chunk) = self.chunks.get_mut(i) {
                chunk.turn_end = self.turns.len();
            }
        }
        self.refresh_summary();
    }

    pub fn recent(&self) -> &[Turn] {
        let start = self.turns.len().saturating_sub(self.recent_n);
        &self.turns[start..]
    }

    pub fn last_user(&self) -> Option<&str> {
        self.turns.last().map(|t| t.user.as_str())
    }

    pub fn refresh_summary(&mut self) {
        self.summary = simple_summary(&self.turns, self.recent_n, 120);
    }

    /// Open a new topic chunk and make it current.
    pub fn open_chunk(&mut self, summary: impl Into<String>, embedding: Vec<f32>) -> usize {
        let id = self.chunks.last().map(|c| c.id + 1).unwrap_or(0);
        let idx = self.chunks.len();
        self.chunks
            .push(TopicChunk::new(id, summary, embedding, self.turns.len()));
        self.set_current_chunk(idx);
        idx
    }

    /// Switch the active topic to an existing chunk (topic return).
    pub fn return_to_chunk(&mut self, index: usize) -> bool {
        if index < self.chunks.len() {
            self.set_current_chunk(index);
            true
        } else {
            false
        }
    }

    /// Update `current_chunk`, remembering the prior one as `previous_chunk`.
    pub fn set_current_chunk(&mut self, index: usize) {
        if self.current_chunk == Some(index) {
            return;
        }
        self.previous_chunk = self.current_chunk;
        self.current_chunk = Some(index);
        self.pending_clarify = None;
    }

    pub fn set_pending_clarify(&mut self, pending: PendingClarification) {
        self.pending_clarify = Some(pending);
    }

    pub fn clear_pending_clarify(&mut self) {
        self.pending_clarify = None;
    }

    pub fn chunk_labels(&self) -> Vec<String> {
        self.chunks.iter().map(|c| c.summary.clone()).collect()
    }

    pub fn current_embedding(&self) -> Option<&[f32]> {
        let i = self.current_chunk?;
        self.chunks.get(i).map(|c| c.embedding.as_slice())
    }

    /// Past chunks excluding the current one (for S1 return scoring).
    pub fn past_chunk_embeddings(&self) -> Vec<(usize, &[f32])> {
        let cur = self.current_chunk;
        self.chunks
            .iter()
            .enumerate()
            .filter(|(i, _)| Some(*i) != cur)
            .map(|(i, c)| (i, c.embedding.as_slice()))
            .collect()
    }

    /// Text suitable for a LiteRT-LM system message `content` field.
    ///
    /// Prefer [`crate::compile`] per turn (P4). This helper is a cold-start
    /// resident-only snapshot (Continue switch).
    pub fn system_preamble(&self) -> Option<String> {
        let cfg = crate::CompilerConfig {
            // Cold-start preamble: summary + topic only (Conversation will see live turns).
            recent_turn_window: 0,
            ..crate::CompilerConfig::default()
        };
        crate::compile(self, crate::TopicSwitch::Continue, &cfg).render_notes(&cfg)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, SessionMemoryError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(path)?;
        let mut mem: Self = serde_json::from_str(&raw)?;
        if mem.recent_n == 0 {
            mem.recent_n = DEFAULT_RECENT_N;
        }
        if mem.summary.is_empty() && !mem.turns.is_empty() {
            mem.refresh_summary();
        }
        Ok(mem)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SessionMemoryError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let raw = serde_json::to_string_pretty(self)?;
        let tmp = PathBuf::from(format!("{}.tmp", path.display()));
        fs::write(&tmp, raw)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.turns.clear();
        self.summary.clear();
        self.chunks.clear();
        self.current_chunk = None;
        self.previous_chunk = None;
        self.pending_clarify = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn append_updates_recent_and_summary() {
        let mut mem = SessionMemory::new(2);
        mem.append("one", "a1");
        mem.append("two", "a2");
        mem.append("three", "a3");
        assert_eq!(mem.turns.len(), 3);
        assert_eq!(mem.recent().len(), 2);
        assert_eq!(mem.recent()[0].user, "two");
        assert!(mem.summary.contains("two"));
        assert!(!mem.summary.contains("one"));
        assert!(mem.system_preamble().unwrap().contains("loco-bot"));
    }

    #[test]
    fn round_trip_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("session.json");
        let mut mem = SessionMemory::new(4);
        mem.append("hello", "hi there");
        mem.save(&path).unwrap();
        let loaded = SessionMemory::load(&path).unwrap();
        assert_eq!(loaded.turns.len(), 1);
        assert_eq!(loaded.turns[0].user, "hello");
        assert!(!loaded.summary.is_empty());
    }

    #[test]
    fn load_missing_file_is_empty() {
        let dir = tempdir().unwrap();
        let mem = SessionMemory::load(dir.path().join("nope.json")).unwrap();
        assert!(mem.turns.is_empty());
    }

    #[test]
    fn open_chunk_and_return() {
        let mut mem = SessionMemory::new(4);
        let i0 = mem.open_chunk("subway", vec![1.0, 0.0]);
        mem.append("u1", "a1");
        assert_eq!(mem.chunks[i0].turn_end, 1);
        let i1 = mem.open_chunk("cooking", vec![0.0, 1.0]);
        assert_eq!(mem.current_chunk, Some(i1));
        assert!(mem.return_to_chunk(i0));
        assert_eq!(mem.current_chunk, Some(i0));
        assert_eq!(mem.previous_chunk, Some(i1));
        let past = mem.past_chunk_embeddings();
        assert_eq!(past.len(), 1);
        assert_eq!(past[0].0, i1);
        let preamble = mem.system_preamble().unwrap();
        assert!(preamble.contains("Active topic: subway"));
    }

    #[test]
    fn loads_legacy_json_without_chunks() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("legacy.json");
        fs::write(
            &path,
            r#"{"turns":[{"user":"hi","assistant":"yo","at":"1"}],"summary":"1. User: hi","recent_n":8}"#,
        )
        .unwrap();
        let mem = SessionMemory::load(&path).unwrap();
        assert!(mem.chunks.is_empty());
        assert_eq!(mem.turns.len(), 1);
    }
}
