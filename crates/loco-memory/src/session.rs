//! In-memory + JSON file session store.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMemory {
    pub turns: Vec<Turn>,
    /// Rolling text summary of recent turns (no LLM).
    pub summary: String,
    /// How many turns to keep in the recent window / summary.
    pub recent_n: usize,
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
        }
    }

    pub fn append(&mut self, user: impl Into<String>, assistant: impl Into<String>) {
        self.turns.push(Turn::new(user, assistant));
        self.refresh_summary();
    }

    pub fn recent(&self) -> &[Turn] {
        let start = self.turns.len().saturating_sub(self.recent_n);
        &self.turns[start..]
    }

    pub fn refresh_summary(&mut self) {
        self.summary = simple_summary(&self.turns, self.recent_n, 120);
    }

    /// Text suitable for a LiteRT-LM system message `content` field.
    pub fn system_preamble(&self) -> Option<String> {
        if self.summary.is_empty() {
            return None;
        }
        Some(format!(
            "You are loco-bot, a local on-device assistant. Prior session notes (most recent first is not required; numbered chronologically):\n{}",
            self.summary
        ))
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
}
