//! A single user/assistant exchange.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    pub user: String,
    pub assistant: String,
    /// RFC 3339 timestamp (UTC).
    pub at: String,
}

impl Turn {
    pub fn new(user: impl Into<String>, assistant: impl Into<String>) -> Self {
        Self {
            user: user.into(),
            assistant: assistant.into(),
            at: now_rfc3339(),
        }
    }
}

fn now_rfc3339() -> String {
    // Avoid chrono dependency for P2: format UNIX seconds as a stable marker.
    // Good enough for ordering; upgrade to chrono when we need true RFC3339.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_stores_text() {
        let t = Turn::new("hi", "hello");
        assert_eq!(t.user, "hi");
        assert_eq!(t.assistant, "hello");
        assert!(!t.at.is_empty());
    }
}
