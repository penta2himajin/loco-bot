//! Tool consent gate for agent turns.

use serde_json::{Map, Value};

use loco_engine::ToolConsentGate;

/// Risk class for tool execution prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ToolRisk {
    /// Built-in low-impact tools (clock, notes, stats).
    #[default]
    Low,
    /// Local filesystem or similar.
    Medium,
    /// Network / OAuth / destructive writes.
    High,
}

impl ToolRisk {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    pub fn for_tool(name: &str) -> Self {
        match name {
            "get_current_time" | "note_write" | "note_read" | "session_stats" => Self::Low,
            "fs_list" | "fs_move" | "fs_rename" => Self::Medium,
            "web_search" | "mail_list" | "drive_list" => Self::High,
            _ => Self::High,
        }
    }
}

/// Decide whether a tool may run for this turn.
pub trait ToolConsent {
    fn allow(&mut self, name: &str, args: &Map<String, Value>, risk: ToolRisk) -> bool;
}

/// Bridge [`ToolConsent`] into [`loco_engine::ChatSession::reply_with_tools_consent`].
pub struct ConsentBridge<'a> {
    pub inner: &'a mut dyn ToolConsent,
}

impl ToolConsentGate for ConsentBridge<'_> {
    fn allow_tool(&mut self, name: &str, args: &Map<String, Value>) -> bool {
        let risk = ToolRisk::for_tool(name);
        self.inner.allow(name, args, risk)
    }
}

/// Allows only an explicit name list (tests / locked-down shells).
#[derive(Debug, Clone)]
pub struct AllowListedTools {
    pub names: Vec<String>,
}

impl ToolConsent for AllowListedTools {
    fn allow(&mut self, name: &str, _args: &Map<String, Value>, _risk: ToolRisk) -> bool {
        self.names.iter().any(|n| n == name)
    }
}

/// Auto-allow tools at or below a risk ceiling.
#[derive(Debug, Clone, Copy)]
pub struct AllowUpTo {
    pub max: ToolRisk,
}

impl Default for AllowUpTo {
    fn default() -> Self {
        Self { max: ToolRisk::Low }
    }
}

impl ToolConsent for AllowUpTo {
    fn allow(&mut self, _name: &str, _args: &Map<String, Value>, risk: ToolRisk) -> bool {
        risk <= self.max
    }
}

/// Auto-allow low risk; deny medium/high (CLI default until interactive prompts land).
#[derive(Debug, Default, Clone, Copy)]
pub struct AllowLowRiskOnly;

impl ToolConsent for AllowLowRiskOnly {
    fn allow(&mut self, name: &str, args: &Map<String, Value>, risk: ToolRisk) -> bool {
        AllowUpTo { max: ToolRisk::Low }.allow(name, args, risk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_tools_are_low_risk() {
        assert_eq!(ToolRisk::for_tool("get_current_time"), ToolRisk::Low);
        assert_eq!(ToolRisk::for_tool("web_search"), ToolRisk::High);
    }

    #[test]
    fn allow_low_risk_only_denies_high() {
        let mut gate = AllowLowRiskOnly;
        assert!(gate.allow("note_read", &Map::new(), ToolRisk::Low));
        assert!(!gate.allow("web_search", &Map::new(), ToolRisk::High));
    }
}
