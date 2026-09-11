//! Events emitted during one agent turn (for any UI shell).

use serde::{Deserialize, Serialize};

/// One turn's collected outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnOutcome {
    pub events: Vec<AgentEvent>,
    /// Final user-visible text (clarify question, ack, or model reply).
    pub reply_text: String,
}

/// Streaming-friendly events produced while handling a user turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Topic clarification required before calling the chat model.
    Clarify {
        question: String,
        choices: Vec<ClarifyChoice>,
    },
    /// Short acknowledgement (e.g. after bare clarify reply).
    Ack {
        text: String,
    },
    /// Topic switch applied for this turn.
    Topic {
        kind: String,
        chunk_index: Option<usize>,
    },
    /// Context notes were injected into the model prompt.
    Context {
        chars: usize,
        has_dynamic: bool,
    },
    /// Model requested a tool; shells may show this before/after consent.
    ToolRequest {
        name: String,
        arguments: serde_json::Map<String, serde_json::Value>,
        risk: String,
        /// Present when the surface must fulfill this call (`awaiting_tool`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
    },
    ToolResult {
        name: String,
        ok: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
    },
    /// Incremental model text (reserved; non-streaming turns may omit).
    Token {
        delta: String,
    },
    Done {
        text: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClarifyChoice {
    pub label: String,
    /// 1-based index for numbered replies, when applicable.
    pub index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clarify_event_round_trips_json() {
        let ev = AgentEvent::Clarify {
            question: "どの話？".into(),
            choices: vec![ClarifyChoice {
                label: "地下鉄".into(),
                index: 1,
            }],
        };
        let raw = serde_json::to_string(&ev).unwrap();
        let back: AgentEvent = serde_json::from_str(&raw).unwrap();
        assert_eq!(ev, back);
        assert!(raw.contains("\"type\":\"clarify\""));
    }
}
