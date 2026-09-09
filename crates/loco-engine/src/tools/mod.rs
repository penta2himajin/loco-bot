//! Small on-device tool surface for LiteRT-LM / Gemma 4 function calling.
//!
//! Tool schemas are OpenAI Chat Completions JSON (what
//! `ConversationConfig::set_tools` expects). Execution is in-process.

mod builtin;
mod parse;
mod schema;

pub use builtin::ToolHost;
pub use parse::{
    extract_assistant_text, extract_tool_calls, tool_response_json, ToolCall, MAX_TOOL_ROUNDS,
};
pub use schema::default_tools_json;

use serde_json::{Map, Value};

/// Minimal consent hook used by [`crate::ChatSession::reply_with_tools_consent`].
pub trait ToolConsentGate {
    fn allow_tool(&mut self, name: &str, args: &Map<String, Value>) -> bool;
}
