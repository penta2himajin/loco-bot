//! JSONL serve protocol: surfaces advertise host tools and fulfill `tool_result`s.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::events::TurnOutcome;

/// Where a tool runs. Surfaces own `host`; loco-bot keeps portable `local` tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecKind {
    Host,
    Local,
}

/// Tool schema advertised by a surface via `configure`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostToolSpec {
    pub name: String,
    pub description: String,
    #[serde(default = "empty_object")]
    pub parameters: Value,
    #[serde(default = "default_risk")]
    pub risk: String,
    #[serde(default = "default_host_exec")]
    pub exec: ToolExecKind,
}

fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

fn default_risk() -> String {
    "medium".into()
}

fn default_host_exec() -> ToolExecKind {
    ToolExecKind::Host
}

impl HostToolSpec {
    /// OpenAI Chat Completions function-tool object for `set_tools`.
    pub fn to_openai_tool(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// Inbound line from a surface → `loco serve`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ServeClientMessage {
    Configure {
        tools: Vec<HostToolSpec>,
    },
    Turn {
        user: String,
    },
    ToolResult {
        call_id: String,
        ok: bool,
        #[serde(default = "empty_object")]
        content: Value,
    },
}

/// Parse one stdin line: tagged `op` messages, or legacy `{"user":…}`.
pub fn parse_serve_client_line(line: &str) -> Result<ServeClientMessage, serde_json::Error> {
    let value: Value = serde_json::from_str(line)?;
    if value.get("op").is_some() {
        return serde_json::from_value(value);
    }
    if let Some(user) = value.get("user").and_then(|u| u.as_str()) {
        return Ok(ServeClientMessage::Turn {
            user: user.to_string(),
        });
    }
    serde_json::from_value(value)
}

/// Outbound line from `loco serve` → surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ServeServerMessage {
    Done {
        #[serde(flatten)]
        outcome: TurnOutcome,
    },
    AwaitingTool {
        call_id: String,
        #[serde(flatten)]
        outcome: TurnOutcome,
    },
}

impl ServeServerMessage {
    pub fn done(outcome: TurnOutcome) -> Self {
        Self::Done { outcome }
    }

    pub fn awaiting_tool(call_id: String, outcome: TurnOutcome) -> Self {
        Self::AwaitingTool { call_id, outcome }
    }
}

/// Merge portable default tools JSON array with host specs (host names win on clash).
pub fn merge_tools_json(
    defaults_json: &str,
    host: &[HostToolSpec],
) -> Result<String, serde_json::Error> {
    let mut arr: Vec<Value> = serde_json::from_str(defaults_json)?;
    let host_names: std::collections::HashSet<&str> =
        host.iter().map(|t| t.name.as_str()).collect();
    arr.retain(|tool| {
        tool.pointer("/function/name")
            .and_then(|n| n.as_str())
            .map(|n| !host_names.contains(n))
            .unwrap_or(true)
    });
    for spec in host {
        arr.push(spec.to_openai_tool());
    }
    serde_json::to_string(&arr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{AgentEvent, TurnOutcome};

    fn open_url_spec() -> HostToolSpec {
        HostToolSpec {
            name: "open_url".into(),
            description: "Open a URL".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "url": { "type": "string" } },
                "required": ["url"]
            }),
            risk: "medium".into(),
            exec: ToolExecKind::Host,
        }
    }

    #[test]
    fn configure_round_trips() {
        let msg = ServeClientMessage::Configure {
            tools: vec![open_url_spec()],
        };
        let raw = serde_json::to_string(&msg).unwrap();
        assert!(raw.contains("\"op\":\"configure\""));
        let back: ServeClientMessage = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn legacy_user_line_parses_as_turn() {
        let msg = parse_serve_client_line(r#"{"user":"hello"}"#).unwrap();
        assert_eq!(
            msg,
            ServeClientMessage::Turn {
                user: "hello".into()
            }
        );
    }

    #[test]
    fn awaiting_tool_includes_call_id_and_events() {
        let outcome = TurnOutcome {
            events: vec![AgentEvent::ToolRequest {
                name: "open_url".into(),
                arguments: serde_json::Map::from_iter([(
                    "url".into(),
                    Value::String("https://example.com".into()),
                )]),
                risk: "medium".into(),
                call_id: Some("c1".into()),
            }],
            reply_text: String::new(),
        };
        let msg = ServeServerMessage::awaiting_tool("c1".into(), outcome);
        let raw = serde_json::to_string(&msg).unwrap();
        assert!(raw.contains("\"status\":\"awaiting_tool\""));
        assert!(raw.contains("\"call_id\":\"c1\""));
        let back: ServeServerMessage = serde_json::from_str(&raw).unwrap();
        match back {
            ServeServerMessage::AwaitingTool { call_id, .. } => assert_eq!(call_id, "c1"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn merge_tools_appends_host_and_dedupes_name() {
        let defaults = r#"[{"type":"function","function":{"name":"get_current_time","description":"t","parameters":{}}},{"type":"function","function":{"name":"open_url","description":"old","parameters":{}}}]"#;
        let merged = merge_tools_json(defaults, &[open_url_spec()]).unwrap();
        let arr: Vec<Value> = serde_json::from_str(&merged).unwrap();
        let names: Vec<_> = arr
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(|n| n.as_str()))
            .collect();
        assert_eq!(names, vec!["get_current_time", "open_url"]);
        assert!(merged.contains("Open a URL"));
    }
}
