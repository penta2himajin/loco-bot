//! Parse LiteRT-LM / Gemma 4 assistant messages for tool_calls.

use serde_json::{json, Map, Value};

/// Safety cap for the tool-call → execute → continue loop.
pub const MAX_TOOL_ROUNDS: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
    pub id: Option<String>,
}

/// Extract OpenAI-style tool calls from an assistant message JSON string.
///
/// Accepts top-level `tool_calls`, multimodal `content[].type == tool_call`,
/// and concatenated `{...}{...}` documents. Strips Gemma 4 `<|"|>` escapes.
pub fn extract_tool_calls(raw: &str) -> Vec<ToolCall> {
    let mut out = Vec::new();
    for fragment in split_concatenated_json(raw) {
        let Ok(value) = serde_json::from_str::<Value>(fragment) else {
            continue;
        };
        harvest_calls(&value, &mut out);
    }
    if out.is_empty() && raw.contains("<|tool_call>") {
        harvest_raw_token_calls(raw, &mut out);
    }
    out
}

/// Plain text from an assistant message (empty when the turn is tool-only).
pub fn extract_assistant_text(raw: &str) -> String {
    for fragment in split_concatenated_json(raw) {
        if let Ok(value) = serde_json::from_str::<Value>(fragment) {
            let text = text_from_message(&value);
            if !text.is_empty() {
                return text;
            }
        }
    }
    if raw.trim_start().starts_with('{') {
        String::new()
    } else {
        raw.to_string()
    }
}

/// Build a `role: tool` message for the next Conversation turn.
pub fn tool_response_json(name: &str, response: &Value, tool_call_id: Option<&str>) -> String {
    let mut msg = json!({
        "role": "tool",
        "content": [{
            "name": name,
            "response": response,
        }],
    });
    if let Some(id) = tool_call_id {
        msg.as_object_mut()
            .unwrap()
            .insert("tool_call_id".into(), Value::String(id.to_string()));
    }
    msg.to_string()
}

fn text_from_message(value: &Value) -> String {
    match value.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn harvest_calls(value: &Value, out: &mut Vec<ToolCall>) {
    if let Some(Value::Array(calls)) = value.get("tool_calls") {
        for call in calls {
            if let Some(tc) = call_from_object(call) {
                out.push(tc);
            }
        }
    }
    if let Some(Value::Array(blocks)) = value.get("content") {
        for block in blocks {
            if block.get("type").and_then(|t| t.as_str()) == Some("tool_call") {
                if let Some(tc) = call_from_object(block.get("tool_call").unwrap_or(block)) {
                    out.push(tc);
                }
            }
        }
    }
}

fn call_from_object(raw: &Value) -> Option<ToolCall> {
    let Value::Object(map) = raw else {
        return None;
    };
    let fn_obj = if map.get("type").and_then(|t| t.as_str()) == Some("function") {
        map.get("function")?.as_object()?
    } else {
        map
    };
    let name = fn_obj.get("name")?.as_str()?.to_string();
    let arguments = normalize_arguments(fn_obj.get("arguments"));
    let id = map
        .get("id")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or_else(|| {
            fn_obj
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        });
    Some(ToolCall {
        name,
        arguments,
        id,
    })
}

fn normalize_arguments(raw: Option<&Value>) -> Map<String, Value> {
    let cleaned = strip_escape_tokens(raw.cloned().unwrap_or(Value::Object(Map::new())));
    match cleaned {
        Value::Object(m) => m,
        Value::String(s) => serde_json::from_str(&s)
            .ok()
            .and_then(|v: Value| v.as_object().cloned())
            .unwrap_or_default(),
        _ => Map::new(),
    }
}

fn strip_escape_tokens(value: Value) -> Value {
    match value {
        Value::String(s) => Value::String(s.replace("<|\"|>", "")),
        Value::Array(arr) => Value::Array(arr.into_iter().map(strip_escape_tokens).collect()),
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, strip_escape_tokens(v)))
                .collect(),
        ),
        other => other,
    }
}

fn split_concatenated_json(input: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start: Option<usize> = None;
    let mut in_string = false;
    let mut escape = false;
    for (i, c) in input.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' {
            escape = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if c == '{' {
            if depth == 0 {
                start = Some(i);
            }
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                if let Some(s) = start {
                    out.push(&input[s..=i]);
                }
                start = None;
            }
        }
    }
    if out.is_empty() && !input.trim().is_empty() {
        out.push(input.trim());
    }
    out
}

fn harvest_raw_token_calls(text: &str, out: &mut Vec<ToolCall>) {
    // `<|tool_call>call:NAME{key:<|"|>value<|"|>,...}<tool_call|>`
    let mut rest = text;
    while let Some(open) = rest.find("<|tool_call>") {
        rest = &rest[open + "<|tool_call>".len()..];
        let body = if let Some(close) = rest.find("<tool_call|>") {
            let b = &rest[..close];
            rest = &rest[close + "<tool_call|>".len()..];
            b
        } else {
            let b = rest;
            rest = "";
            b
        };
        let Some(name_end) = body.find('{') else {
            continue;
        };
        let name_part = body[..name_end].trim();
        let name = name_part
            .strip_prefix("call:")
            .unwrap_or(name_part)
            .trim()
            .to_string();
        if name.is_empty() {
            continue;
        }
        let params = &body[name_end..];
        let mut arguments = Map::new();
        // key:<|"|>value<|"|>
        let mut p = params.trim_start_matches('{').trim_end_matches('}');
        while let Some(colon) = p.find(":<|\"|>") {
            let key = p[..colon].rsplit([',', '{']).next().unwrap_or("").trim();
            p = &p[colon + ":<|\"|>".len()..];
            if let Some(end) = p.find("<|\"|>") {
                let val = p[..end].to_string();
                if !key.is_empty() {
                    arguments.insert(key.to_string(), Value::String(val));
                }
                p = &p[end + "<|\"|>".len()..];
            } else {
                break;
            }
        }
        out.push(ToolCall {
            name,
            arguments,
            id: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_tool_calls() {
        let raw = r#"{"role":"assistant","content":null,"tool_calls":[{"type":"function","function":{"name":"get_current_time","arguments":{}}}]}"#;
        let calls = extract_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_current_time");
        assert!(extract_assistant_text(raw).is_empty());
    }

    #[test]
    fn strips_escape_tokens_in_args() {
        let raw = r#"{"tool_calls":[{"type":"function","function":{"name":"note_write","arguments":{"key":"<|\"|>todo<|\"|>","text":"buy milk"}}}]}"#;
        let calls = extract_tool_calls(raw);
        assert_eq!(calls[0].arguments["key"], "todo");
    }

    #[test]
    fn tool_response_shape() {
        let s = tool_response_json(
            "get_current_time",
            &json!({"utc":"2026-01-01T00:00:00Z"}),
            None,
        );
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["role"], "tool");
        assert_eq!(v["content"][0]["name"], "get_current_time");
    }

    #[test]
    fn concatenated_json_documents() {
        let raw = r#"{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"a","arguments":{}}}]}{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"b","arguments":{}}}]}"#;
        let calls = extract_tool_calls(raw);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }
}
