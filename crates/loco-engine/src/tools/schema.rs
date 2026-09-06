//! OpenAI-style tool declarations for LiteRT-LM.

/// JSON array passed to `ConversationConfig::set_tools`.
pub fn default_tools_json() -> String {
    serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Return the current UTC date and time as ISO-8601. Use when the user asks what time or date it is.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "note_write",
                "description": "Store or overwrite a short durable note by key under the loco-bot notes file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Short note key, e.g. shopping-list"
                        },
                        "text": {
                            "type": "string",
                            "description": "Note body to store"
                        }
                    },
                    "required": ["key", "text"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "note_read",
                "description": "Read a previously stored note by key. Returns empty if missing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Note key to read"
                        }
                    },
                    "required": ["key"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "session_stats",
                "description": "Report how many turns and topic chunks are stored in the current loco-bot session memory file.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }
        }
    ])
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemas_are_valid_json_array() {
        let v: serde_json::Value = serde_json::from_str(&default_tools_json()).unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr[0]["function"]["name"], "get_current_time");
    }
}
