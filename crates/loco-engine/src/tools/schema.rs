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
        },
        {
            "type": "function",
            "function": {
                "name": "fs_list",
                "description": "List files and directories under the user-allowed filesystem sandbox root. Paths are relative to that root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative directory to list (default `.`)"
                        }
                    },
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fs_move",
                "description": "Move or rename a file/directory inside the filesystem sandbox. Prefer dry_run=true first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from": { "type": "string", "description": "Relative source path" },
                        "to": { "type": "string", "description": "Relative destination path" },
                        "dry_run": {
                            "type": "boolean",
                            "description": "If true, validate only and do not modify disk"
                        }
                    },
                    "required": ["from", "to"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fs_rename",
                "description": "Rename a file/directory inside the filesystem sandbox (alias of fs_move).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from": { "type": "string" },
                        "to": { "type": "string" },
                        "dry_run": { "type": "boolean" }
                    },
                    "required": ["from", "to"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web. Requires network and an explicit provider configuration; may be denied by consent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query" }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mail_list",
                "description": "List recent mail messages (OAuth). Not configured until P8 mail wiring lands.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": { "type": "integer", "description": "Max messages" }
                    },
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "drive_list",
                "description": "List cloud drive files (OAuth). Not configured until P8 drive wiring lands.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Folder path or id" }
                    },
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
        assert_eq!(arr.len(), 10);
        assert_eq!(arr[0]["function"]["name"], "get_current_time");
        assert_eq!(arr[4]["function"]["name"], "fs_list");
        assert_eq!(arr[7]["function"]["name"], "web_search");
    }
}
