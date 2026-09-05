//! Prompt helpers shared by CLI and (later) higher layers.

/// Build the LiteRT-LM Conversation message JSON for a plain user turn.
pub fn user_message_json(text: &str) -> String {
    serde_json::json!({
        "role": "user",
        "content": text,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_quotes_and_newlines() {
        let json = user_message_json("say \"hi\"\nplease");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "say \"hi\"\nplease");
    }
}
