//! Prompt helpers shared by CLI and (later) higher layers.

/// Build the LiteRT-LM Conversation message JSON for a plain user turn.
pub fn user_message_json(text: &str) -> String {
    serde_json::json!({
        "role": "user",
        "content": text,
    })
    .to_string()
}

/// Attach optional session notes in front of the user text.
///
/// Small on-device models often honor in-band context more reliably than a
/// separate system message alone.
pub fn with_session_notes(notes: Option<&str>, user_text: &str) -> String {
    match notes.map(str::trim).filter(|s| !s.is_empty()) {
        Some(notes) => format!(
            "Session notes from earlier turns:\n{notes}\n\nCurrent user message:\n{user_text}"
        ),
        None => user_text.to_string(),
    }
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

    #[test]
    fn prepends_notes_when_present() {
        let out = with_session_notes(Some("1. user: teal"), "what color?");
        assert!(out.contains("teal"));
        assert!(out.contains("what color?"));
        assert_eq!(with_session_notes(None, "x"), "x");
    }
}
