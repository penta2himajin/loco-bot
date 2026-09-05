//! Thin LiteRT-LM chat session (plain text, streaming).

use std::io::{self, Write};
use std::path::Path;

use litertlm_rs::{
    extract_text, set_min_log_level, ConversationConfig, Engine, EngineSettings, LogSeverity,
};
use thiserror::Error;

use crate::backend::InferenceBackend;
use crate::prompt::user_message_json;

#[derive(Debug, Error)]
pub enum ChatError {
    #[error("model file not found: {0}")]
    ModelMissing(String),
    #[error("liteRT-LM error: {0}")]
    LiteRt(String),
    #[error("io error: {0}")]
    Io(#[from] io::Error),
}

impl From<litertlm_rs::Error> for ChatError {
    fn from(value: litertlm_rs::Error) -> Self {
        Self::LiteRt(value.to_string())
    }
}

/// Loaded engine + conversation for multi-turn chat.
pub struct ChatSession {
    // Drop conversation before engine to avoid LiteRT teardown warnings.
    conversation: Option<litertlm_rs::Conversation>,
    engine: Option<Engine>,
}

impl Drop for ChatSession {
    fn drop(&mut self) {
        self.conversation.take();
        self.engine.take();
    }
}

impl ChatSession {
    /// Load `.litertlm` and create a conversation.
    ///
    /// `system_text`, when set, becomes the conversation system message content
    /// (used to inject a thin memory preamble).
    pub fn open(
        model_path: impl AsRef<Path>,
        backend: InferenceBackend,
        system_text: Option<&str>,
    ) -> Result<Self, ChatError> {
        let path = model_path.as_ref();
        if !path.is_file() {
            return Err(ChatError::ModelMissing(path.display().to_string()));
        }

        set_min_log_level(LogSeverity::Error);

        let settings = EngineSettings::new(
            path.to_string_lossy().as_ref(),
            backend.as_str(),
            None,
            None,
        )?;
        let engine = Engine::new(&settings)?;

        let conversation = if let Some(system) = system_text.filter(|s| !s.is_empty()) {
            let mut config = ConversationConfig::new()?;
            let system_json = serde_json::json!({
                "role": "system",
                "content": system,
            })
            .to_string();
            config.set_system_message(&system_json)?;
            engine.create_conversation_with_config(&config)?
        } else {
            engine.create_conversation()?
        };

        Ok(Self {
            conversation: Some(conversation),
            engine: Some(engine),
        })
    }

    fn conversation(&self) -> Result<&litertlm_rs::Conversation, ChatError> {
        self.conversation
            .as_ref()
            .ok_or_else(|| ChatError::LiteRt("conversation already closed".into()))
    }

    /// Stream a reply for one user turn; `on_text` receives decoded text deltas.
    pub fn reply_stream<F>(&mut self, user_text: &str, mut on_text: F) -> Result<(), ChatError>
    where
        F: FnMut(&str),
    {
        let message_json = user_message_json(user_text);
        self.conversation()?
            .send_message_stream(&message_json, |chunk| {
                if let Some(err) = chunk.error() {
                    eprintln!("\n[stream error] {err}");
                    return;
                }
                if let Some(raw) = chunk.text() {
                    let text = extract_text(&raw);
                    if !text.is_empty() {
                        on_text(&text);
                    }
                }
            })?;
        Ok(())
    }

    /// Non-streaming convenience: collect the full reply text.
    pub fn reply(&mut self, user_text: &str) -> Result<String, ChatError> {
        let mut out = String::new();
        self.reply_stream(user_text, |delta| out.push_str(delta))?;
        Ok(out)
    }

    /// Stream to stdout (CLI helper).
    pub fn reply_to_stdout(&mut self, user_text: &str) -> Result<(), ChatError> {
        let mut printed = false;
        self.reply_stream(user_text, |delta| {
            print!("{delta}");
            let _ = io::stdout().flush();
            printed = true;
        })?;
        if printed {
            println!();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::InferenceBackend;

    #[test]
    fn open_missing_model_errors() {
        let result = ChatSession::open("/no/such/model.litertlm", InferenceBackend::Cpu, None);
        assert!(matches!(result, Err(ChatError::ModelMissing(_))));
    }
}
