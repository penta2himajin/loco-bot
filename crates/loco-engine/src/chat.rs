//! Thin LiteRT-LM chat session (plain text, optional tools).

use std::collections::VecDeque;
use std::io;
use std::path::Path;

use litertlm_rs::{
    extract_text, set_min_log_level, ConversationConfig, Engine, EngineSettings, LogSeverity,
};
use serde_json::{json, Value};
use thiserror::Error;

use crate::backend::InferenceBackend;
use crate::prompt::user_message_json;
use crate::tools::{
    extract_assistant_text, extract_tool_calls, tool_response_json, ToolCall, ToolHost,
    MAX_TOOL_ROUNDS,
};

#[derive(Debug, Error)]
pub enum ChatError {
    #[error("model file not found: {0}")]
    ModelMissing(String),
    #[error("liteRT-LM error: {0}")]
    LiteRt(String),
    #[error("tool loop exceeded {0} rounds")]
    ToolLoopExceeded(usize),
    #[error("io error: {0}")]
    Io(#[from] io::Error),
}

impl From<litertlm_rs::Error> for ChatError {
    fn from(value: litertlm_rs::Error) -> Self {
        Self::LiteRt(value.to_string())
    }
}

/// Result of one (segment of a) tool-enabled turn.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolsTurnProgress {
    /// Model produced final assistant text.
    Done(String),
    /// Surface must run this tool and call [`ChatSession::resume_with_host_tool_result`].
    NeedHostTool {
        call: ToolCall,
        /// Same-batch calls still pending after `call`.
        queued: Vec<ToolCall>,
        rounds_used: usize,
    },
}

/// Inputs for [`ChatSession::resume_with_host_tool_result`].
pub struct HostToolResume<'a> {
    pub call: &'a ToolCall,
    pub response: Value,
    pub queued: Vec<ToolCall>,
    pub rounds_used: usize,
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
    /// `tools_json`, when set, is an OpenAI-style tools array for function calling.
    pub fn open(
        model_path: impl AsRef<Path>,
        backend: InferenceBackend,
        system_text: Option<&str>,
        tools_json: Option<&str>,
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

        let mut session = Self {
            conversation: None,
            engine: Some(engine),
        };
        session.replace_conversation(system_text, tools_json)?;
        Ok(session)
    }

    /// Replace the conversation (keeps the loaded engine). Used after `configure`
    /// merges host tool schemas. Clears chat history.
    pub fn replace_conversation(
        &mut self,
        system_text: Option<&str>,
        tools_json: Option<&str>,
    ) -> Result<(), ChatError> {
        self.conversation.take();
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| ChatError::LiteRt("engine already closed".into()))?;

        let system = system_text.filter(|s| !s.is_empty());
        let tools = tools_json.filter(|s| !s.is_empty());
        let conversation = if system.is_some() || tools.is_some() {
            let mut config = ConversationConfig::new()?;
            if let Some(sys) = system {
                let system_json = serde_json::json!({
                    "role": "system",
                    "content": sys,
                })
                .to_string();
                config.set_system_message(&system_json)?;
            }
            if let Some(tools_json) = tools {
                config.set_tools(tools_json)?;
            }
            engine.create_conversation_with_config(&config)?
        } else {
            engine.create_conversation()?
        };
        self.conversation = Some(conversation);
        Ok(())
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

    /// Agent loop: send user → execute tool_calls → feed tool results → final text.
    pub fn reply_with_tools(
        &mut self,
        user_text: &str,
        host: &ToolHost,
    ) -> Result<String, ChatError> {
        match self.reply_with_tools_routed(
            user_text,
            host,
            |_| false,
            &mut AlwaysAllow,
            |_, _, _| {},
        )? {
            ToolsTurnProgress::Done(text) => Ok(text),
            ToolsTurnProgress::NeedHostTool { call, .. } => Err(ChatError::LiteRt(format!(
                "unexpected host tool pause for {}",
                call.name
            ))),
        }
    }

    /// Agent loop with a consent gate before each tool execution.
    pub fn reply_with_tools_consent<C>(
        &mut self,
        user_text: &str,
        host: &ToolHost,
        consent: &mut C,
        mut on_tool: impl FnMut(&str, &serde_json::Map<String, serde_json::Value>, bool),
    ) -> Result<String, ChatError>
    where
        C: crate::tools::ToolConsentGate,
    {
        match self.reply_with_tools_routed(user_text, host, |_| false, consent, &mut on_tool)? {
            ToolsTurnProgress::Done(text) => Ok(text),
            ToolsTurnProgress::NeedHostTool { call, .. } => Err(ChatError::LiteRt(format!(
                "unexpected host tool pause for {}",
                call.name
            ))),
        }
    }

    /// Like [`Self::reply_with_tools_consent`], but pauses when `is_host_tool` is true.
    pub fn reply_with_tools_routed<C, H>(
        &mut self,
        user_text: &str,
        host: &ToolHost,
        is_host_tool: H,
        consent: &mut C,
        mut on_tool: impl FnMut(&str, &serde_json::Map<String, serde_json::Value>, bool),
    ) -> Result<ToolsTurnProgress, ChatError>
    where
        C: crate::tools::ToolConsentGate,
        H: Fn(&str) -> bool,
    {
        let message_json = user_message_json(user_text);
        let raw = self.conversation()?.send_message(&message_json)?;
        self.drive_tool_loop(raw, host, &is_host_tool, consent, &mut on_tool, 0)
    }

    /// Resume after a surface fulfilled a [`ToolsTurnProgress::NeedHostTool`].
    pub fn resume_with_host_tool_result<C, H>(
        &mut self,
        resume: HostToolResume<'_>,
        host: &ToolHost,
        is_host_tool: H,
        consent: &mut C,
        mut on_tool: impl FnMut(&str, &serde_json::Map<String, serde_json::Value>, bool),
    ) -> Result<ToolsTurnProgress, ChatError>
    where
        C: crate::tools::ToolConsentGate,
        H: Fn(&str) -> bool,
    {
        let HostToolResume {
            call,
            response,
            queued,
            rounds_used,
        } = resume;
        let tool_msg = tool_response_json(&call.name, &response, call.id.as_deref());
        let mut raw = self.conversation()?.send_message(&tool_msg)?;
        let mut queue: VecDeque<ToolCall> = queued.into();
        while let Some(next) = queue.pop_front() {
            if is_host_tool(&next.name) {
                return Ok(ToolsTurnProgress::NeedHostTool {
                    call: next,
                    queued: queue.into(),
                    rounds_used,
                });
            }
            let allowed = consent.allow_tool(&next.name, &next.arguments);
            on_tool(&next.name, &next.arguments, allowed);
            let response = if allowed {
                match host.execute(&next.name, &next.arguments) {
                    Ok(v) => v,
                    Err(err) => json!({ "error": err.to_string() }),
                }
            } else {
                json!({ "error": "tool denied by consent policy", "tool": next.name })
            };
            let tool_msg = tool_response_json(&next.name, &response, next.id.as_deref());
            raw = self.conversation()?.send_message(&tool_msg)?;
        }
        self.drive_tool_loop(raw, host, &is_host_tool, consent, &mut on_tool, rounds_used)
    }

    fn drive_tool_loop<C, H>(
        &mut self,
        mut raw: String,
        host: &ToolHost,
        is_host_tool: &H,
        consent: &mut C,
        on_tool: &mut impl FnMut(&str, &serde_json::Map<String, serde_json::Value>, bool),
        mut rounds_used: usize,
    ) -> Result<ToolsTurnProgress, ChatError>
    where
        C: crate::tools::ToolConsentGate,
        H: Fn(&str) -> bool,
    {
        loop {
            let calls = extract_tool_calls(&raw);
            if calls.is_empty() {
                let text = extract_assistant_text(&raw);
                if !text.is_empty() {
                    return Ok(ToolsTurnProgress::Done(text));
                }
                let fallback = extract_text(&raw);
                return Ok(ToolsTurnProgress::Done(fallback));
            }

            if rounds_used >= MAX_TOOL_ROUNDS {
                return Err(ChatError::ToolLoopExceeded(MAX_TOOL_ROUNDS));
            }
            rounds_used += 1;

            let mut iter = calls.into_iter();
            while let Some(call) = iter.next() {
                if is_host_tool(&call.name) {
                    let queued: Vec<ToolCall> = iter.collect();
                    return Ok(ToolsTurnProgress::NeedHostTool {
                        call,
                        queued,
                        rounds_used,
                    });
                }
                let allowed = consent.allow_tool(&call.name, &call.arguments);
                on_tool(&call.name, &call.arguments, allowed);
                let response = if allowed {
                    match host.execute(&call.name, &call.arguments) {
                        Ok(v) => v,
                        Err(err) => json!({ "error": err.to_string() }),
                    }
                } else {
                    json!({ "error": "tool denied by consent policy", "tool": call.name })
                };
                eprintln!("[tool] {}", call.name);
                let tool_msg = tool_response_json(&call.name, &response, call.id.as_deref());
                raw = self.conversation()?.send_message(&tool_msg)?;
            }
        }
    }
}

struct AlwaysAllow;

impl crate::tools::ToolConsentGate for AlwaysAllow {
    fn allow_tool(&mut self, _name: &str, _args: &serde_json::Map<String, Value>) -> bool {
        true
    }
}

/// Ensure every tool call has an id (surfaces need stable `call_id`s).
pub fn ensure_tool_call_id(call: &mut ToolCall, fallback: impl Into<String>) {
    if call.id.as_ref().map(|s| s.is_empty()).unwrap_or(true) {
        call.id = Some(fallback.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::InferenceBackend;

    #[test]
    fn open_missing_model_errors() {
        let result =
            ChatSession::open("/no/such/model.litertlm", InferenceBackend::Cpu, None, None);
        assert!(matches!(result, Err(ChatError::ModelMissing(_))));
    }

    #[test]
    fn ensure_tool_call_id_fills_missing() {
        let mut call = ToolCall {
            name: "open_url".into(),
            arguments: Default::default(),
            id: None,
        };
        ensure_tool_call_id(&mut call, "c-1");
        assert_eq!(call.id.as_deref(), Some("c-1"));
    }
}
