//! One request per process: versioned JSON on stdin, assistant message on stdout.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::ChatError;

pub(crate) struct ExternalSession {
    program: PathBuf,
    system: String,
    tools: Vec<Value>,
    history: Vec<Vec<Value>>,
}

fn error(message: impl Into<String>) -> ChatError {
    ChatError::External(message.into())
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Response {
    Reply(Reply),
    Calls(Calls),
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reply {
    content: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Calls {
    tool_calls: Vec<Call>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Call {
    name: String,
    arguments: Map<String, Value>,
}

impl ExternalSession {
    pub(crate) fn open(system: Option<&str>, tools: Option<&str>) -> Result<Self, ChatError> {
        let program = std::env::var_os("LOCO_INFERENCE_COMMAND")
            .filter(|p| !p.is_empty())
            .ok_or_else(|| {
                error("set LOCO_INFERENCE_COMMAND to an external inference executable")
            })?;
        Self::new(program.into(), system, tools)
    }

    fn new(program: PathBuf, system: Option<&str>, tools: Option<&str>) -> Result<Self, ChatError> {
        let tools = serde_json::from_str(tools.filter(|s| !s.is_empty()).unwrap_or("[]"))
            .map_err(|e| error(format!("invalid tool definitions: {e}")))?;
        Ok(Self {
            program,
            system: system.unwrap_or("").into(),
            tools,
            history: Vec::new(),
        })
    }

    pub(crate) fn replace_conversation(
        &mut self,
        system: Option<&str>,
        tools: Option<&str>,
    ) -> Result<(), ChatError> {
        *self = Self::new(self.program.clone(), system, tools)?;
        Ok(())
    }

    pub(crate) fn send_message(&mut self, message: &str) -> Result<String, ChatError> {
        let message: Value = serde_json::from_str(message).map_err(|e| error(e.to_string()))?;
        let mut history = self.history.clone();
        match message["role"].as_str() {
            Some("user") => history.push(vec![message]),
            Some("tool") => history
                .last_mut()
                .ok_or_else(|| error("tool result without a user turn"))?
                .push(message),
            _ => return Err(error("expected user message or tool result")),
        }
        // ponytail: last 10 user turns; adapters apply their model's finer token budget.
        while history.len() > 10 {
            history.remove(0);
        }
        let messages: Vec<_> = history.iter().flatten().collect();
        let request =
            json!({"version":1, "system":self.system, "messages":messages, "tools":self.tools});
        let raw = self.invoke(&request)?;
        let response: Response = serde_json::from_slice(&raw)
            .map_err(|e| error(format!("invalid external response: {e}")))?;
        let mut value = match response {
            Response::Reply(reply) if !reply.content.trim().is_empty() => {
                json!({"content":reply.content})
            }
            Response::Calls(calls) if calls.tool_calls.len() == 1 => {
                let call = &calls.tool_calls[0];
                if !self
                    .tools
                    .iter()
                    .any(|t| t["function"]["name"] == call.name)
                {
                    return Err(error(format!("undeclared external tool: {}", call.name)));
                }
                json!({"tool_calls":[{"name":call.name,"arguments":call.arguments}]})
            }
            _ => return Err(error("expected a nonempty answer or exactly one tool call")),
        };
        value["role"] = json!("assistant");
        history
            .last_mut()
            .expect("current turn")
            .push(value.clone());
        self.history = history;
        Ok(value.to_string())
    }

    fn invoke(&self, request: &Value) -> Result<Vec<u8>, ChatError> {
        // Anonymous files avoid pipe deadlocks and command-line size/quoting limits.
        let mut input = tempfile::tempfile()?;
        serde_json::to_writer(&mut input, request).map_err(|e| error(e.to_string()))?;
        input.seek(SeekFrom::Start(0))?;
        let mut output = tempfile::tempfile()?;
        let mut stderr = tempfile::tempfile()?;
        let mut child = Command::new(&self.program)
            .stdin(Stdio::from(input))
            .stdout(output.try_clone()?)
            .stderr(stderr.try_clone()?)
            .spawn()
            .map_err(|e| error(format!("{}: {e}", self.program.display())))?;
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if started.elapsed() >= Duration::from_secs(120) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error("external inference timed out after 120 seconds"));
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        if !status.success() {
            stderr.seek(SeekFrom::Start(0))?;
            let mut detail = String::new();
            stderr.take(8192).read_to_string(&mut detail)?;
            return Err(error(format!(
                "external inference failed ({status}): {}",
                detail.trim()
            )));
        }
        output.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::new();
        output.take(1_048_577).read_to_end(&mut bytes)?;
        if bytes.len() > 1_048_576 {
            return Err(error("external response exceeds 1 MiB"));
        }
        Ok(bytes)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn fixture() -> (tempfile::TempDir, ExternalSession) {
        let dir = tempfile::tempdir().unwrap();
        let program = dir.path().join("adapter");
        std::fs::write(
            &program,
            r#"#!/bin/sh
cat > "$0.request"
if test -f "$0.fail"; then echo 'adapter failed' >&2; exit 7; fi
cat "$0.response"
"#,
        )
        .unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::write(program.with_extension("response"), r#"{"content":"hello"}"#).unwrap();
        let session = ExternalSession::new(
            program,
            Some("session background"),
            Some(&crate::default_tools_json()),
        )
        .unwrap();
        (dir, session)
    }

    #[test]
    fn external_schema_response_and_recent_ten_turns() {
        let (dir, mut session) = fixture();
        for n in 0..12 {
            let raw = session
                .send_message(&json!({"role":"user","content":format!("turn-{n}")}).to_string())
                .unwrap();
            assert_eq!(crate::extract_assistant_text(&raw), "hello");
        }
        assert_eq!(session.history.len(), 10);
        let request: Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("adapter.request")).unwrap())
                .unwrap();
        assert_eq!(request["version"], 1);
        assert_eq!(request["system"], "session background");
        assert_eq!(request["messages"][0]["content"], "turn-2");
        assert_eq!(request["tools"][0]["function"]["name"], "get_current_time");
    }

    #[test]
    fn external_preserves_tool_exchange_and_rejects_bad_output_without_committing() {
        let (dir, mut session) = fixture();
        let response = dir.path().join("adapter.response");
        std::fs::write(
            &response,
            r#"{"tool_calls":[{"name":"get_current_time","arguments":{}}]}"#,
        )
        .unwrap();
        let raw = session
            .send_message(r#"{"role":"user","content":"time?"}"#)
            .unwrap();
        assert_eq!(crate::extract_tool_calls(&raw)[0].name, "get_current_time");
        std::fs::write(&response, r#"{"content":"noon"}"#).unwrap();
        session
            .send_message(&crate::tool_response_json(
                "get_current_time",
                &json!({"utc":"12:00"}),
                None,
            ))
            .unwrap();
        assert_eq!(session.history[0].len(), 4);
        let original = session.history.clone();
        for response_text in [
            r#"{"tool_calls":[{"name":"shell","arguments":{}}]}"#,
            r#"{"tool_calls":[{"name":"get_current_time","arguments":"invalid"}]}"#,
            r#"{"tool_calls":[]}"#,
            r#"{"content":"ok","tool_calls":[{"name":"get_current_time","arguments":{}}]}"#,
            "not JSON",
        ] {
            std::fs::write(&response, response_text).unwrap();
            assert!(session
                .send_message(r#"{"role":"user","content":"hi"}"#)
                .is_err());
            assert_eq!(session.history, original);
        }
        std::fs::write(dir.path().join("adapter.fail"), "").unwrap();
        assert!(session
            .send_message(r#"{"role":"user","content":"hi"}"#)
            .unwrap_err()
            .to_string()
            .contains("adapter failed"));
        assert_eq!(session.history, original);
        session.replace_conversation(None, None).unwrap();
        assert!(session.history.is_empty());
    }
}
