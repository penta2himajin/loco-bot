#![cfg(all(unix, feature = "inference"))]

use serde_json::{json, Value};
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Stdio};

#[test]
fn external_serve_without_gemma_preserves_host_tool_handoff() {
    let dir = tempfile::tempdir().unwrap();
    let program = dir.path().join("adapter");
    std::fs::write(
        &program,
        r#"#!/bin/sh
input=$(cat)
case "$input" in
*'"role":"tool"'*) echo '{"content":"Tool result received."}' ;;
*) echo '{"tool_calls":[{"name":"open_url","arguments":{"url":"https://example.com"}}]}' ;;
esac
"#,
    )
    .unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o700)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_loco"))
        .args([
            "serve",
            "--backend",
            "external",
            "--no-memory",
            "--no-topic",
            "--cache-dir",
        ])
        .arg(dir.path().join("empty-cache"))
        .env("LOCO_INFERENCE_COMMAND", program)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let requests = [
        json!({"op":"configure","tools":[{
            "name":"open_url","description":"Open a web URL","risk":"medium","exec":"host",
            "parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"],"additionalProperties":false}
        }]}),
        json!({"op":"turn","user":"Open example.com"}),
        json!({"op":"tool_result","call_id":"host-1","ok":false,"content":{"error":"denied"}}),
    ];
    let mut stdin = child.stdin.take().unwrap();
    for request in requests {
        writeln!(stdin, "{request}").unwrap();
    }
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let replies: Vec<Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(replies.len(), 3);
    assert_eq!(replies[0]["status"], "done");
    assert_eq!(replies[1]["status"], "awaiting_tool");
    assert_eq!(replies[1]["call_id"], "host-1");
    let request = replies[1]["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["type"] == "tool_request")
        .unwrap();
    assert_eq!(request["risk"], "medium");
    assert_eq!(request["arguments"]["url"], "https://example.com");
    assert_eq!(replies[2]["status"], "done");
    assert_eq!(replies[2]["reply_text"], "Tool result received.");
    assert!(replies[2]["events"]
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e["type"] == "tool_result" && e["ok"] == false));
    assert!(
        !dir.path().join("empty-cache").exists(),
        "no model download or memory write"
    );
}
