# External inference protocol v1

`loco chat --backend external` and `loco serve --backend external` delegate model
inference to the executable named by `LOCO_INFERENCE_COMMAND` (an executable path,
not a shell command). No Gemma model file is required. CPU/GPU behavior is unchanged.
The existing inference build feature still links LiteRT; no LiteRT model is loaded
for external inference.

The executable runs once per generation, including after each tool result. It reads
one JSON document from stdin and writes one JSON document to stdout, then exits.
Diagnostics go to stderr. Exit failure, invalid JSON, and a 120-second timeout are
errors; there is no automatic retry of generation or already-executed tools.
Output is limited to 1 MiB. No server is started by this protocol.

Request:

```json
{"version":1,"system":"session background","messages":[{"role":"user","content":"What time is it?"}],"tools":[{"type":"function","function":{"name":"get_current_time","description":"Current UTC time","parameters":{"type":"object","properties":{}}}}]}
```

Reply, exactly one of:

```json
{"content":"The answer"}
```

```json
{"tool_calls":[{"name":"get_current_time","arguments":{}}]}
```

v1 permits one tool call per generation. Tool names must be declared and arguments
must be JSON objects. Adapters validate model-specific structured output; tools
retain their argument/path/URL validation and consent rules. The runtime assigns
host call IDs and produces `TurnOutcome` / `AgentEvent`; adapters must not invent
execution outcomes or permission decisions.

The runtime keeps up to 10 whole user-turn groups, including assistant messages
and tool exchanges. Messages may contain assistant `tool_calls`, and tool results
use `{"role":"tool","content":[{"name":"...","response":{...}}]}` with an optional
`tool_call_id`. Session/topic notes are compiled by the existing agent and attached
to user messages. Persisted memory is unaffected by the request history cap.
Adapters apply their model's token budget and can omit older whole groups, but
must keep the current user/tool exchange intact or report an oversized-input error.

The macOS Foundation Models adapter belongs to **loco-macos**. `fm` commands,
schemas, token counting, and OS availability are not implemented in this repository.
