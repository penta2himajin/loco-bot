//! Surface-agnostic agent turn API for loco-bot.
//!
//! CLI, laptop UI, Android, and glasses shells should call [`AgentSession::turn`]
//! instead of re-implementing resolve / compile / chat / tools.

mod consent;
mod events;
mod serve_protocol;
mod session;

pub use consent::{AllowListedTools, AllowLowRiskOnly, AllowUpTo, ToolConsent, ToolRisk};
pub use events::{AgentEvent, ClarifyChoice, TurnOutcome};
pub use serve_protocol::{
    merge_tools_json, parse_serve_client_line, HostToolSpec, ServeClientMessage,
    ServeServerMessage, ToolExecKind,
};
pub use session::{AgentError, AgentSession, AgentSessionConfig, AgentTurnProgress};
