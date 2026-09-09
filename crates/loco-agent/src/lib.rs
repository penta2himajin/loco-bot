//! Surface-agnostic agent turn API for loco-bot.
//!
//! CLI, laptop UI, Android, and glasses shells should call [`AgentSession::turn`]
//! instead of re-implementing resolve / compile / chat / tools.

mod consent;
mod events;
mod session;

pub use consent::{AllowListedTools, AllowLowRiskOnly, AllowUpTo, ToolConsent, ToolRisk};
pub use events::{AgentEvent, ClarifyChoice, TurnOutcome};
pub use session::{AgentError, AgentSession, AgentSessionConfig};
