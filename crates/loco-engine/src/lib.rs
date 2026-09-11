//! Model catalog, cache layout, readiness checks, and LiteRT-LM chat.

mod backend;
mod catalog;
mod install;
mod paths;
mod prompt;
mod status;
mod tools;

#[cfg(feature = "inference")]
mod chat;

pub use backend::{BackendParseError, InferenceBackend};
pub use catalog::{ModelId, ModelSpec, BEKKO_A8M, GEMMA4_E4B_IT};
pub use install::{install_model_file, InstallError};
pub use paths::{default_cache_root, model_file_path, CacheLayout};
pub use prompt::{user_message_json, with_session_notes};
pub use status::{file_status, model_fully_ready, model_status, ModelStatus};
pub use tools::{
    default_tools_json, extract_assistant_text, extract_tool_calls, tool_response_json, ToolCall,
    ToolConsentGate, ToolHost, MAX_TOOL_ROUNDS,
};

#[cfg(feature = "inference")]
pub use chat::{ensure_tool_call_id, ChatError, ChatSession, HostToolResume, ToolsTurnProgress};
