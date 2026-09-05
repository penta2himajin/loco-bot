//! Model catalog, cache layout, readiness checks, and LiteRT-LM chat.

mod backend;
mod catalog;
mod install;
mod paths;
mod prompt;
mod status;

#[cfg(feature = "inference")]
mod chat;

pub use backend::{BackendParseError, InferenceBackend};
pub use catalog::{ModelId, ModelSpec, GEMMA4_E4B_IT};
pub use install::{install_model_file, InstallError};
pub use paths::{default_cache_root, model_file_path, CacheLayout};
pub use prompt::user_message_json;
pub use status::{model_status, ModelStatus};

#[cfg(feature = "inference")]
pub use chat::{ChatError, ChatSession};
