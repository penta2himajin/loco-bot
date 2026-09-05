//! Inference backend selection for LiteRT-LM.

use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceBackend {
    Cpu,
    Gpu,
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("unknown inference backend: {0} (expected cpu or gpu)")]
pub struct BackendParseError(String);

impl InferenceBackend {
    pub fn parse(s: &str) -> Result<Self, BackendParseError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" | "metal" => Ok(Self::Gpu),
            other => Err(BackendParseError(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

impl fmt::Display for InferenceBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aliases() {
        assert_eq!(
            InferenceBackend::parse("cpu").unwrap(),
            InferenceBackend::Cpu
        );
        assert_eq!(
            InferenceBackend::parse("GPU").unwrap(),
            InferenceBackend::Gpu
        );
        assert_eq!(
            InferenceBackend::parse("metal").unwrap(),
            InferenceBackend::Gpu
        );
        assert!(InferenceBackend::parse("npu").is_err());
    }
}
