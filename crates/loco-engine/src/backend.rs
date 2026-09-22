//! Inference selection: LiteRT-LM CPU/GPU or a platform-owned external adapter.

use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceBackend {
    Cpu,
    Gpu,
    External,
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("unknown inference backend: {0} (expected cpu, gpu, or external)")]
pub struct BackendParseError(String);

impl InferenceBackend {
    pub fn parse(s: &str) -> Result<Self, BackendParseError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "external" => Ok(Self::External),
            "cpu" => Ok(Self::Cpu),
            "gpu" | "metal" => Ok(Self::Gpu),
            other => Err(BackendParseError(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::External => "external",
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
        assert_eq!(
            InferenceBackend::parse(" EXTERNAL ").unwrap(),
            InferenceBackend::External
        );
        assert_eq!(InferenceBackend::External.as_str(), "external");
        assert!(InferenceBackend::parse("npu").is_err());
    }
}
