//! Known on-device model identities.

use serde::{Deserialize, Serialize};

/// Stable id used by the CLI (`loco download gemma4-e4b`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelId {
    Gemma4E4b,
}

impl ModelId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gemma4E4b => "gemma4-e4b",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "gemma4-e4b" | "gemma-4-e4b" | "e4b" => Some(Self::Gemma4E4b),
            _ => None,
        }
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Hugging Face artifact coordinates for a `.litertlm` (or related) file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub id: ModelId,
    pub hf_repo: &'static str,
    pub filename: &'static str,
    pub display_name: &'static str,
}

impl ModelSpec {
    pub fn for_id(id: ModelId) -> &'static ModelSpec {
        match id {
            ModelId::Gemma4E4b => &GEMMA4_E4B_IT,
        }
    }
}

/// Official LiteRT-LM pack of Gemma 4 E4B instruct.
pub const GEMMA4_E4B_IT: ModelSpec = ModelSpec {
    id: ModelId::Gemma4E4b,
    hf_repo: "litert-community/gemma-4-E4B-it-litert-lm",
    filename: "gemma-4-E4B-it.litertlm",
    display_name: "Gemma 4 E4B (LiteRT-LM)",
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aliases() {
        assert_eq!(ModelId::parse("gemma4-e4b"), Some(ModelId::Gemma4E4b));
        assert_eq!(ModelId::parse("e4b"), Some(ModelId::Gemma4E4b));
        assert_eq!(ModelId::parse("nope"), None);
    }

    #[test]
    fn gemma_spec_points_at_litert_community() {
        let spec = ModelSpec::for_id(ModelId::Gemma4E4b);
        assert_eq!(spec.hf_repo, "litert-community/gemma-4-E4B-it-litert-lm");
        assert!(spec.filename.ends_with(".litertlm"));
    }
}
