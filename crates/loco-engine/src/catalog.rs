//! Known on-device model identities.

use serde::{Deserialize, Serialize};

/// Stable id used by the CLI (`loco download gemma4-e4b`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelId {
    Gemma4E4b,
    /// ibm-granite/granite-embedding-97m-multilingual-r2 (S1).
    Granite97m,
}

impl ModelId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gemma4E4b => "gemma4-e4b",
            Self::Granite97m => "granite-97m",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "gemma4-e4b" | "gemma-4-e4b" | "e4b" => Some(Self::Gemma4E4b),
            "granite-97m" | "granite97m" | "granite-embedding-97m" => Some(Self::Granite97m),
            _ => None,
        }
    }

    pub fn all() -> &'static [ModelId] {
        &[Self::Gemma4E4b, Self::Granite97m]
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Hugging Face artifact coordinates for one or more files under a model id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub id: ModelId,
    pub hf_repo: &'static str,
    /// Relative paths inside the HF repo (and under the loco cache model dir).
    pub files: &'static [&'static str],
    pub display_name: &'static str,
}

impl ModelSpec {
    pub fn for_id(id: ModelId) -> &'static ModelSpec {
        match id {
            ModelId::Gemma4E4b => &GEMMA4_E4B_IT,
            ModelId::Granite97m => &GRANITE_97M,
        }
    }

    /// Primary artifact (first file) — used by single-file helpers.
    pub fn primary_file(&self) -> &'static str {
        self.files[0]
    }

    /// Backward-compatible alias for [`Self::primary_file`].
    pub fn filename(&self) -> &'static str {
        self.primary_file()
    }
}

/// Official LiteRT-LM pack of Gemma 4 E4B instruct.
pub const GEMMA4_E4B_IT: ModelSpec = ModelSpec {
    id: ModelId::Gemma4E4b,
    hf_repo: "litert-community/gemma-4-E4B-it-litert-lm",
    files: &["gemma-4-E4B-it.litertlm"],
    display_name: "Gemma 4 E4B (LiteRT-LM)",
};

/// Granite embedding 97M multilingual R2 (ONNX + tokenizer) for S1.
pub const GRANITE_97M: ModelSpec = ModelSpec {
    id: ModelId::Granite97m,
    hf_repo: "ibm-granite/granite-embedding-97m-multilingual-r2",
    files: &["onnx/model.onnx", "tokenizer.json"],
    display_name: "Granite Embedding 97M Multilingual R2",
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aliases() {
        assert_eq!(ModelId::parse("gemma4-e4b"), Some(ModelId::Gemma4E4b));
        assert_eq!(ModelId::parse("e4b"), Some(ModelId::Gemma4E4b));
        assert_eq!(ModelId::parse("granite-97m"), Some(ModelId::Granite97m));
        assert_eq!(ModelId::parse("nope"), None);
    }

    #[test]
    fn gemma_spec_points_at_litert_community() {
        let spec = ModelSpec::for_id(ModelId::Gemma4E4b);
        assert_eq!(spec.hf_repo, "litert-community/gemma-4-E4B-it-litert-lm");
        assert!(spec.primary_file().ends_with(".litertlm"));
    }

    #[test]
    fn granite_lists_onnx_and_tokenizer() {
        let spec = ModelSpec::for_id(ModelId::Granite97m);
        assert_eq!(spec.files.len(), 2);
        assert!(spec.files[0].ends_with("model.onnx"));
        assert_eq!(spec.files[1], "tokenizer.json");
    }
}
