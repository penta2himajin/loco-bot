//! On-disk cache layout under the user cache directory.

use std::path::{Path, PathBuf};

use crate::catalog::{ModelId, ModelSpec};

/// Resolves where loco-bot stores downloaded models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheLayout {
    root: PathBuf,
}

impl CacheLayout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    pub fn model_dir(&self, id: ModelId) -> PathBuf {
        self.models_dir().join(id.as_str())
    }

    pub fn model_path(&self, spec: &ModelSpec) -> PathBuf {
        self.model_dir(spec.id).join(spec.filename)
    }
}

/// Default cache root: `$XDG_CACHE_HOME/loco-bot` or platform equivalent.
pub fn default_cache_root() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("loco-bot")
}

/// Convenience: default layout path for a model file.
pub fn model_file_path(id: ModelId) -> PathBuf {
    let layout = CacheLayout::new(default_cache_root());
    layout.model_path(ModelSpec::for_id(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::GEMMA4_E4B_IT;

    #[test]
    fn layout_nests_by_model_id() {
        let layout = CacheLayout::new("/tmp/loco-cache");
        assert_eq!(
            layout.model_path(&GEMMA4_E4B_IT),
            PathBuf::from("/tmp/loco-cache/models/gemma4-e4b/gemma-4-E4B-it.litertlm")
        );
    }
}
