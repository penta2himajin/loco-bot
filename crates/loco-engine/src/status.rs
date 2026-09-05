//! Readiness of a cached model file (presence + non-empty).

use std::path::PathBuf;

use crate::catalog::{ModelId, ModelSpec};
use crate::paths::CacheLayout;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    Missing { expected: PathBuf },
    Present { path: PathBuf, bytes: u64 },
}

impl ModelStatus {
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Present { bytes, .. } if *bytes > 0)
    }
}

pub fn model_status(layout: &CacheLayout, id: ModelId) -> ModelStatus {
    let spec = ModelSpec::for_id(id);
    let path = layout.model_path(spec);
    match std::fs::metadata(&path) {
        Ok(meta) if meta.is_file() => ModelStatus::Present {
            path,
            bytes: meta.len(),
        },
        _ => ModelStatus::Missing { expected: path },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::CacheLayout;
    use tempfile::tempdir;

    #[test]
    fn missing_when_file_absent() {
        let dir = tempdir().unwrap();
        let layout = CacheLayout::new(dir.path());
        let status = model_status(&layout, ModelId::Gemma4E4b);
        assert!(!status.is_ready());
        match status {
            ModelStatus::Missing { expected } => {
                assert!(expected.ends_with("gemma-4-E4B-it.litertlm"));
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn present_when_nonempty_file_exists() {
        let dir = tempdir().unwrap();
        let layout = CacheLayout::new(dir.path());
        let path = layout.model_path(ModelSpec::for_id(ModelId::Gemma4E4b));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"fake-weights").unwrap();

        let status = model_status(&layout, ModelId::Gemma4E4b);
        assert!(status.is_ready());
        match status {
            ModelStatus::Present { bytes, .. } => assert_eq!(bytes, 12),
            other => panic!("expected Present, got {other:?}"),
        }
    }
}
