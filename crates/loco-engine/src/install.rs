//! Install downloaded artifacts into the loco-bot cache layout.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum InstallError {
    #[error("source file not found: {0}")]
    SourceMissing(PathBuf),
    #[error("io error: {0}")]
    Io(#[from] io::Error),
}

/// Place `source` at `dest`, following any symlinks so relative HF hub links
/// do not break when copied into another directory tree.
///
/// Prefer a hard link to the resolved blob (cheap, no extra disk). Fall back
/// to a full copy when linking is not possible (cross-device, etc.).
pub fn install_model_file(source: &Path, dest: &Path) -> Result<u64, InstallError> {
    if !source.exists() {
        return Err(InstallError::SourceMissing(source.to_path_buf()));
    }

    let resolved = fs::canonicalize(source)?;
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    // Replace broken symlinks / partial files.
    match fs::symlink_metadata(dest) {
        Ok(_) => {
            fs::remove_file(dest)?;
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }

    if let Err(err) = fs::hard_link(&resolved, dest) {
        fs::copy(&resolved, dest).map_err(|copy_err| {
            io::Error::new(
                copy_err.kind(),
                format!(
                    "copy after hard_link failed ({err}): {} -> {}",
                    resolved.display(),
                    dest.display()
                ),
            )
        })?;
    }

    Ok(fs::metadata(dest)?.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[cfg(unix)]
    #[test]
    fn follows_relative_symlink_like_hf_hub() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let blobs = dir.path().join("blobs");
        let snaps = dir.path().join("snapshots").join("rev");
        fs::create_dir_all(&blobs).unwrap();
        fs::create_dir_all(&snaps).unwrap();

        let blob = blobs.join("deadbeef");
        fs::write(&blob, b"real-weights-bytes").unwrap();

        let snap = snaps.join("model.litertlm");
        symlink("../../blobs/deadbeef", &snap).unwrap();

        let dest_root = dir.path().join("loco-cache").join("models").join("e4b");
        let dest = dest_root.join("model.litertlm");
        let bytes = install_model_file(&snap, &dest).unwrap();
        assert_eq!(bytes, 18);
        assert!(dest.is_file());
        assert!(!dest.is_symlink());
        assert_eq!(fs::read(&dest).unwrap(), b"real-weights-bytes");
    }

    #[cfg(unix)]
    #[test]
    fn replaces_broken_symlink_dest() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let src = dir.path().join("src.bin");
        fs::write(&src, b"ok").unwrap();
        let dest = dir.path().join("out.bin");
        symlink("missing-target", &dest).unwrap();
        assert!(!dest.exists()); // broken

        let bytes = install_model_file(&src, &dest).unwrap();
        assert_eq!(bytes, 2);
        assert_eq!(fs::read(&dest).unwrap(), b"ok");
    }
}
