//! Ensure macOS `@rpath/liblitert-lm.so` resolves to the shipped `.dylib`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    if env::var("CARGO_FEATURE_INFERENCE").is_err() {
        return;
    }
    if env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("macos") {
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let Some(profile_dir) = out_dir.ancestors().nth(3).map(Path::to_path_buf) else {
        return;
    };

    let dylib_name = "liblitert-lm.dylib";
    let so_name = "liblitert-lm.so";

    for dir in [
        profile_dir.clone(),
        profile_dir.join("deps"),
        profile_dir.join("examples"),
    ] {
        let _ = fs::create_dir_all(&dir);
        let dylib = dir.join(dylib_name);
        let so = dir.join(so_name);
        if !dylib.exists() {
            continue;
        }
        let _ = fs::remove_file(&so);
        #[cfg(unix)]
        {
            let _ = std::os::unix::fs::symlink(dylib_name, &so);
        }
    }
}
