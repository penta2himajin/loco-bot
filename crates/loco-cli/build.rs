//! Ensure the `loco` binary can load LiteRT-LM at runtime on macOS/Linux.
//!
//! `litertlm-rs` copies `liblitert-lm.*` into the profile dir, but its
//! `cargo:rustc-link-arg` rpath does **not** apply to dependent binaries.
//! macOS prebuilts also advertise install_name `@rpath/liblitert-lm.so`
//! while shipping `*.dylib`, so we add a `.so` symlink beside the binary.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    if env::var("CARGO_FEATURE_INFERENCE").is_err() {
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let Some(profile_dir) = out_dir.ancestors().nth(3).map(Path::to_path_buf) else {
        return;
    };

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let dylib_name = if target_os == "windows" {
        "litert-lm.dll"
    } else if target_os == "macos" {
        "liblitert-lm.dylib"
    } else {
        "liblitert-lm.so"
    };

    // Prefer metadata from litertlm-sys when visible; otherwise use a copy
    // already placed by litertlm-rs into the profile directory.
    let lib_src = env::var_os("DEP_LITERT_LM_LIB_DIR")
        .map(PathBuf::from)
        .and_then(|dir| {
            let name = env::var("DEP_LITERT_LM_LIB_FILENAME").unwrap_or_else(|_| dylib_name.into());
            let p = dir.join(name);
            p.exists().then_some(p)
        })
        .or_else(|| {
            let p = profile_dir.join(dylib_name);
            p.exists().then_some(p)
        });

    for dest_dir in [
        profile_dir.clone(),
        profile_dir.join("deps"),
        profile_dir.join("examples"),
    ] {
        let _ = fs::create_dir_all(&dest_dir);
        if let Some(ref src) = lib_src {
            let dest = dest_dir.join(dylib_name);
            if src != &dest {
                let _ = fs::copy(src, &dest);
            }
        }
        if target_os == "macos" {
            let so = dest_dir.join("liblitert-lm.so");
            let dylib = dest_dir.join(dylib_name);
            if dylib.exists() {
                let _ = fs::remove_file(&so);
                #[cfg(unix)]
                {
                    let _ = std::os::unix::fs::symlink(dylib_name, &so);
                }
            }
        }
    }

    match target_os.as_str() {
        "macos" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        }
        "windows" => {}
        _ => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        }
    }
}
