//! Real granite-97m ONNX fixtures under `fixtures/s1_onnx/`.
//!
//! Skips cleanly when weights are missing (CI without download).

#![cfg(feature = "ort")]

use std::path::PathBuf;

use loco_embed::{
    default_granite_dir, granite_ready, load_suites_dir, run_suite_with_embedder, GraniteEmbedder,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/s1_onnx")
}

#[test]
fn granite_text_suites_pass_if_cached() {
    let Some(dir) = default_granite_dir() else {
        eprintln!("skip: no granite dir");
        return;
    };
    if !granite_ready(&dir) {
        eprintln!("skip: granite not ready at {}", dir.display());
        return;
    }

    let mut emb = GraniteEmbedder::open(&dir).expect("open granite");
    let suites = load_suites_dir(fixtures_dir()).expect("load fixtures/s1_onnx");
    assert!(!suites.is_empty());

    let mut failed = 0usize;
    for (_name, suite) in &suites {
        assert_eq!(suite.version, 1);
        let report = run_suite_with_embedder(suite, &mut emb);
        eprint!("{report}");
        if !report.all_passed() {
            failed += report.failed;
        }
    }
    assert_eq!(failed, 0, "{failed} ONNX eval case(s) failed");
}
