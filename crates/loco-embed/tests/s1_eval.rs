//! Run declarative S1 / deixis fixtures under `fixtures/s1/`.

use std::path::PathBuf;

use loco_embed::{load_suites_dir, run_suite};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/s1")
}

#[test]
fn logic_suites_pass() {
    let suites = load_suites_dir(fixtures_dir()).expect("load fixtures/s1");
    assert!(
        !suites.is_empty(),
        "expected at least one suite under fixtures/s1"
    );

    let mut failed = 0usize;
    let mut total = 0usize;
    for (_name, suite) in &suites {
        assert_eq!(suite.version, 1, "unsupported suite version");
        let report = run_suite(suite);
        eprint!("{report}");
        total += report.passed + report.failed;
        if !report.all_passed() {
            failed += report.failed;
        }
    }
    assert!(
        total >= 90,
        "expected expanded fixture coverage, got {total}"
    );
    assert_eq!(failed, 0, "{failed} eval case(s) failed");
}
