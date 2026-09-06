//! Assert every declared S1 pattern has ≥3 mapped cases present in fixtures.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use loco_embed::load_suites_dir;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct CoverageMap {
    min_cases_per_pattern: usize,
    patterns: std::collections::BTreeMap<String, Vec<String>>,
}

fn fixtures_s1() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/s1")
}

fn fixtures_onnx() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/s1_onnx")
}

fn collect_case_ids() -> HashSet<String> {
    let mut ids = HashSet::new();
    for dir in [fixtures_s1(), fixtures_onnx()] {
        let suites = load_suites_dir(&dir).unwrap_or_default();
        for (_name, suite) in suites {
            for case in &suite.cases {
                ids.insert(case_id(case).to_string());
            }
        }
    }
    ids
}

fn case_id(case: &loco_embed::EvalCase) -> &str {
    use loco_embed::EvalCase::*;
    match case {
        Deixis { id, .. }
        | Expand { id, .. }
        | Resolve { id, .. }
        | ClarifyMatch { id, .. }
        | ResolveText { id, .. }
        | EmbedRank { id, .. } => id,
    }
}

#[test]
fn coverage_map_has_three_cases_per_pattern() {
    let raw = fs::read_to_string(fixtures_s1().join("coverage_map.json")).expect("coverage_map");
    let map: CoverageMap = serde_json::from_str(&raw).expect("parse coverage_map");
    assert_eq!(map.min_cases_per_pattern, 3);
    assert!(
        map.patterns.len() >= 35,
        "expected a full pattern taxonomy, got {}",
        map.patterns.len()
    );

    let present = collect_case_ids();
    let mut short = Vec::new();
    let mut missing = Vec::new();
    for (pattern, ids) in &map.patterns {
        if ids.len() < map.min_cases_per_pattern {
            short.push(format!("{pattern}: {}", ids.len()));
        }
        for id in ids {
            if !present.contains(id) {
                missing.push(format!("{pattern} → {id}"));
            }
        }
    }
    assert!(
        short.is_empty(),
        "patterns below {} cases: {short:?}",
        map.min_cases_per_pattern
    );
    assert!(
        missing.is_empty(),
        "coverage_map references missing case ids: {missing:?}"
    );
}
