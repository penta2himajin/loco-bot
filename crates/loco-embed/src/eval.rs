//! Declarative S1 / deixis evaluation harness (logic layer).
//!
//! Fixtures live under `fixtures/s1/`. Cases use synthetic embeddings so CI
//! never needs ONNX or LiteRT-LM. Optional text→embed suites can be added later
//! behind the `ort` feature.

use std::fmt;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

use crate::deixis::{classify_deixis, DeixisKind};
use crate::expand::expand_query;
use crate::l2_normalize;
use crate::resolve::{
    match_clarification, resolve_topic, Clarification, ClarifyAction, ClarifyCandidate,
    ResolveInput, ResolveOutcome, DEFAULT_AMBIGUITY_DELTA,
};
use crate::s1::{ChunkScore, S1Thresholds};

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("case `{id}`: {message}")]
    Case { id: String, message: String },
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvalSuite {
    pub version: u32,
    #[serde(default)]
    pub name: String,
    pub cases: Vec<EvalCase>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvalCase {
    Deixis {
        id: String,
        user: String,
        expect_deixis: DeixisKindExpect,
    },
    Expand {
        id: String,
        user: String,
        #[serde(default)]
        previous_user: Option<String>,
        #[serde(default)]
        expect_expanded: Option<String>,
        #[serde(default)]
        expect_expanded_contains: Option<String>,
        #[serde(default)]
        expect_unchanged: bool,
    },
    Resolve {
        id: String,
        user: String,
        query_emb: EmbSpec,
        #[serde(default)]
        current: Option<EmbSpec>,
        #[serde(default)]
        past: Vec<PastChunkSpec>,
        #[serde(default)]
        previous_chunk: Option<usize>,
        #[serde(default)]
        chunk_labels: Vec<String>,
        #[serde(default)]
        thresholds: Option<ThresholdSpec>,
        #[serde(default)]
        ambiguity_delta: Option<f32>,
        expect: OutcomeExpect,
    },
    ClarifyMatch {
        id: String,
        user: String,
        candidates: Vec<ClarifyCandidateSpec>,
        expect_action: ActionExpect,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum DeixisKindExpect {
    Plain,
    ContinueHint,
    ReturnNamed,
    ReturnUnspecified,
}

impl From<DeixisKindExpect> for DeixisKind {
    fn from(value: DeixisKindExpect) -> Self {
        match value {
            DeixisKindExpect::Plain => DeixisKind::Plain,
            DeixisKindExpect::ContinueHint => DeixisKind::ContinueHint,
            DeixisKindExpect::ReturnNamed => DeixisKind::ReturnNamed,
            DeixisKindExpect::ReturnUnspecified => DeixisKind::ReturnUnspecified,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbSpec {
    /// One-hot unit vector: `[dim, hot_index]`.
    Unit { unit: [usize; 2] },
    /// Explicit vector; optionally L2-normalize.
    Vec {
        vec: Vec<f32>,
        #[serde(default)]
        normalize: bool,
    },
}

impl EmbSpec {
    fn materialize(&self) -> Result<Vec<f32>, String> {
        match self {
            EmbSpec::Unit { unit: [dim, hot] } => {
                if *hot >= *dim {
                    return Err(format!("hot index {hot} >= dim {dim}"));
                }
                let mut v = vec![0.0f32; *dim];
                v[*hot] = 1.0;
                Ok(v)
            }
            EmbSpec::Vec { vec, normalize } => {
                let mut v = vec.clone();
                if *normalize {
                    l2_normalize(&mut v);
                }
                Ok(v)
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PastChunkSpec {
    pub index: usize,
    #[serde(flatten)]
    pub emb: EmbSpec,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThresholdSpec {
    pub continue_min: f32,
    pub return_min: f32,
    pub new_max: f32,
}

impl From<&ThresholdSpec> for S1Thresholds {
    fn from(value: &ThresholdSpec) -> Self {
        S1Thresholds {
            continue_min: value.continue_min,
            return_min: value.return_min,
            new_max: value.new_max,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "outcome", rename_all = "PascalCase")]
pub enum OutcomeExpect {
    Continue,
    New,
    Return {
        chunk_index: usize,
    },
    NeedsClarification {
        #[serde(default)]
        min_candidates: Option<usize>,
        #[serde(default)]
        question_contains: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClarifyCandidateSpec {
    pub label: String,
    pub action: ActionExpect,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "PascalCase")]
pub enum ActionExpect {
    ContinueCurrent,
    ReturnTo { chunk_index: usize },
}

impl From<&ActionExpect> for ClarifyAction {
    fn from(value: &ActionExpect) -> Self {
        match value {
            ActionExpect::ContinueCurrent => ClarifyAction::ContinueCurrent,
            ActionExpect::ReturnTo { chunk_index } => ClarifyAction::ReturnTo {
                chunk_index: *chunk_index,
            },
        }
    }
}

impl ActionExpect {
    fn from_action(action: ClarifyAction) -> Self {
        match action {
            ClarifyAction::ContinueCurrent => ActionExpect::ContinueCurrent,
            ClarifyAction::ReturnTo { chunk_index } => ActionExpect::ReturnTo { chunk_index },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseResult {
    pub id: String,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuiteReport {
    pub name: String,
    pub passed: usize,
    pub failed: usize,
    pub results: Vec<CaseResult>,
}

impl SuiteReport {
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }
}

impl fmt::Display for SuiteReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "suite `{}`: {}/{} passed",
            if self.name.is_empty() {
                "(unnamed)"
            } else {
                &self.name
            },
            self.passed,
            self.passed + self.failed
        )?;
        for r in &self.results {
            if r.ok {
                writeln!(f, "  PASS {}", r.id)?;
            } else {
                writeln!(f, "  FAIL {}: {}", r.id, r.detail)?;
            }
        }
        Ok(())
    }
}

/// Load a suite JSON file from disk.
pub fn load_suite(path: impl AsRef<Path>) -> Result<EvalSuite, EvalError> {
    let raw = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw)?)
}

/// Load every `*.json` suite under a directory (non-recursive).
pub fn load_suites_dir(dir: impl AsRef<Path>) -> Result<Vec<(String, EvalSuite)>, EvalError> {
    let mut out = Vec::new();
    let mut entries: Vec<_> = fs::read_dir(dir.as_ref())?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| x.eq_ignore_ascii_case("json"))
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("suite")
            .to_string();
        let mut suite = load_suite(&path)?;
        if suite.name.is_empty() {
            suite.name = name.clone();
        }
        out.push((name, suite));
    }
    Ok(out)
}

/// Run all cases in a suite.
pub fn run_suite(suite: &EvalSuite) -> SuiteReport {
    let mut results = Vec::with_capacity(suite.cases.len());
    let mut passed = 0usize;
    let mut failed = 0usize;
    for case in &suite.cases {
        let result = run_case(case);
        if result.ok {
            passed += 1;
        } else {
            failed += 1;
        }
        results.push(result);
    }
    SuiteReport {
        name: suite.name.clone(),
        passed,
        failed,
        results,
    }
}

fn case_id(case: &EvalCase) -> &str {
    match case {
        EvalCase::Deixis { id, .. }
        | EvalCase::Expand { id, .. }
        | EvalCase::Resolve { id, .. }
        | EvalCase::ClarifyMatch { id, .. } => id,
    }
}

fn run_case(case: &EvalCase) -> CaseResult {
    let id = case_id(case).to_string();
    match eval_case(case) {
        Ok(()) => CaseResult {
            id,
            ok: true,
            detail: String::new(),
        },
        Err(message) => CaseResult {
            id,
            ok: false,
            detail: message,
        },
    }
}

fn eval_case(case: &EvalCase) -> Result<(), String> {
    match case {
        EvalCase::Deixis {
            user,
            expect_deixis,
            ..
        } => {
            let got = classify_deixis(user);
            let expect: DeixisKind = (*expect_deixis).into();
            if got != expect {
                return Err(format!("deixis got {got:?}, expect {expect:?}"));
            }
            Ok(())
        }
        EvalCase::Expand {
            user,
            previous_user,
            expect_expanded,
            expect_expanded_contains,
            expect_unchanged,
            ..
        } => {
            let got = expand_query(user, previous_user.as_deref());
            if *expect_unchanged && got != user.trim() {
                return Err(format!("expected unchanged `{user}`, got `{got}`"));
            }
            if let Some(exact) = expect_expanded {
                if &got != exact {
                    return Err(format!("expanded got `{got}`, expect `{exact}`"));
                }
            }
            if let Some(needle) = expect_expanded_contains {
                if !got.contains(needle) {
                    return Err(format!("expanded `{got}` missing `{needle}`"));
                }
            }
            Ok(())
        }
        EvalCase::Resolve {
            user,
            query_emb,
            current,
            past,
            previous_chunk,
            chunk_labels,
            thresholds,
            ambiguity_delta,
            expect,
            ..
        } => {
            let q = query_emb.materialize()?;
            let cur_owned = current.as_ref().map(|c| c.materialize()).transpose()?;
            let past_owned: Vec<(usize, Vec<f32>)> = past
                .iter()
                .map(|p| Ok((p.index, p.emb.materialize()?)))
                .collect::<Result<_, String>>()?;
            let past_scores: Vec<ChunkScore<'_>> = past_owned
                .iter()
                .map(|(index, emb)| ChunkScore {
                    index: *index,
                    embedding: emb.as_slice(),
                })
                .collect();
            let th = thresholds
                .as_ref()
                .map(S1Thresholds::from)
                .unwrap_or_default();
            let out = resolve_topic(&ResolveInput {
                user,
                query_emb: &q,
                current: cur_owned.as_deref(),
                past: &past_scores,
                previous_chunk: *previous_chunk,
                chunk_labels,
                thresholds: th,
                ambiguity_delta: ambiguity_delta.unwrap_or(DEFAULT_AMBIGUITY_DELTA),
            });
            match_outcome(&out, expect)
        }
        EvalCase::ClarifyMatch {
            user,
            candidates,
            expect_action,
            ..
        } => {
            let clarification = Clarification {
                question: "q".into(),
                candidates: candidates
                    .iter()
                    .map(|c| ClarifyCandidate {
                        label: c.label.clone(),
                        action: ClarifyAction::from(&c.action),
                    })
                    .collect(),
            };
            let got = match_clarification(user, &clarification);
            match got {
                Some(action) => {
                    let got_expect = ActionExpect::from_action(action);
                    if &got_expect != expect_action {
                        return Err(format!(
                            "action got {got_expect:?}, expect {expect_action:?}"
                        ));
                    }
                    Ok(())
                }
                None => Err(format!("no match for `{user}`, expect {expect_action:?}")),
            }
        }
    }
}

fn match_outcome(got: &ResolveOutcome, expect: &OutcomeExpect) -> Result<(), String> {
    match (got, expect) {
        (ResolveOutcome::Continue, OutcomeExpect::Continue)
        | (ResolveOutcome::New, OutcomeExpect::New) => Ok(()),
        (ResolveOutcome::Return { chunk_index: g }, OutcomeExpect::Return { chunk_index: e })
            if g == e =>
        {
            Ok(())
        }
        (
            ResolveOutcome::NeedsClarification(c),
            OutcomeExpect::NeedsClarification {
                min_candidates,
                question_contains,
            },
        ) => {
            if let Some(min) = min_candidates {
                if c.candidates.len() < *min {
                    return Err(format!(
                        "clarify candidates {} < min {min}",
                        c.candidates.len()
                    ));
                }
            }
            if let Some(needle) = question_contains {
                if !c.question.contains(needle) {
                    return Err(format!("question `{}` missing `{needle}`", c.question));
                }
            }
            Ok(())
        }
        _ => Err(format!("outcome got {got:?}, expect {expect:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_runs_inline_deixis() {
        let suite: EvalSuite = serde_json::from_str(
            r#"{
              "version": 1,
              "name": "smoke",
              "cases": [
                {
                  "kind": "deixis",
                  "id": "u1",
                  "user": "さっきの話",
                  "expect_deixis": "ReturnUnspecified"
                }
              ]
            }"#,
        )
        .unwrap();
        let report = run_suite(&suite);
        assert!(report.all_passed(), "{report}");
    }

    #[test]
    fn unit_emb_materialize() {
        let v = EmbSpec::Unit { unit: [4, 2] }.materialize().unwrap();
        assert_eq!(v, vec![0.0, 0.0, 1.0, 0.0]);
    }
}
