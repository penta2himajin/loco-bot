//! Topic resolve: deixis rules + S1 scores + previous-chunk stack + clarify + S2.

use crate::cosine::cosine;
use crate::deixis::{classify_deixis, DeixisKind};
use crate::s1::{decide_s1, ChunkScore, S1Outcome, S1Thresholds};
use crate::s2::{S2Decision, TopicS2};

/// Default: if top-2 past scores are within this gap and both are strong, ask.
pub const DEFAULT_AMBIGUITY_DELTA: f32 = 0.05;

#[derive(Debug, Clone, PartialEq)]
pub struct Clarification {
    pub question: String,
    pub candidates: Vec<ClarifyCandidate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClarifyCandidate {
    pub label: String,
    pub action: ClarifyAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClarifyAction {
    ContinueCurrent,
    ReturnTo { chunk_index: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolveOutcome {
    Continue,
    New,
    Return { chunk_index: usize },
    NeedsClarification(Clarification),
}

/// Inputs for one resolve step (embeddings already computed by the caller).
pub struct ResolveInput<'a> {
    pub user: &'a str,
    pub query_emb: &'a [f32],
    pub current: Option<&'a [f32]>,
    pub past: &'a [ChunkScore<'a>],
    /// Chunk that was current before the active one (topic stack N-1).
    pub previous_chunk: Option<usize>,
    /// Summaries indexed by chunk index (including current) for clarify labels.
    pub chunk_labels: &'a [String],
    pub thresholds: S1Thresholds,
    pub ambiguity_delta: f32,
    /// Optional S2 for the S1 gray band. When `None`, gray → New.
    pub s2: Option<&'a mut dyn TopicS2>,
}

/// Resolve topic transition with deixis-aware rules.
pub fn resolve_topic(input: &mut ResolveInput<'_>) -> ResolveOutcome {
    match classify_deixis(input.user) {
        DeixisKind::ReturnUnspecified => resolve_unspecified_return(input),
        DeixisKind::ReturnNamed | DeixisKind::Plain | DeixisKind::ContinueHint => {
            resolve_with_s1(input)
        }
    }
}

fn resolve_unspecified_return(input: &ResolveInput<'_>) -> ResolveOutcome {
    if let Some(i) = input.previous_chunk {
        return ResolveOutcome::Return { chunk_index: i };
    }
    match input.past.len() {
        0 => {
            if input.current.is_some() {
                ResolveOutcome::Continue
            } else {
                ResolveOutcome::New
            }
        }
        1 => ResolveOutcome::Return {
            chunk_index: input.past[0].index,
        },
        _ => ResolveOutcome::NeedsClarification(build_clarification(
            input.past,
            input.chunk_labels,
            true,
        )),
    }
}

fn resolve_with_s1(input: &mut ResolveInput<'_>) -> ResolveOutcome {
    if let Some(clarify) = ambiguous_past_return(input) {
        return ResolveOutcome::NeedsClarification(clarify);
    }

    match decide_s1(
        input.query_emb,
        input.current,
        input.past,
        &input.thresholds,
    ) {
        S1Outcome::Continue => ResolveOutcome::Continue,
        S1Outcome::New => ResolveOutcome::New,
        S1Outcome::Return { chunk_index } => ResolveOutcome::Return { chunk_index },
        S1Outcome::Gray(evidence) => {
            let decision = match input.s2.as_mut() {
                Some(s2) => s2.decide_gray(&evidence),
                None => S2Decision::New,
            };
            match decision {
                S2Decision::Continue => ResolveOutcome::Continue,
                S2Decision::New => ResolveOutcome::New,
                S2Decision::Return { chunk_index } => ResolveOutcome::Return { chunk_index },
                S2Decision::ClarifyPast => ResolveOutcome::NeedsClarification(build_clarification(
                    input.past,
                    input.chunk_labels,
                    false,
                )),
            }
        }
    }
}

fn ambiguous_past_return(input: &ResolveInput<'_>) -> Option<Clarification> {
    if input.past.len() < 2 {
        return None;
    }
    let mut scored: Vec<(usize, f32)> = input
        .past
        .iter()
        .map(|c| (c.index, cosine(input.query_emb, c.embedding)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let s0 = scored[0].1;
    let s1 = scored[1].1;
    if s0 < input.thresholds.return_min {
        return None;
    }
    if s0 - s1 >= input.ambiguity_delta {
        return None;
    }
    if let Some(cur) = input.current {
        let cs = cosine(input.query_emb, cur);
        if cs >= input.thresholds.continue_min {
            return None;
        }
    }
    Some(build_clarification(input.past, input.chunk_labels, false))
}

fn build_clarification(
    past: &[ChunkScore<'_>],
    labels: &[String],
    include_current_option: bool,
) -> Clarification {
    let mut candidates = Vec::new();
    if include_current_option {
        candidates.push(ClarifyCandidate {
            label: "今の話題のまま".into(),
            action: ClarifyAction::ContinueCurrent,
        });
    }
    let mut seen = std::collections::BTreeSet::new();
    for c in past {
        if !seen.insert(c.index) {
            continue;
        }
        let label = labels
            .get(c.index)
            .cloned()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("topic #{}", c.index));
        candidates.push(ClarifyCandidate {
            label,
            action: ClarifyAction::ReturnTo {
                chunk_index: c.index,
            },
        });
    }
    Clarification {
        question: format_clarify_question(&candidates),
        candidates,
    }
}

fn format_clarify_question(candidates: &[ClarifyCandidate]) -> String {
    let mut lines = vec!["どの話に戻りますか？".to_string()];
    for (i, c) in candidates.iter().enumerate() {
        lines.push(format!("{}. {}", i + 1, c.label));
    }
    lines.push("番号か話題のキーワードで答えてください。".into());
    lines.join("\n")
}

/// Match a user reply against a pending clarification.
pub fn match_clarification(user: &str, clarification: &Clarification) -> Option<ClarifyAction> {
    let trimmed = user.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(n) = trimmed.parse::<usize>() {
        if (1..=clarification.candidates.len()).contains(&n) {
            return Some(clarification.candidates[n - 1].action);
        }
    }
    let lower = trimmed.to_lowercase();
    for c in &clarification.candidates {
        let label = c.label.to_lowercase();
        if lower.contains(&label) || label.contains(&lower) {
            return Some(c.action);
        }
    }
    for c in &clarification.candidates {
        let label = c.label.to_lowercase();
        let key: String = label.chars().take(6).collect();
        if key.chars().count() >= 2 && lower.contains(&key) {
            return Some(c.action);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s2::GraySafetyS2;

    fn unit(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[hot] = 1.0;
        v
    }

    fn input<'a>(
        user: &'a str,
        q: &'a [f32],
        cur: Option<&'a [f32]>,
        past: &'a [ChunkScore<'a>],
        previous_chunk: Option<usize>,
        labels: &'a [String],
        s2: Option<&'a mut dyn TopicS2>,
    ) -> ResolveInput<'a> {
        ResolveInput {
            user,
            query_emb: q,
            current: cur,
            past,
            previous_chunk,
            chunk_labels: labels,
            thresholds: S1Thresholds::default(),
            ambiguity_delta: DEFAULT_AMBIGUITY_DELTA,
            s2,
        }
    }

    #[test]
    fn unspecified_uses_previous_chunk() {
        let q = unit(4, 0);
        let cur = unit(4, 1);
        let past_emb = unit(4, 0);
        let past = [ChunkScore {
            index: 0,
            embedding: &past_emb,
        }];
        let labels = ["subway".into(), "cooking".into()];
        let mut inp = input("さっきの話", &q, Some(&cur), &past, Some(0), &labels, None);
        assert_eq!(
            resolve_topic(&mut inp),
            ResolveOutcome::Return { chunk_index: 0 }
        );
    }

    #[test]
    fn unspecified_without_previous_clarifies() {
        let q = unit(4, 0);
        let cur = unit(4, 2);
        let p0 = unit(4, 0);
        let p1 = unit(4, 1);
        let past = [
            ChunkScore {
                index: 0,
                embedding: &p0,
            },
            ChunkScore {
                index: 1,
                embedding: &p1,
            },
        ];
        let labels = vec!["subway".into(), "cooking".into(), "now".into()];
        let mut inp = input(
            "さっきの話に戻って",
            &q,
            Some(&cur),
            &past,
            None,
            &labels,
            None,
        );
        match resolve_topic(&mut inp) {
            ResolveOutcome::NeedsClarification(c) => {
                assert!(c.question.contains("どの話"));
                assert!(c.candidates.len() >= 3);
            }
            other => panic!("expected clarify, got {other:?}"),
        }
    }

    #[test]
    fn named_return_still_uses_s1() {
        let q = unit(4, 0);
        let cur = unit(4, 1);
        let past_emb = unit(4, 0);
        let past = [ChunkScore {
            index: 0,
            embedding: &past_emb,
        }];
        let labels = ["subway".into(), "cooking".into()];
        let th = S1Thresholds {
            continue_min: 0.95,
            return_min: 0.90,
            new_max: 0.20,
        };
        let mut inp = ResolveInput {
            user: "さっきの地下鉄の話に戻って",
            query_emb: &q,
            current: Some(&cur),
            past: &past,
            previous_chunk: Some(0),
            chunk_labels: &labels,
            thresholds: th,
            ambiguity_delta: DEFAULT_AMBIGUITY_DELTA,
            s2: None,
        };
        assert_eq!(
            resolve_topic(&mut inp),
            ResolveOutcome::Return { chunk_index: 0 }
        );
    }

    #[test]
    fn close_past_scores_clarify_via_s1_or_s2() {
        // Near-equal strong pasts: either S1 gray→S2 or ambiguous_past_return → Clarify.
        let mut q = vec![0.7f32, 0.7, 0.0, 0.0];
        crate::l2_normalize(&mut q);
        let mut p0 = vec![1.0f32, 0.05, 0.0, 0.0];
        let mut p1 = vec![0.05f32, 1.0, 0.0, 0.0];
        crate::l2_normalize(&mut p0);
        crate::l2_normalize(&mut p1);
        let cur = unit(4, 3);
        let s0 = cosine(&q, &p0);
        let s1 = cosine(&q, &p1);
        assert!((s0 - s1).abs() < 0.05, "s0={s0} s1={s1}");
        assert!(s0 > 0.60 && s1 > 0.60, "s0={s0} s1={s1}");

        let past = [
            ChunkScore {
                index: 0,
                embedding: &p0,
            },
            ChunkScore {
                index: 1,
                embedding: &p1,
            },
        ];
        let labels = vec!["alpha".into(), "beta".into(), "gamma".into()];
        let mut s2 = GraySafetyS2::default();
        let mut inp = input(
            "関連する話に戻したい",
            &q,
            Some(&cur),
            &past,
            Some(0),
            &labels,
            Some(&mut s2),
        );
        assert!(
            matches!(
                resolve_topic(&mut inp),
                ResolveOutcome::NeedsClarification(_)
            ),
            "got clarify via S2 gray; s0={s0} s1={s1}"
        );
    }

    #[test]
    fn match_clarification_by_number_and_label() {
        let c = Clarification {
            question: "q".into(),
            candidates: vec![
                ClarifyCandidate {
                    label: "今の話題のまま".into(),
                    action: ClarifyAction::ContinueCurrent,
                },
                ClarifyCandidate {
                    label: "地下鉄について一言".into(),
                    action: ClarifyAction::ReturnTo { chunk_index: 0 },
                },
            ],
        };
        assert_eq!(
            match_clarification("2", &c),
            Some(ClarifyAction::ReturnTo { chunk_index: 0 })
        );
        assert_eq!(
            match_clarification("地下鉄", &c),
            Some(ClarifyAction::ReturnTo { chunk_index: 0 })
        );
        assert_eq!(
            match_clarification("1", &c),
            Some(ClarifyAction::ContinueCurrent)
        );
    }
}
