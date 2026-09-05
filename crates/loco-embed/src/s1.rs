//! S1 topic cascade: continue / new / return via cosine vs chunk embeddings.
//!
//! Default thresholds were calibrated against granite-97m CLS embeddings
//! (2026-09-06, Mac cache): true Continue/Return pairs scored ≈0.80–0.90,
//! while hard negatives (e.g. quantum vs curry) sat ≈0.71–0.73. Setting
//! `continue_min` / `return_min` to 0.78 separates those bands; the residual
//! gray zone escalates to [`crate::s2`].

use crate::cosine::cosine;

/// Tunable thresholds (granite-97m calibrated defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct S1Thresholds {
    /// Current-chunk cosine at or above this → Continue.
    pub continue_min: f32,
    /// Best past-chunk cosine at or above this → Return.
    pub return_min: f32,
    /// If the best of current/past is below this → confident New.
    pub new_max: f32,
}

impl Default for S1Thresholds {
    fn default() -> Self {
        Self::granite_calibrated()
    }
}

impl S1Thresholds {
    /// Thresholds measured against granite-97m multilingual-r2 (CLS + L2).
    pub const fn granite_calibrated() -> Self {
        Self {
            continue_min: 0.78,
            return_min: 0.78,
            new_max: 0.50,
        }
    }
}

/// Final topic decision after S1 (and optional S2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopicDecision {
    Continue,
    New,
    Return { chunk_index: usize },
}

/// Evidence for the S1 gray band (neither confident Continue/Return nor New).
#[derive(Debug, Clone, PartialEq)]
pub struct GrayEvidence {
    pub current_sim: f32,
    pub best_past_index: Option<usize>,
    pub best_past_sim: f32,
    /// Past chunk scores sorted descending by cosine.
    pub past_ranked: Vec<(usize, f32)>,
}

/// Raw S1 cascade outcome before S2.
#[derive(Debug, Clone, PartialEq)]
pub enum S1Outcome {
    Continue,
    New,
    Return { chunk_index: usize },
    Gray(GrayEvidence),
}

/// Past chunk embedding with its index into the session chunk list.
#[derive(Debug, Clone, Copy)]
pub struct ChunkScore<'a> {
    pub index: usize,
    pub embedding: &'a [f32],
}

/// Decide topic transition for `query` against the current chunk and past chunks.
///
/// Cascade (no EMA on the query vector):
/// 1. No current chunk → New
/// 2. Current sim ≥ `continue_min` → Continue
/// 3. Best past sim ≥ `return_min` → Return
/// 4. Best of current/past < `new_max` → New
/// 5. Else → [`S1Outcome::Gray`] (caller should run S2)
pub fn decide_s1(
    query: &[f32],
    current: Option<&[f32]>,
    past: &[ChunkScore<'_>],
    th: &S1Thresholds,
) -> S1Outcome {
    let Some(current_emb) = current else {
        return S1Outcome::New;
    };

    let current_sim = cosine(query, current_emb);
    if current_sim >= th.continue_min {
        return S1Outcome::Continue;
    }

    let mut past_ranked: Vec<(usize, f32)> = past
        .iter()
        .map(|c| (c.index, cosine(query, c.embedding)))
        .collect();
    past_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let (best_past_index, best_past_sim) = past_ranked
        .first()
        .copied()
        .map(|(i, s)| (Some(i), s))
        .unwrap_or((None, f32::NEG_INFINITY));

    if let Some(index) = best_past_index {
        if best_past_sim >= th.return_min {
            return S1Outcome::Return { chunk_index: index };
        }
    }

    let best_overall = current_sim.max(if best_past_sim.is_finite() {
        best_past_sim
    } else {
        0.0
    });
    if best_overall < th.new_max {
        return S1Outcome::New;
    }

    S1Outcome::Gray(GrayEvidence {
        current_sim,
        best_past_index,
        best_past_sim: if best_past_sim.is_finite() {
            best_past_sim
        } else {
            0.0
        },
        past_ranked,
    })
}

/// Convenience: map [`S1Outcome`] to [`TopicDecision`], treating Gray as New.
pub fn decide_s1_or_new(
    query: &[f32],
    current: Option<&[f32]>,
    past: &[ChunkScore<'_>],
    th: &S1Thresholds,
) -> TopicDecision {
    match decide_s1(query, current, past, th) {
        S1Outcome::Continue => TopicDecision::Continue,
        S1Outcome::New => TopicDecision::New,
        S1Outcome::Return { chunk_index } => TopicDecision::Return { chunk_index },
        S1Outcome::Gray(_) => TopicDecision::New,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[hot] = 1.0;
        v
    }

    #[test]
    fn no_current_means_new() {
        assert_eq!(
            decide_s1(&unit(4, 0), None, &[], &S1Thresholds::default()),
            S1Outcome::New
        );
    }

    #[test]
    fn high_current_continues() {
        let q = unit(4, 0);
        let cur = unit(4, 0);
        assert_eq!(
            decide_s1(&q, Some(&cur), &[], &S1Thresholds::default()),
            S1Outcome::Continue
        );
    }

    #[test]
    fn strong_past_returns() {
        let q = unit(4, 2);
        let cur = unit(4, 0);
        let past_emb = unit(4, 2);
        let past = [ChunkScore {
            index: 0,
            embedding: &past_emb,
        }];
        let th = S1Thresholds {
            continue_min: 0.95,
            return_min: 0.90,
            new_max: 0.20,
        };
        assert_eq!(
            decide_s1(&q, Some(&cur), &past, &th),
            S1Outcome::Return { chunk_index: 0 }
        );
    }

    #[test]
    fn both_weak_opens_new() {
        let q = unit(4, 3);
        let cur = unit(4, 0);
        let past_emb = unit(4, 1);
        let past = [ChunkScore {
            index: 0,
            embedding: &past_emb,
        }];
        let th = S1Thresholds {
            continue_min: 0.95,
            return_min: 0.95,
            new_max: 0.50,
        };
        assert_eq!(decide_s1(&q, Some(&cur), &past, &th), S1Outcome::New);
    }

    #[test]
    fn mid_band_is_gray_not_silent_new() {
        let mut q = vec![0.7f32, 0.7, 0.0, 0.0];
        let mut cur = vec![1.0f32, 0.0, 0.0, 0.0];
        crate::l2_normalize(&mut q);
        crate::l2_normalize(&mut cur);
        let sim = cosine(&q, &cur);
        let th = S1Thresholds::default();
        assert!(
            sim > th.new_max && sim < th.continue_min,
            "sim={sim} th={th:?}"
        );
        assert!(matches!(
            decide_s1(&q, Some(&cur), &[], &th),
            S1Outcome::Gray(_)
        ));
        assert_eq!(
            decide_s1_or_new(&q, Some(&cur), &[], &th),
            TopicDecision::New
        );
    }

    #[test]
    fn calibrated_defaults_split_measured_bands() {
        let th = S1Thresholds::granite_calibrated();
        assert!((th.continue_min - 0.78).abs() < f32::EPSILON);
        assert!((th.return_min - 0.78).abs() < f32::EPSILON);
        // Positive band (~0.80+) above; hard-negative band (~0.73) below.
        assert!(0.82 >= th.return_min);
        assert!(0.73 < th.return_min);
    }
}
