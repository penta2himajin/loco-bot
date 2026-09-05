//! S1 topic cascade: continue / new / return via cosine vs chunk embeddings.

use crate::cosine::cosine;

/// Tunable thresholds (chatstream-inspired defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct S1Thresholds {
    /// Current-chunk cosine at or above this → Continue.
    pub continue_min: f32,
    /// Best past-chunk cosine at or above this → Return.
    pub return_min: f32,
    /// If the best of current/past is below this → New.
    pub new_max: f32,
}

impl Default for S1Thresholds {
    fn default() -> Self {
        Self {
            continue_min: 0.75,
            return_min: 0.65,
            new_max: 0.35,
        }
    }
}

/// Outcome of one S1 decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopicDecision {
    Continue,
    New,
    Return { chunk_index: usize },
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
/// 1. No current chunk → [`TopicDecision::New`]
/// 2. Current sim ≥ `continue_min` → Continue
/// 3. Best past sim ≥ `return_min` → Return
/// 4. Best of current/past < `new_max` → New
/// 5. Gray zone → Continue (defer S2)
pub fn decide_s1(
    query: &[f32],
    current: Option<&[f32]>,
    past: &[ChunkScore<'_>],
    th: &S1Thresholds,
) -> TopicDecision {
    let Some(current_emb) = current else {
        return TopicDecision::New;
    };

    let current_sim = cosine(query, current_emb);
    if current_sim >= th.continue_min {
        return TopicDecision::Continue;
    }

    let mut best_idx = None;
    let mut best_sim = f32::NEG_INFINITY;
    for chunk in past {
        let s = cosine(query, chunk.embedding);
        if s > best_sim {
            best_sim = s;
            best_idx = Some(chunk.index);
        }
    }

    if let Some(index) = best_idx {
        if best_sim >= th.return_min {
            return TopicDecision::Return { chunk_index: index };
        }
    }

    let best_overall = current_sim.max(if best_sim.is_finite() { best_sim } else { 0.0 });
    if best_overall < th.new_max {
        return TopicDecision::New;
    }

    TopicDecision::Continue
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
        let q = unit(4, 0);
        assert_eq!(
            decide_s1(&q, None, &[], &S1Thresholds::default()),
            TopicDecision::New
        );
    }

    #[test]
    fn high_current_continues() {
        let q = unit(4, 0);
        let cur = unit(4, 0);
        assert_eq!(
            decide_s1(&q, Some(&cur), &[], &S1Thresholds::default()),
            TopicDecision::Continue
        );
    }

    #[test]
    fn strong_past_returns() {
        let q = unit(4, 2);
        let cur = unit(4, 0); // weak match
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
            TopicDecision::Return { chunk_index: 0 }
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
            new_max: 0.50, // orthogonal sims are ~0
        };
        assert_eq!(decide_s1(&q, Some(&cur), &past, &th), TopicDecision::New);
    }

    #[test]
    fn gray_zone_stays_continue() {
        // Moderate current similarity in gray band.
        let mut q = vec![0.7f32, 0.7, 0.0, 0.0];
        let mut cur = vec![1.0f32, 0.0, 0.0, 0.0];
        crate::l2_normalize(&mut q);
        crate::l2_normalize(&mut cur);
        let sim = cosine(&q, &cur);
        assert!(sim > 0.35 && sim < 0.75, "sim={sim}");
        let th = S1Thresholds::default();
        assert_eq!(decide_s1(&q, Some(&cur), &[], &th), TopicDecision::Continue);
    }
}
