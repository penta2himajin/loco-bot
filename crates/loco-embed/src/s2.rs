//! S2: gray-zone topic decision after S1.
//!
//! S1 maximizes instant decisions. When cosine lands between `new_max` and
//! the Continue/Return floors, S2 picks New or asks the user — never a silent
//! false Return on weak bi-encoder scores.

use crate::s1::{GrayEvidence, TopicDecision};

/// Pluggable gray-zone resolver.
pub trait TopicS2 {
    fn decide_gray(&mut self, evidence: &GrayEvidence) -> S2Decision;
}

/// S2 outcome (may request clarification).
#[derive(Debug, Clone, PartialEq)]
pub enum S2Decision {
    Continue,
    New,
    Return {
        chunk_index: usize,
    },
    /// Ask which past topic (resolve builds the question).
    ClarifyPast,
}

/// Safety-net S2 used by default in the CLI.
///
/// Measured against bekko-a8m (2026-09-06):
/// - Hard-negative multipast New scores ≈0.04–0.16 → must stay **New**
/// - Near-floor ambiguous pairs (rare) just under return_min → **Clarify**
/// - Never soft-Return below `S1Thresholds::return_min` (0.26)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraySafetyS2 {
    /// Both top past scores must be ≥ this (just under return_min) to clarify.
    pub clarify_min: f32,
    /// Top-2 past gap below this → ambiguous.
    pub ambiguity_delta: f32,
}

impl Default for GraySafetyS2 {
    fn default() -> Self {
        Self {
            // Below return_min (0.26), above typical hard-negative pairs (~0.16).
            clarify_min: 0.23,
            ambiguity_delta: 0.05,
        }
    }
}

impl TopicS2 for GraySafetyS2 {
    fn decide_gray(&mut self, evidence: &GrayEvidence) -> S2Decision {
        if evidence.past_ranked.len() >= 2 {
            let (i0, s0) = evidence.past_ranked[0];
            let (_i1, s1) = evidence.past_ranked[1];
            if s0 >= self.clarify_min && s1 >= self.clarify_min && (s0 - s1) < self.ambiguity_delta
            {
                let _ = i0;
                return S2Decision::ClarifyPast;
            }
        }
        S2Decision::New
    }
}

/// Map an S2 decision into a final [`TopicDecision`] (Clarify stays separate).
pub fn topic_from_s2(decision: &S2Decision) -> Option<TopicDecision> {
    match decision {
        S2Decision::Continue => Some(TopicDecision::Continue),
        S2Decision::New => Some(TopicDecision::New),
        S2Decision::Return { chunk_index } => Some(TopicDecision::Return {
            chunk_index: *chunk_index,
        }),
        S2Decision::ClarifyPast => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_safety_prefers_new_on_single_mid_past() {
        let mut s2 = GraySafetyS2::default();
        let ev = GrayEvidence {
            current_sim: 0.18,
            best_past_index: Some(1),
            best_past_sim: 0.21,
            past_ranked: vec![(1, 0.21)],
        };
        assert_eq!(s2.decide_gray(&ev), S2Decision::New);
    }

    #[test]
    fn gray_safety_new_when_dual_past_below_clarify_floor() {
        // Multipast New: hard negatives stay under clarify_min.
        let mut s2 = GraySafetyS2::default();
        let ev = GrayEvidence {
            current_sim: 0.10,
            best_past_index: Some(0),
            best_past_sim: 0.16,
            past_ranked: vec![(0, 0.16), (1, 0.15)],
        };
        assert_eq!(s2.decide_gray(&ev), S2Decision::New);
    }

    #[test]
    fn gray_safety_clarifies_close_near_return_floor() {
        let mut s2 = GraySafetyS2::default();
        let ev = GrayEvidence {
            current_sim: 0.20,
            best_past_index: Some(0),
            best_past_sim: 0.245,
            past_ranked: vec![(0, 0.245), (1, 0.238)],
        };
        assert_eq!(s2.decide_gray(&ev), S2Decision::ClarifyPast);
    }
}
