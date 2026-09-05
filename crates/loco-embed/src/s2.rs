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
/// - Does **not** soft-Return on mid scores (granite hard-negatives live ~0.71–0.73).
/// - If two past chunks are both moderately strong and close → clarify.
/// - Otherwise → New (clean topic split).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraySafetyS2 {
    /// Both top past scores must be ≥ this to trigger clarify.
    pub clarify_min: f32,
    /// Top-2 past gap below this → ambiguous.
    pub ambiguity_delta: f32,
}

impl Default for GraySafetyS2 {
    fn default() -> Self {
        Self {
            clarify_min: 0.60,
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
            current_sim: 0.69,
            best_past_index: Some(1),
            best_past_sim: 0.73,
            past_ranked: vec![(1, 0.73)],
        };
        assert_eq!(s2.decide_gray(&ev), S2Decision::New);
    }

    #[test]
    fn gray_safety_clarifies_close_mid_pasts() {
        let mut s2 = GraySafetyS2::default();
        let ev = GrayEvidence {
            current_sim: 0.74,
            best_past_index: Some(0),
            best_past_sim: 0.745,
            past_ranked: vec![(0, 0.745), (1, 0.738)],
        };
        assert_eq!(s2.decide_gray(&ev), S2Decision::ClarifyPast);
    }
}
