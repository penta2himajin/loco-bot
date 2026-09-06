//! Embedding helpers and S1 topic detection for loco-bot.
//!
//! Pure cosine / cascade / deixis resolve logic is always available.
//! The bekko-a8m ONNX runtime sits behind the `ort` feature.
//! Declarative eval fixtures live under `fixtures/s1/` (`eval` module).

mod cosine;
mod deixis;
mod eval;
mod expand;
mod resolve;
mod s1;
mod s2;

#[cfg(feature = "ort")]
mod ort_bekko;

pub use cosine::{cosine, l2_normalize};
pub use deixis::{classify_deixis, DeixisKind};
pub use eval::{
    default_bekko_dir, embedder_ready, load_suite, load_suites_dir, run_suite,
    run_suite_with_embedder, CaseResult, EvalCase, EvalError, EvalSuite, SuiteReport, TextEmbedder,
};
pub use expand::expand_query;
pub use resolve::{
    match_clarification, resolve_topic, Clarification, ClarifyAction, ClarifyCandidate,
    ResolveInput, ResolveOutcome, DEFAULT_AMBIGUITY_DELTA,
};
pub use s1::{
    decide_s1, decide_s1_or_new, ChunkScore, GrayEvidence, S1Outcome, S1Thresholds, TopicDecision,
};
pub use s2::{topic_from_s2, GraySafetyS2, S2Decision, TopicS2};

#[cfg(feature = "ort")]
pub use ort_bekko::{BekkoEmbedder, EmbedError};

/// Fixed embedding size for bekko-embedding-v1-a8m (Matryoshka 384-d).
pub const EMBED_DIM: usize = 384;
