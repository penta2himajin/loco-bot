//! Embedding helpers and S1 topic detection for loco-bot.
//!
//! Pure cosine / cascade / deixis resolve logic is always available.
//! The granite-97m ONNX runtime sits behind the `ort` feature.
//! Declarative eval fixtures live under `fixtures/s1/` (`eval` module).

mod cosine;
mod deixis;
mod eval;
mod expand;
mod resolve;
mod s1;

#[cfg(feature = "ort")]
mod ort_granite;

pub use cosine::{cosine, l2_normalize};
pub use deixis::{classify_deixis, DeixisKind};
pub use eval::{
    default_granite_dir, granite_ready, load_suite, load_suites_dir, run_suite,
    run_suite_with_embedder, CaseResult, EvalCase, EvalError, EvalSuite, SuiteReport, TextEmbedder,
};
pub use expand::expand_query;
pub use resolve::{
    match_clarification, resolve_topic, Clarification, ClarifyAction, ClarifyCandidate,
    ResolveInput, ResolveOutcome, DEFAULT_AMBIGUITY_DELTA,
};
pub use s1::{decide_s1, ChunkScore, S1Thresholds, TopicDecision};

#[cfg(feature = "ort")]
pub use ort_granite::{EmbedError, GraniteEmbedder};

/// Fixed embedding size for granite-embedding-97m-multilingual-r2.
pub const GRANITE_DIM: usize = 384;
