//! Bekko-a8m ONNX embedder (mean pool + L2 normalize).

use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::cosine::l2_normalize;
use crate::EMBED_DIM;

const MAX_SEQ: usize = 512;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("onnx: {0}")]
    Onnx(String),
    #[error("missing model input `{0}`")]
    MissingInput(String),
    #[error("unexpected output shape")]
    BadShape,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl EmbedError {
    fn onnx(err: impl std::fmt::Display) -> Self {
        Self::Onnx(err.to_string())
    }
}

/// Loads `onnx/model.onnx` + `tokenizer.json` and embeds text to 384-d vectors.
///
/// Pooling matches Sentence Transformers: mean over non-padding tokens, then L2.
pub struct BekkoEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    input_ids_name: String,
    attention_mask_name: String,
}

impl BekkoEmbedder {
    /// Open from a model directory that contains `onnx/model.onnx` and `tokenizer.json`.
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self, EmbedError> {
        let model_dir = model_dir.as_ref();
        let onnx = model_dir.join("onnx").join("model.onnx");
        let tok_path = model_dir.join("tokenizer.json");
        Self::open_files(&onnx, &tok_path)
    }

    pub fn open_files(onnx: &Path, tokenizer_json: &Path) -> Result<Self, EmbedError> {
        let tokenizer = Tokenizer::from_file(tokenizer_json)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let session = Session::builder()
            .map_err(EmbedError::onnx)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(EmbedError::onnx)?
            .with_intra_threads(2)
            .map_err(EmbedError::onnx)?
            .commit_from_file(onnx)
            .map_err(EmbedError::onnx)?;

        let input_ids_name = find_input_name(&session, &["input_ids"])?;
        let attention_mask_name = find_input_name(&session, &["attention_mask"])?;

        Ok(Self {
            session,
            tokenizer,
            input_ids_name,
            attention_mask_name,
        })
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mut mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        if ids.len() > MAX_SEQ {
            ids.truncate(MAX_SEQ);
            mask.truncate(MAX_SEQ);
        }
        let seq = ids.len();
        if seq == 0 {
            return Ok(vec![0.0; EMBED_DIM]);
        }

        let ids_tensor = Tensor::from_array(([1usize, seq], ids)).map_err(EmbedError::onnx)?;
        let mask_tensor =
            Tensor::from_array(([1usize, seq], mask.clone())).map_err(EmbedError::onnx)?;

        let outputs = self
            .session
            .run(ort::inputs![
                self.input_ids_name.as_str() => ids_tensor,
                self.attention_mask_name.as_str() => mask_tensor,
            ])
            .map_err(EmbedError::onnx)?;

        let (_out_name, out_value) = outputs
            .iter()
            .find(|(name, _)| {
                let n = name.to_ascii_lowercase();
                n.contains("last_hidden") || n == "output" || n.contains("token_embeddings")
            })
            .or_else(|| outputs.iter().next())
            .ok_or(EmbedError::BadShape)?;

        let (shape, data) = out_value
            .try_extract_tensor::<f32>()
            .map_err(EmbedError::onnx)?;
        mean_pool_l2(shape, data, &mask)
    }
}

fn mean_pool_l2(shape: &[i64], data: &[f32], mask: &[i64]) -> Result<Vec<f32>, EmbedError> {
    let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    let mut emb = match dims.as_slice() {
        [1, s, h] if *h == EMBED_DIM => mean_pool(data, *s, *h, mask)?,
        [1, h] if *h == EMBED_DIM => data[..EMBED_DIM].to_vec(),
        [h] if *h == EMBED_DIM => data[..EMBED_DIM].to_vec(),
        [batch, s, h] if *batch >= 1 && *h == EMBED_DIM => mean_pool(data, *s, *h, mask)?,
        _ => return Err(EmbedError::BadShape),
    };
    l2_normalize(&mut emb);
    Ok(emb)
}

fn mean_pool(data: &[f32], seq: usize, dim: usize, mask: &[i64]) -> Result<Vec<f32>, EmbedError> {
    if data.len() < seq * dim || mask.len() < seq {
        return Err(EmbedError::BadShape);
    }
    let mut out = vec![0.0f32; dim];
    let mut count = 0.0f32;
    for (t, &m) in mask.iter().enumerate().take(seq) {
        if m == 0 {
            continue;
        }
        let off = t * dim;
        for d in 0..dim {
            out[d] += data[off + d];
        }
        count += 1.0;
    }
    if count > 0.0 {
        for v in &mut out {
            *v /= count;
        }
    }
    Ok(out)
}

fn find_input_name(session: &Session, candidates: &[&str]) -> Result<String, EmbedError> {
    let inputs = session.inputs();
    for input in inputs.iter() {
        let name = input.name();
        if candidates.iter().any(|c| name.eq_ignore_ascii_case(c)) {
            return Ok(name.to_string());
        }
    }
    if candidates[0] == "input_ids" {
        inputs
            .first()
            .map(|i| i.name().to_string())
            .ok_or_else(|| EmbedError::MissingInput("input_ids".into()))
    } else {
        inputs
            .get(1)
            .or_else(|| inputs.first())
            .map(|i| i.name().to_string())
            .ok_or_else(|| EmbedError::MissingInput(candidates[0].into()))
    }
}

#[cfg(all(test, feature = "ort"))]
mod smoke {
    use super::*;
    use crate::cosine::cosine;
    use std::path::PathBuf;

    fn cache_dir() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("LOCO_BEKKO_DIR") {
            return Some(PathBuf::from(p));
        }
        let home = std::env::var_os("HOME")?;
        Some(PathBuf::from(home).join("Library/Caches/loco-bot/models/bekko-a8m"))
    }

    #[test]
    fn real_bekko_mean_embed_if_cached() {
        let Some(dir) = cache_dir() else {
            eprintln!("skip: no cache dir");
            return;
        };
        if !dir.join("onnx/model.onnx").is_file() || !dir.join("tokenizer.json").is_file() {
            eprintln!("skip: bekko not at {}", dir.display());
            return;
        }
        let mut emb = BekkoEmbedder::open(&dir).expect("open bekko");
        let a = emb.embed("東京の地下鉄について教えて").expect("embed a");
        let b = emb
            .embed("Tell me about the Tokyo subway")
            .expect("embed b");
        let c = emb.embed("バナナのスムージーの作り方").expect("embed c");
        assert_eq!(a.len(), EMBED_DIM);
        let related = cosine(&a, &b);
        let unrelated = cosine(&a, &c);
        assert!(
            related > unrelated,
            "expected subway pair closer than banana: related={related} unrelated={unrelated}"
        );
    }
}
