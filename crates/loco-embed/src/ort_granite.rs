//! Granite-97m ONNX embedder (CLS pool + L2 normalize).

use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::cosine::l2_normalize;
use crate::GRANITE_DIM;

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
pub struct GraniteEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    input_ids_name: String,
    attention_mask_name: String,
}

impl GraniteEmbedder {
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
            return Ok(vec![0.0; GRANITE_DIM]);
        }

        let ids_tensor = Tensor::from_array(([1usize, seq], ids)).map_err(EmbedError::onnx)?;
        let mask_tensor = Tensor::from_array(([1usize, seq], mask)).map_err(EmbedError::onnx)?;

        let outputs = self
            .session
            .run(ort::inputs![
                self.input_ids_name.as_str() => ids_tensor,
                self.attention_mask_name.as_str() => mask_tensor,
            ])
            .map_err(EmbedError::onnx)?;

        let (out_name, out_value) = outputs
            .iter()
            .find(|(name, _)| {
                let n = name.to_ascii_lowercase();
                n.contains("last_hidden") || n == "output" || n.contains("token_embeddings")
            })
            .or_else(|| outputs.iter().next())
            .ok_or(EmbedError::BadShape)?;
        let _ = out_name;

        let (shape, data) = out_value
            .try_extract_tensor::<f32>()
            .map_err(EmbedError::onnx)?;

        // Shape is typically [1, seq, hidden] or [1, hidden].
        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        let mut emb = match dims.as_slice() {
            [1, _s, h] if *h == GRANITE_DIM => data[..GRANITE_DIM].to_vec(),
            [1, h] if *h == GRANITE_DIM => data[..GRANITE_DIM].to_vec(),
            [h] if *h == GRANITE_DIM => data[..GRANITE_DIM].to_vec(),
            [batch, _s, h] if *batch >= 1 && *h == GRANITE_DIM => data[..GRANITE_DIM].to_vec(),
            _ => return Err(EmbedError::BadShape),
        };

        l2_normalize(&mut emb);
        Ok(emb)
    }
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
        if let Some(p) = std::env::var_os("LOCO_GRANITE_DIR") {
            return Some(PathBuf::from(p));
        }
        let home = std::env::var_os("HOME")?;
        Some(PathBuf::from(home).join("Library/Caches/loco-bot/models/granite-97m"))
    }

    #[test]
    fn real_granite_cls_embed_if_cached() {
        let Some(dir) = cache_dir() else {
            eprintln!("skip: no cache dir");
            return;
        };
        if !dir.join("onnx/model.onnx").is_file() || !dir.join("tokenizer.json").is_file() {
            eprintln!("skip: granite not at {}", dir.display());
            return;
        }
        let mut emb = GraniteEmbedder::open(&dir).expect("open granite");
        let a = emb.embed("東京の地下鉄について教えて").expect("embed a");
        let b = emb
            .embed("Tell me about the Tokyo subway")
            .expect("embed b");
        let c = emb.embed("バナナのスムージーの作り方").expect("embed c");
        assert_eq!(a.len(), GRANITE_DIM);
        let related = cosine(&a, &b);
        let unrelated = cosine(&a, &c);
        assert!(
            related > unrelated,
            "expected subway pair closer than banana: related={related} unrelated={unrelated}"
        );
    }
}
