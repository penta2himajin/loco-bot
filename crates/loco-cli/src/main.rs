//! loco — CLI for the loco-bot local agent.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use hf_hub::api::sync::ApiBuilder;
use loco_engine::{
    install_model_file, model_fully_ready, model_status, with_session_notes, CacheLayout,
    InferenceBackend, ModelId, ModelSpec, ModelStatus, GEMMA4_E4B_IT, GRANITE_97M,
};
use loco_memory::SessionMemory;

#[cfg(feature = "inference")]
use loco_engine::ChatSession;

#[cfg(feature = "embed")]
use loco_embed::{
    decide_s1, expand_query, ChunkScore, GraniteEmbedder, S1Thresholds, TopicDecision,
};

#[derive(Debug, Parser)]
#[command(
    name = "loco",
    about = "Small on-device agent powered by LiteRT-LM + Gemma 4 E4B"
)]
struct Cli {
    /// Override cache root (default: platform cache dir / loco-bot).
    #[arg(long, global = true, env = "LOCO_CACHE_DIR")]
    cache_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Show toolchain / model readiness.
    Doctor,
    /// Download a model into the local cache (first-run style).
    Download {
        /// Model id: gemma4-e4b | granite-97m
        #[arg(default_value = "gemma4-e4b")]
        model: String,
        /// Re-download even if files already exist.
        #[arg(long)]
        force: bool,
    },
    /// List known models and whether they are cached.
    Models,
    /// Inspect or clear thin session memory.
    Memory {
        #[command(subcommand)]
        action: MemoryCmd,
    },
    /// Chat with the local Gemma 4 E4B model (streaming).
    Chat {
        /// Model id (default: gemma4-e4b).
        #[arg(long, default_value = "gemma4-e4b")]
        model: String,
        /// Inference backend: cpu or gpu (metal maps to gpu).
        #[arg(long, default_value = "cpu")]
        backend: String,
        /// Do not load/save session memory.
        #[arg(long)]
        no_memory: bool,
        /// Skip S1 topic detection even if granite is cached.
        #[arg(long)]
        no_topic: bool,
        /// Optional one-shot prompt. If omitted, starts an interactive REPL.
        prompt: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
enum MemoryCmd {
    /// Show turn count, chunks, and rolling summary.
    Show,
    /// Delete persisted session memory.
    Clear,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let layout = CacheLayout::new(
        cli.cache_dir
            .unwrap_or_else(loco_engine::default_cache_root),
    );

    match cli.command {
        Commands::Doctor => cmd_doctor(&layout),
        Commands::Download { model, force } => cmd_download(&layout, &model, force),
        Commands::Models => cmd_models(&layout),
        Commands::Memory { action } => cmd_memory(&layout, action),
        Commands::Chat {
            model,
            backend,
            no_memory,
            no_topic,
            prompt,
        } => cmd_chat(
            &layout,
            &model,
            &backend,
            no_memory,
            no_topic,
            prompt.as_deref(),
        ),
    }
}

fn cmd_doctor(layout: &CacheLayout) -> Result<()> {
    println!("loco-bot doctor");
    println!("  cache root: {}", layout.root().display());
    #[cfg(feature = "inference")]
    println!("  inference:  enabled (LiteRT-LM)");
    #[cfg(not(feature = "inference"))]
    println!("  inference:  disabled (build with --features inference)");
    #[cfg(feature = "embed")]
    println!("  embed:      enabled (ONNX granite)");
    #[cfg(not(feature = "embed"))]
    println!("  embed:      disabled (build with --features embed)");
    let mem = SessionMemory::load(layout.memory_path()).unwrap_or_default();
    println!(
        "  memory:     {} turns, {} chunks ({})",
        mem.turns.len(),
        mem.chunks.len(),
        layout.memory_path().display()
    );
    println!();

    print_model_line(layout, &GEMMA4_E4B_IT);
    print_model_line(layout, &GRANITE_97M);

    if !model_fully_ready(layout, ModelId::Gemma4E4b) {
        println!();
        println!("Next: loco download gemma4-e4b");
        bail!("primary chat model is not ready");
    }
    if !model_fully_ready(layout, ModelId::Granite97m) {
        println!();
        println!("Tip: loco download granite-97m  # enables S1 topic detection");
    }
    Ok(())
}

fn cmd_models(layout: &CacheLayout) -> Result<()> {
    for id in ModelId::all() {
        print_model_line(layout, ModelSpec::for_id(*id));
    }
    Ok(())
}

fn print_model_line(layout: &CacheLayout, spec: &ModelSpec) {
    if model_fully_ready(layout, spec.id) {
        let mut total = 0u64;
        for rel in spec.files {
            if let ModelStatus::Present { bytes, .. } = model_status_rel(layout, spec.id, rel) {
                total += bytes;
            }
        }
        println!(
            "  {} [{}]: OK ({:.1} MB, {} files)",
            spec.display_name,
            spec.id,
            total as f64 / (1024.0 * 1024.0),
            spec.files.len()
        );
    } else {
        let missing: Vec<_> = spec
            .files
            .iter()
            .filter(|rel| !model_status_rel(layout, spec.id, rel).is_ready())
            .copied()
            .collect();
        println!(
            "  {} [{}]: MISSING ({})",
            spec.display_name,
            spec.id,
            missing.join(", ")
        );
    }
}

fn model_status_rel(layout: &CacheLayout, id: ModelId, relative: &str) -> ModelStatus {
    loco_engine::file_status(layout, id, relative)
}

fn cmd_memory(layout: &CacheLayout, action: MemoryCmd) -> Result<()> {
    let path = layout.memory_path();
    match action {
        MemoryCmd::Show => {
            let mem = SessionMemory::load(&path)?;
            println!("memory file: {}", path.display());
            println!("turns: {}", mem.turns.len());
            println!("recent_n: {}", mem.recent_n);
            println!("chunks: {}", mem.chunks.len());
            if let Some(i) = mem.current_chunk {
                println!("current_chunk: {i}");
            }
            for (i, c) in mem.chunks.iter().enumerate() {
                println!(
                    "  [{i}] id={} turns=[{}..{}) emb_dim={} summary={}",
                    c.id,
                    c.turn_start,
                    c.turn_end,
                    c.embedding.len(),
                    truncate(&c.summary, 60)
                );
            }
            if mem.summary.is_empty() {
                println!("summary: (empty)");
            } else {
                println!("summary:\n{}", mem.summary);
            }
        }
        MemoryCmd::Clear => {
            if path.exists() {
                std::fs::remove_file(&path)
                    .with_context(|| format!("remove {}", path.display()))?;
                println!("cleared {}", path.display());
            } else {
                println!("nothing to clear ({})", path.display());
            }
        }
    }
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    let mut t: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        t.push('…');
    }
    t
}

fn cmd_download(layout: &CacheLayout, model: &str, force: bool) -> Result<()> {
    let id = ModelId::parse(model).with_context(|| format!("unknown model id: {model}"))?;
    let spec = ModelSpec::for_id(id);

    if model_fully_ready(layout, id) && !force {
        println!(
            "already present under {}. Use --force to re-download.",
            layout.model_dir(id).display()
        );
        return Ok(());
    }

    println!(
        "downloading {} from Hugging Face ({}) …",
        spec.display_name, spec.hf_repo
    );

    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("init Hugging Face API")?;
    let repo = api.model(spec.hf_repo.to_string());

    for rel in spec.files {
        let dest = layout.model_file(id, rel);
        if dest.is_file() && !force {
            let bytes = std::fs::metadata(&dest)?.len();
            println!(
                "  skip {}: already present ({:.1} MB)",
                rel,
                bytes as f64 / (1024.0 * 1024.0)
            );
            continue;
        }
        println!("  fetching {rel} …");
        let cached = repo
            .get(rel)
            .with_context(|| format!("download {}/{rel}", spec.hf_repo))?;
        let bytes = install_model_file(&cached, &dest)
            .with_context(|| format!("install into {}", dest.display()))?;
        println!(
            "  ready: {} ({:.1} MB)",
            dest.display(),
            bytes as f64 / (1024.0 * 1024.0)
        );
    }
    Ok(())
}

fn cmd_chat(
    layout: &CacheLayout,
    model: &str,
    backend: &str,
    no_memory: bool,
    no_topic: bool,
    prompt: Option<&str>,
) -> Result<()> {
    #[cfg(not(feature = "inference"))]
    {
        let _ = (layout, model, backend, no_memory, no_topic, prompt);
        bail!("chat requires the `inference` feature (default for loco-cli)");
    }

    #[cfg(feature = "inference")]
    chat_with_inference(layout, model, backend, no_memory, no_topic, prompt)
}

#[cfg(feature = "inference")]
fn chat_with_inference(
    layout: &CacheLayout,
    model: &str,
    backend: &str,
    no_memory: bool,
    no_topic: bool,
    prompt: Option<&str>,
) -> Result<()> {
    let id = ModelId::parse(model).with_context(|| format!("unknown model id: {model}"))?;
    if id != ModelId::Gemma4E4b {
        bail!("chat currently supports only gemma4-e4b (got {id})");
    }
    let backend = InferenceBackend::parse(backend)?;
    let path = match model_status(layout, id) {
        ModelStatus::Present { path, bytes } if bytes > 0 => path,
        ModelStatus::Missing { expected } => {
            bail!(
                "model not ready at {}. Run: loco download {}",
                expected.display(),
                id
            );
        }
        ModelStatus::Present { path, .. } => {
            bail!("model file is empty: {}", path.display());
        }
    };

    let memory_path = layout.memory_path();
    let mut memory = if no_memory {
        SessionMemory::default()
    } else {
        SessionMemory::load(&memory_path).context("load session memory")?
    };

    #[cfg(feature = "embed")]
    let mut embedder = load_embedder(layout, no_topic || no_memory);
    #[cfg(not(feature = "embed"))]
    let _ = no_topic;

    let notes = if no_memory {
        None
    } else {
        memory.system_preamble()
    };

    eprintln!(
        "loading {} ({backend}) from {} …",
        ModelSpec::for_id(id).display_name,
        path.display()
    );
    if let Some(ref p) = notes {
        eprintln!("memory: will attach {} chars of session notes", p.len());
    } else if !no_memory {
        eprintln!("memory: empty ({})", memory_path.display());
    }
    let mut session =
        ChatSession::open(&path, backend, notes.as_deref()).context("open chat session")?;
    eprintln!("ready.\n");

    let mut attach_notes = notes.is_some();

    if let Some(one_shot) = prompt {
        let reply = chat_reply(
            one_shot,
            &mut memory,
            &mut attach_notes,
            notes.as_deref(),
            &mut session,
            #[cfg(feature = "embed")]
            &mut embedder,
        )?;
        println!("{reply}");
        if !no_memory {
            memory.append(one_shot, &reply);
            memory.save(&memory_path).context("save session memory")?;
        }
        return Ok(());
    }

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        let n = stdin.lock().read_line(&mut line)?;
        if n == 0 {
            println!();
            break;
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if matches!(text, "/quit" | "/exit" | ":q") {
            break;
        }
        if text == "/memory" {
            println!("turns: {}", memory.turns.len());
            println!("chunks: {}", memory.chunks.len());
            if memory.summary.is_empty() {
                println!("summary: (empty)");
            } else {
                println!("{}", memory.summary);
            }
            continue;
        }
        match chat_reply(
            text,
            &mut memory,
            &mut attach_notes,
            notes.as_deref(),
            &mut session,
            #[cfg(feature = "embed")]
            &mut embedder,
        ) {
            Ok(reply) => {
                println!("{reply}");
                if !no_memory {
                    memory.append(text, &reply);
                    if let Err(err) = memory.save(&memory_path) {
                        eprintln!("[memory] save failed: {err}");
                    }
                }
            }
            Err(err) => eprintln!("[error] {err}"),
        }
        println!();
    }
    Ok(())
}

#[cfg(feature = "inference")]
fn chat_reply(
    text: &str,
    memory: &mut SessionMemory,
    attach_notes: &mut bool,
    notes: Option<&str>,
    session: &mut ChatSession,
    #[cfg(feature = "embed")] embedder: &mut Option<GraniteEmbedder>,
) -> Result<String> {
    #[cfg(feature = "embed")]
    if let Some(emb) = embedder.as_mut() {
        apply_s1(emb, memory, text)?;
    }

    let payload = if *attach_notes {
        *attach_notes = false;
        with_session_notes(notes, text)
    } else {
        text.to_string()
    };
    session.reply(&payload).context("generate reply")
}

#[cfg(all(feature = "inference", feature = "embed"))]
fn load_embedder(layout: &CacheLayout, disabled: bool) -> Option<GraniteEmbedder> {
    if disabled {
        return None;
    }
    if !model_fully_ready(layout, ModelId::Granite97m) {
        eprintln!("topic: granite-97m not cached (loco download granite-97m)");
        return None;
    }
    match GraniteEmbedder::open(layout.model_dir(ModelId::Granite97m)) {
        Ok(e) => {
            eprintln!("topic: S1 enabled (granite-97m)");
            Some(e)
        }
        Err(err) => {
            eprintln!("topic: failed to load granite ({err}); continuing without S1");
            None
        }
    }
}

#[cfg(feature = "embed")]
fn apply_s1(embedder: &mut GraniteEmbedder, memory: &mut SessionMemory, user: &str) -> Result<()> {
    let expanded = expand_query(user, memory.last_user());
    let query = embedder
        .embed(&expanded)
        .context("embed user text for S1")?;
    let past_owned: Vec<(usize, Vec<f32>)> = memory
        .past_chunk_embeddings()
        .into_iter()
        .map(|(i, e)| (i, e.to_vec()))
        .collect();
    let past: Vec<ChunkScore<'_>> = past_owned
        .iter()
        .map(|(i, e)| ChunkScore {
            index: *i,
            embedding: e.as_slice(),
        })
        .collect();
    let current = memory.current_embedding().map(|e| e.to_vec());
    let decision = decide_s1(&query, current.as_deref(), &past, &S1Thresholds::default());
    match decision {
        TopicDecision::Continue => {
            eprintln!("[topic: continue]");
        }
        TopicDecision::New => {
            let preview: String = user.chars().take(80).collect();
            let idx = memory.open_chunk(preview, query);
            eprintln!("[topic: new #{idx}]");
        }
        TopicDecision::Return { chunk_index } => {
            memory.return_to_chunk(chunk_index);
            eprintln!("[topic: return #{chunk_index}]");
        }
    }
    Ok(())
}
