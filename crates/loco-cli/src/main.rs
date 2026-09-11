//! loco — CLI for the loco-bot local agent.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use hf_hub::api::sync::ApiBuilder;
use loco_agent::{AgentEvent, AgentSession, AgentSessionConfig, ToolRisk};
use loco_engine::{
    install_model_file, model_fully_ready, model_status, CacheLayout, InferenceBackend, ModelId,
    ModelSpec, ModelStatus, BEKKO_A8M, GEMMA4_E4B_IT,
};
use loco_memory::SessionMemory;

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
        /// Model id: gemma4-e4b | bekko-a8m
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
    /// Chat with the local Gemma 4 E4B model (via Agent API).
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
        /// Skip S1 topic detection even if bekko is cached.
        #[arg(long)]
        no_topic: bool,
        /// Disable built-in tools (clock, notes, session_stats, fs_*).
        #[arg(long)]
        no_tools: bool,
        /// Sandbox root for fs_list / fs_move / fs_rename.
        #[arg(long)]
        fs_root: Option<PathBuf>,
        /// Allow medium-risk tools (local FS) without interactive prompts.
        #[arg(long)]
        allow_fs: bool,
        /// Optional one-shot prompt. If omitted, starts an interactive REPL.
        prompt: Option<String>,
    },
    /// JSONL Agent API over stdio (laptop / thin UI shells).
    Serve {
        /// Inference backend: cpu or gpu (metal maps to gpu).
        #[arg(long, default_value = "cpu")]
        backend: String,
        /// Do not load/save session memory.
        #[arg(long)]
        no_memory: bool,
        /// Skip S1 topic detection even if bekko is cached.
        #[arg(long)]
        no_topic: bool,
        /// Disable built-in tools.
        #[arg(long)]
        no_tools: bool,
        /// Sandbox root for fs_* tools.
        #[arg(long)]
        fs_root: Option<PathBuf>,
        /// Allow medium-risk tools (local FS).
        #[arg(long)]
        allow_fs: bool,
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
            no_tools,
            fs_root,
            allow_fs,
            prompt,
        } => cmd_chat(
            &layout,
            SessionLaunch {
                model,
                backend,
                no_memory,
                no_topic,
                no_tools,
                fs_root,
                allow_fs,
            },
            prompt.as_deref(),
        ),
        Commands::Serve {
            backend,
            no_memory,
            no_topic,
            no_tools,
            fs_root,
            allow_fs,
        } => cmd_serve(
            &layout,
            SessionLaunch {
                model: "gemma4-e4b".into(),
                backend,
                no_memory,
                no_topic,
                no_tools,
                fs_root,
                allow_fs,
            },
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
    println!("  embed:      enabled (ONNX bekko-a8m)");
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
    print_model_line(layout, &BEKKO_A8M);

    if !model_fully_ready(layout, ModelId::Gemma4E4b) {
        println!();
        println!("Next: loco download gemma4-e4b");
        bail!("primary chat model is not ready");
    }
    if !model_fully_ready(layout, ModelId::BekkoA8m) {
        println!();
        println!("Tip: loco download bekko-a8m  # enables S1 topic detection");
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
            if let Some(i) = mem.previous_chunk {
                println!("previous_chunk: {i}");
            }
            if mem.pending_clarify.is_some() {
                println!("pending_clarify: yes");
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

struct SessionLaunch {
    model: String,
    backend: String,
    no_memory: bool,
    no_topic: bool,
    no_tools: bool,
    fs_root: Option<PathBuf>,
    allow_fs: bool,
}

#[cfg(feature = "inference")]
fn cmd_chat(layout: &CacheLayout, launch: SessionLaunch, prompt: Option<&str>) -> Result<()> {
    let no_memory = launch.no_memory;
    let mut agent = open_agent(layout, launch)?;
    eprintln!("ready.\n");

    if let Some(one_shot) = prompt {
        let outcome = agent.turn(one_shot).context("agent turn")?;
        log_events(&outcome.events);
        println!("{}", outcome.reply_text);
        if !no_memory {
            agent.memory.append(one_shot, &outcome.reply_text);
            agent.save_memory()?;
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
            println!("turns: {}", agent.memory.turns.len());
            println!("chunks: {}", agent.memory.chunks.len());
            if agent.memory.summary.is_empty() {
                println!("summary: (empty)");
            } else {
                println!("{}", agent.memory.summary);
            }
            continue;
        }
        match agent.turn(text) {
            Ok(outcome) => {
                log_events(&outcome.events);
                println!("{}", outcome.reply_text);
                if !no_memory {
                    agent.memory.append(text, &outcome.reply_text);
                    if let Err(err) = agent.save_memory() {
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

#[cfg(not(feature = "inference"))]
fn cmd_chat(_layout: &CacheLayout, _launch: SessionLaunch, _prompt: Option<&str>) -> Result<()> {
    bail!("chat requires the `inference` feature (LiteRT-LM)")
}

/// One JSON object per stdin line → one JSON serve response per stdout line.
///
/// Supports `configure` / `turn` / `tool_result`, plus legacy `{"user":…}` as a turn.
#[cfg(feature = "inference")]
fn cmd_serve(layout: &CacheLayout, launch: SessionLaunch) -> Result<()> {
    use loco_agent::{
        parse_serve_client_line, AgentTurnProgress, ServeClientMessage, ServeServerMessage,
        TurnOutcome,
    };

    let no_memory = launch.no_memory;
    let mut agent = open_agent(layout, launch)?;
    eprintln!(
        "serve: JSONL Agent API on stdin/stdout \
         (ops: configure | turn | tool_result; legacy {{\"user\":…}} still works)"
    );
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let req = parse_serve_client_line(line).context("parse serve request JSON")?;
        let message = match req {
            ServeClientMessage::Configure { tools } => {
                agent
                    .configure_host_tools(tools)
                    .context("configure host tools")?;
                ServeServerMessage::done(TurnOutcome {
                    events: vec![],
                    reply_text: String::new(),
                })
            }
            ServeClientMessage::Turn { user } => {
                let user = user.trim();
                if user.is_empty() {
                    continue;
                }
                let progress = agent.turn_progress(user).context("agent turn")?;
                if let AgentTurnProgress::Done(ref outcome) = progress {
                    if !no_memory {
                        agent.memory.append(user, &outcome.reply_text);
                        agent.save_memory()?;
                    }
                }
                progress.into_serve_message()
            }
            ServeClientMessage::ToolResult {
                call_id,
                ok,
                content,
            } => {
                let memory_user = agent.pending_memory_user().unwrap_or("").to_string();
                let progress = agent
                    .resume_host_tool(&call_id, ok, content)
                    .context("resume host tool")?;
                if let AgentTurnProgress::Done(ref outcome) = progress {
                    if !no_memory && !memory_user.is_empty() {
                        agent.memory.append(&memory_user, &outcome.reply_text);
                        agent.save_memory()?;
                    }
                }
                progress.into_serve_message()
            }
        };
        println!("{}", serde_json::to_string(&message)?);
        io::stdout().flush().ok();
    }
    Ok(())
}

#[cfg(not(feature = "inference"))]
fn cmd_serve(_layout: &CacheLayout, _launch: SessionLaunch) -> Result<()> {
    bail!("serve requires the `inference` feature (LiteRT-LM)")
}

#[cfg(feature = "inference")]
fn open_agent(layout: &CacheLayout, launch: SessionLaunch) -> Result<AgentSession> {
    let SessionLaunch {
        model,
        backend,
        no_memory,
        no_topic,
        no_tools,
        fs_root,
        allow_fs,
    } = launch;
    let id = ModelId::parse(&model).with_context(|| format!("unknown model id: {model}"))?;
    if id != ModelId::Gemma4E4b {
        bail!("chat currently supports only gemma4-e4b (got {id})");
    }
    let backend = InferenceBackend::parse(&backend)?;
    match model_status(layout, id) {
        ModelStatus::Present { bytes, .. } if bytes > 0 => {}
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
    }

    let memory = if no_memory {
        SessionMemory::default()
    } else {
        SessionMemory::load(layout.memory_path()).context("load session memory")?
    };

    eprintln!(
        "loading {} ({backend}) …",
        ModelSpec::for_id(id).display_name
    );
    if !no_memory {
        if let Some(p) = memory.system_preamble() {
            eprintln!("memory: system preamble {} chars", p.len());
        } else {
            eprintln!("memory: empty ({})", layout.memory_path().display());
        }
    }
    if !no_tools {
        eprintln!(
            "tools: clock/notes/stats + fs_* / web_search stubs ({})",
            layout.notes_path().display()
        );
        if let Some(ref root) = fs_root {
            eprintln!("fs sandbox: {}", root.display());
        }
        if allow_fs {
            eprintln!("consent: medium (FS) allowed");
        }
    }
    if !no_topic && !no_memory {
        if model_fully_ready(layout, ModelId::BekkoA8m) {
            eprintln!("topic: S1 enabled (bekko-a8m)");
        } else {
            eprintln!("topic: bekko-a8m not cached (loco download bekko-a8m)");
        }
    }

    let config = AgentSessionConfig {
        no_memory,
        no_tools,
        no_topic,
        fs_root,
        consent_ceiling: if allow_fs {
            ToolRisk::Medium
        } else {
            ToolRisk::Low
        },
    };
    let agent = AgentSession::open(layout, backend, memory, config)?;
    if !(no_topic || no_memory || agent.topic_active())
        && model_fully_ready(layout, ModelId::BekkoA8m)
    {
        eprintln!("topic: failed to load bekko; continuing without S1");
    }
    Ok(agent)
}

fn log_events(events: &[AgentEvent]) {
    for ev in events {
        match ev {
            AgentEvent::Topic { kind, chunk_index } => match chunk_index {
                Some(i) => eprintln!("[topic: {kind} #{i}]"),
                None => eprintln!("[topic: {kind}]"),
            },
            AgentEvent::Context { chars, has_dynamic } => {
                let dyn_tag = if *has_dynamic { "+dynamic" } else { "" };
                eprintln!("[context: resident{dyn_tag} {chars} chars]");
            }
            AgentEvent::ToolRequest { name, risk, .. } => {
                eprintln!("[tool request: {name} risk={risk}]");
            }
            AgentEvent::ToolResult { name, ok, .. } => {
                eprintln!("[tool result: {name} ok={ok}]");
            }
            AgentEvent::Clarify { .. }
            | AgentEvent::Ack { .. }
            | AgentEvent::Token { .. }
            | AgentEvent::Done { .. } => {}
        }
    }
}
