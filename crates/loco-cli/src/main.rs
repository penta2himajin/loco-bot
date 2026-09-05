//! loco — CLI for the loco-bot local agent.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use hf_hub::api::sync::ApiBuilder;
use loco_engine::{
    install_model_file, model_status, with_session_notes, CacheLayout, InferenceBackend, ModelId,
    ModelSpec, ModelStatus, GEMMA4_E4B_IT,
};
use loco_memory::SessionMemory;

#[cfg(feature = "inference")]
use loco_engine::ChatSession;

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
        /// Model id (default: gemma4-e4b).
        #[arg(default_value = "gemma4-e4b")]
        model: String,
        /// Re-download even if a file already exists.
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
        /// Optional one-shot prompt. If omitted, starts an interactive REPL.
        prompt: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
enum MemoryCmd {
    /// Show turn count and rolling summary.
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
            prompt,
        } => cmd_chat(&layout, &model, &backend, no_memory, prompt.as_deref()),
    }
}

fn cmd_doctor(layout: &CacheLayout) -> Result<()> {
    println!("loco-bot doctor");
    println!("  cache root: {}", layout.root().display());
    #[cfg(feature = "inference")]
    println!("  inference:  enabled (LiteRT-LM)");
    #[cfg(not(feature = "inference"))]
    println!("  inference:  disabled (build with --features inference)");
    let mem = SessionMemory::load(layout.memory_path()).unwrap_or_default();
    println!(
        "  memory:     {} turns ({})",
        mem.turns.len(),
        layout.memory_path().display()
    );
    println!();

    let status = model_status(layout, ModelId::Gemma4E4b);
    print_status(&GEMMA4_E4B_IT, &status);

    if !status.is_ready() {
        println!();
        println!("Next: loco download gemma4-e4b");
        bail!("primary chat model is not ready");
    }
    Ok(())
}

fn cmd_models(layout: &CacheLayout) -> Result<()> {
    let status = model_status(layout, ModelId::Gemma4E4b);
    print_status(&GEMMA4_E4B_IT, &status);
    Ok(())
}

fn cmd_memory(layout: &CacheLayout, action: MemoryCmd) -> Result<()> {
    let path = layout.memory_path();
    match action {
        MemoryCmd::Show => {
            let mem = SessionMemory::load(&path)?;
            println!("memory file: {}", path.display());
            println!("turns: {}", mem.turns.len());
            println!("recent_n: {}", mem.recent_n);
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

fn print_status(spec: &ModelSpec, status: &ModelStatus) {
    match status {
        ModelStatus::Missing { expected } => {
            println!(
                "  {} [{}]: MISSING (expected {})",
                spec.display_name,
                spec.id,
                expected.display()
            );
        }
        ModelStatus::Present { path, bytes } => {
            println!(
                "  {} [{}]: OK ({} , {:.1} MB)",
                spec.display_name,
                spec.id,
                path.display(),
                *bytes as f64 / (1024.0 * 1024.0)
            );
        }
    }
}

fn cmd_download(layout: &CacheLayout, model: &str, force: bool) -> Result<()> {
    let id = ModelId::parse(model).with_context(|| format!("unknown model id: {model}"))?;
    let spec = ModelSpec::for_id(id);
    let dest = layout.model_path(spec);

    // `is_file` follows symlinks; a broken HF-style link counts as missing.
    if dest.is_file() && !force {
        let bytes = std::fs::metadata(&dest)?.len();
        println!(
            "already present: {} ({:.1} MB). Use --force to re-download.",
            dest.display(),
            bytes as f64 / (1024.0 * 1024.0)
        );
        return Ok(());
    }

    println!(
        "downloading {} from Hugging Face ({}/{}) …",
        spec.display_name, spec.hf_repo, spec.filename
    );

    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("init Hugging Face API")?;
    let repo = api.model(spec.hf_repo.to_string());
    let cached = repo
        .get(spec.filename)
        .with_context(|| format!("download {}/{}", spec.hf_repo, spec.filename))?;

    let bytes = install_model_file(&cached, &dest)
        .with_context(|| format!("install into {}", dest.display()))?;
    println!(
        "ready: {} ({:.1} MB)",
        dest.display(),
        bytes as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

fn cmd_chat(
    layout: &CacheLayout,
    model: &str,
    backend: &str,
    no_memory: bool,
    prompt: Option<&str>,
) -> Result<()> {
    #[cfg(not(feature = "inference"))]
    {
        let _ = (layout, model, backend, no_memory, prompt);
        bail!("chat requires the `inference` feature (default for loco-cli)");
    }

    #[cfg(feature = "inference")]
    {
        let id = ModelId::parse(model).with_context(|| format!("unknown model id: {model}"))?;
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

        if let Some(one_shot) = prompt {
            let payload = with_session_notes(notes.as_deref(), one_shot);
            let reply = session.reply(&payload).context("generate reply")?;
            println!("{reply}");
            if !no_memory {
                memory.append(one_shot, &reply);
                memory.save(&memory_path).context("save session memory")?;
            }
            return Ok(());
        }

        let mut attach_notes = notes.is_some();
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
                if memory.summary.is_empty() {
                    println!("summary: (empty)");
                } else {
                    println!("{}", memory.summary);
                }
                continue;
            }
            let payload = if attach_notes {
                attach_notes = false;
                with_session_notes(notes.as_deref(), text)
            } else {
                text.to_string()
            };
            match session.reply(&payload) {
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
}
