//! loco — CLI for the loco-bot local agent.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use hf_hub::api::sync::ApiBuilder;
use loco_engine::{
    model_status, CacheLayout, InferenceBackend, ModelId, ModelSpec, ModelStatus, GEMMA4_E4B_IT,
};

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
    /// Chat with the local Gemma 4 E4B model (streaming).
    Chat {
        /// Model id (default: gemma4-e4b).
        #[arg(long, default_value = "gemma4-e4b")]
        model: String,
        /// Inference backend: cpu or gpu (metal maps to gpu).
        #[arg(long, default_value = "cpu")]
        backend: String,
        /// Optional one-shot prompt. If omitted, starts an interactive REPL.
        prompt: Option<String>,
    },
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
        Commands::Chat {
            model,
            backend,
            prompt,
        } => cmd_chat(&layout, &model, &backend, prompt.as_deref()),
    }
}

fn cmd_doctor(layout: &CacheLayout) -> Result<()> {
    println!("loco-bot doctor");
    println!("  cache root: {}", layout.root().display());
    #[cfg(feature = "inference")]
    println!("  inference:  enabled (LiteRT-LM)");
    #[cfg(not(feature = "inference"))]
    println!("  inference:  disabled (build with --features inference)");
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

    if dest.is_file() && !force {
        let bytes = std::fs::metadata(&dest)?.len();
        println!(
            "already present: {} ({:.1} MB). Use --force to re-download.",
            dest.display(),
            bytes as f64 / (1024.0 * 1024.0)
        );
        return Ok(());
    }

    std::fs::create_dir_all(dest.parent().expect("model path has parent"))
        .with_context(|| format!("create {}", dest.parent().unwrap().display()))?;

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

    if force && dest.exists() {
        std::fs::remove_file(&dest).ok();
    }
    // Prefer a stable path under our cache; hard-link when possible, else copy.
    if let Err(err) = std::fs::hard_link(&cached, &dest) {
        std::fs::copy(&cached, &dest).with_context(|| {
            format!(
                "copy downloaded file into cache (hard_link failed: {err}): {} -> {}",
                cached.display(),
                dest.display()
            )
        })?;
    }

    let bytes = std::fs::metadata(&dest)?.len();
    println!(
        "ready: {} ({:.1} MB)",
        dest.display(),
        bytes as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

fn cmd_chat(layout: &CacheLayout, model: &str, backend: &str, prompt: Option<&str>) -> Result<()> {
    #[cfg(not(feature = "inference"))]
    {
        let _ = (layout, model, backend, prompt);
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

        eprintln!(
            "loading {} ({backend}) from {} …",
            ModelSpec::for_id(id).display_name,
            path.display()
        );
        let mut session = ChatSession::open(&path, backend).context("open chat session")?;
        eprintln!("ready.\n");

        if let Some(one_shot) = prompt {
            session
                .reply_to_stdout(one_shot)
                .context("generate reply")?;
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
            if let Err(err) = session.reply_to_stdout(text) {
                eprintln!("[error] {err}");
            }
            println!();
        }
        Ok(())
    }
}
