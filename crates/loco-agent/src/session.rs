//! [`AgentSession`]: one turn of resolve → compile → chat → tools.

use anyhow::{Context, Result};
use loco_engine::{with_session_notes, ChatSession, ModelId, ToolHost, GEMMA4_E4B_IT};
use loco_memory::{
    compile, CompilerConfig, PendingCandidate, PendingClarification, SessionMemory, TopicSwitch,
};
use thiserror::Error;

#[cfg(feature = "embed")]
use loco_embed::{
    expand_query, match_clarification, resolve_topic, BekkoEmbedder, ChunkScore, Clarification,
    ClarifyAction, ClarifyCandidate, GraySafetyS2, ResolveInput, ResolveOutcome, S1Thresholds,
    TopicS2, DEFAULT_AMBIGUITY_DELTA,
};

use crate::consent::{AllowLowRiskOnly, ConsentBridge, ToolConsent, ToolRisk};
use crate::events::{AgentEvent, ClarifyChoice, TurnOutcome};

#[derive(Debug, Error)]
pub enum AgentError {
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
    #[cfg(feature = "inference")]
    #[error(transparent)]
    Chat(#[from] loco_engine::ChatError),
}

/// Configuration for constructing an [`AgentSession`].
#[derive(Debug, Clone, Default)]
pub struct AgentSessionConfig {
    pub no_memory: bool,
    pub no_tools: bool,
    pub no_topic: bool,
}

/// Owns memory + optional embedder + chat session for surface-agnostic turns.
pub struct AgentSession {
    pub memory: SessionMemory,
    pub memory_path: std::path::PathBuf,
    config: AgentSessionConfig,
    #[cfg(feature = "inference")]
    chat: ChatSession,
    #[cfg(feature = "inference")]
    tools: Option<ToolHost>,
    #[cfg(feature = "embed")]
    embedder: Option<BekkoEmbedder>,
}

impl AgentSession {
    /// Open chat (+ optional tools / embedder) against an existing cache layout.
    #[cfg(feature = "inference")]
    pub fn open(
        layout: &loco_engine::CacheLayout,
        backend: loco_engine::InferenceBackend,
        memory: SessionMemory,
        config: AgentSessionConfig,
    ) -> Result<Self> {
        use loco_engine::{default_tools_json, model_fully_ready};

        let model_path = layout.model_path(&GEMMA4_E4B_IT);
        let memory_path = layout.memory_path();

        let notes = if config.no_memory {
            None
        } else {
            memory.system_preamble()
        };
        let tools_json = if config.no_tools {
            None
        } else {
            Some(default_tools_json())
        };
        let tools = if config.no_tools {
            None
        } else {
            Some(ToolHost::new(layout.notes_path(), memory_path.clone()))
        };

        let chat = ChatSession::open(
            &model_path,
            backend,
            notes.as_deref(),
            tools_json.as_deref(),
        )
        .context("open chat session")?;

        #[cfg(feature = "embed")]
        let embedder =
            if config.no_topic || config.no_memory || !model_fully_ready(layout, ModelId::BekkoA8m)
            {
                None
            } else {
                BekkoEmbedder::open(layout.model_dir(ModelId::BekkoA8m)).ok()
            };

        Ok(Self {
            memory,
            memory_path,
            config,
            chat,
            tools,
            #[cfg(feature = "embed")]
            embedder,
        })
    }

    /// Whether S1 topic detection is active for this session.
    #[cfg(feature = "embed")]
    pub fn topic_active(&self) -> bool {
        self.embedder.is_some()
    }

    #[cfg(not(feature = "embed"))]
    pub fn topic_active(&self) -> bool {
        false
    }

    pub fn tools_active(&self) -> bool {
        #[cfg(feature = "inference")]
        {
            self.tools.is_some()
        }
        #[cfg(not(feature = "inference"))]
        {
            false
        }
    }

    /// Run one user turn; returns structured events for any UI shell.
    #[cfg(feature = "inference")]
    pub fn turn(&mut self, user: &str) -> Result<TurnOutcome> {
        self.turn_with_consent(user, &mut AllowLowRiskOnly)
    }

    /// Same as [`Self::turn`] with an explicit consent policy.
    #[cfg(feature = "inference")]
    pub fn turn_with_consent(
        &mut self,
        user: &str,
        consent: &mut dyn ToolConsent,
    ) -> Result<TurnOutcome> {
        let mut events = Vec::new();

        let switch = if self.config.no_memory {
            TopicSwitch::Continue
        } else {
            #[cfg(feature = "embed")]
            {
                if let Some(emb) = self.embedder.as_mut() {
                    match apply_resolve(emb, &mut self.memory, user)? {
                        ResolveApply::Clarify(question) => {
                            let choices = pending_choices(&self.memory);
                            events.push(AgentEvent::Clarify {
                                question: question.clone(),
                                choices,
                            });
                            events.push(AgentEvent::Done {
                                text: question.clone(),
                            });
                            return Ok(TurnOutcome {
                                events,
                                reply_text: question,
                            });
                        }
                        ResolveApply::Ack(msg) => {
                            events.push(AgentEvent::Ack { text: msg.clone() });
                            events.push(AgentEvent::Done { text: msg.clone() });
                            return Ok(TurnOutcome {
                                events,
                                reply_text: msg,
                            });
                        }
                        ResolveApply::Switch(sw) => {
                            push_topic_event(&mut events, &sw);
                            sw
                        }
                    }
                } else {
                    TopicSwitch::Continue
                }
            }
            #[cfg(not(feature = "embed"))]
            {
                TopicSwitch::Continue
            }
        };

        let cfg = match switch {
            TopicSwitch::Return { .. } => CompilerConfig::default(),
            TopicSwitch::Continue | TopicSwitch::New => CompilerConfig {
                recent_turn_window: 0,
                ..CompilerConfig::default()
            },
        };
        let compiled = if self.config.no_memory {
            None
        } else {
            let ctx = compile(&self.memory, switch, &cfg);
            let notes = ctx.render_notes(&cfg);
            if let Some(ref n) = notes {
                events.push(AgentEvent::Context {
                    chars: n.len(),
                    has_dynamic: ctx.dynamic.is_some(),
                });
            }
            notes
        };

        let payload = with_session_notes(compiled.as_deref(), user);
        let reply = if let Some(host) = self.tools.as_ref() {
            let mut bridge = ConsentBridge { inner: consent };
            self.chat.reply_with_tools_consent(
                &payload,
                host,
                &mut bridge,
                |name, args, allowed| {
                    events.push(AgentEvent::ToolRequest {
                        name: name.to_string(),
                        arguments: args.clone(),
                        risk: ToolRisk::for_tool(name).as_str().to_string(),
                    });
                    events.push(AgentEvent::ToolResult {
                        name: name.to_string(),
                        ok: allowed,
                    });
                },
            )?
        } else {
            self.chat.reply(&payload)?
        };

        events.push(AgentEvent::Done {
            text: reply.clone(),
        });
        Ok(TurnOutcome {
            events,
            reply_text: reply,
        })
    }

    pub fn save_memory(&self) -> Result<()> {
        if self.config.no_memory {
            return Ok(());
        }
        self.memory
            .save(&self.memory_path)
            .context("save session memory")?;
        Ok(())
    }
}

fn push_topic_event(events: &mut Vec<AgentEvent>, switch: &TopicSwitch) {
    let (kind, chunk_index) = match switch {
        TopicSwitch::Continue => ("continue", None),
        TopicSwitch::New => ("new", None),
        TopicSwitch::Return { chunk_index } => ("return", Some(*chunk_index)),
    };
    events.push(AgentEvent::Topic {
        kind: kind.into(),
        chunk_index,
    });
}

fn pending_choices(memory: &SessionMemory) -> Vec<ClarifyChoice> {
    memory
        .pending_clarify
        .as_ref()
        .map(|p| {
            p.candidates
                .iter()
                .enumerate()
                .map(|(i, c)| ClarifyChoice {
                    label: c.label.clone(),
                    index: i + 1,
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(feature = "embed")]
enum ResolveApply {
    Switch(TopicSwitch),
    Clarify(String),
    Ack(String),
}

#[cfg(feature = "embed")]
fn apply_resolve(
    embedder: &mut BekkoEmbedder,
    memory: &mut SessionMemory,
    user: &str,
) -> Result<ResolveApply> {
    if let Some(pending) = memory.pending_clarify.clone() {
        let clarification = pending_to_clarification(&pending);
        if let Some(action) = match_clarification(user, &clarification) {
            let label = clarification
                .candidates
                .iter()
                .find(|c| c.action == action)
                .map(|c| c.label.as_str())
                .unwrap_or("selected");
            let switch = apply_clarify_action(memory, action);
            memory.clear_pending_clarify();
            if is_bare_clarify_reply(user, &pending) {
                return Ok(ResolveApply::Ack(format!(
                    "了解です。「{label}」に戻ります。続けてどうぞ。"
                )));
            }
            return Ok(ResolveApply::Switch(switch));
        }
        return Ok(ResolveApply::Clarify(pending.question));
    }

    let expanded = expand_query(user, memory.last_user());
    let query = embedder
        .embed(&expanded)
        .map_err(|e| anyhow::anyhow!("embed: {e}"))?;
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
    let labels = memory.chunk_labels();
    let mut s2 = GraySafetyS2::default();
    let mut inp = ResolveInput {
        user,
        query_emb: &query,
        current: current.as_deref(),
        past: &past,
        previous_chunk: memory.previous_chunk,
        chunk_labels: &labels,
        thresholds: S1Thresholds::default(),
        ambiguity_delta: DEFAULT_AMBIGUITY_DELTA,
        s2: Some(&mut s2 as &mut dyn TopicS2),
    };
    let outcome = resolve_topic(&mut inp);

    Ok(match outcome {
        ResolveOutcome::Continue => ResolveApply::Switch(TopicSwitch::Continue),
        ResolveOutcome::New => {
            let preview: String = user.chars().take(80).collect();
            let _idx = memory.open_chunk(preview, query);
            ResolveApply::Switch(TopicSwitch::New)
        }
        ResolveOutcome::Return { chunk_index } => {
            memory.return_to_chunk(chunk_index);
            ResolveApply::Switch(TopicSwitch::Return { chunk_index })
        }
        ResolveOutcome::NeedsClarification(c) => {
            memory.set_pending_clarify(clarification_to_pending(&c));
            ResolveApply::Clarify(c.question)
        }
    })
}

#[cfg(feature = "embed")]
fn apply_clarify_action(memory: &mut SessionMemory, action: ClarifyAction) -> TopicSwitch {
    match action {
        ClarifyAction::ContinueCurrent => TopicSwitch::Continue,
        ClarifyAction::ReturnTo { chunk_index } => {
            memory.return_to_chunk(chunk_index);
            TopicSwitch::Return { chunk_index }
        }
    }
}

#[cfg(feature = "embed")]
fn is_bare_clarify_reply(user: &str, pending: &PendingClarification) -> bool {
    let t = user.trim();
    t.parse::<usize>().is_ok()
        || pending
            .candidates
            .iter()
            .any(|c| c.label.eq_ignore_ascii_case(t))
}

#[cfg(feature = "embed")]
fn clarification_to_pending(c: &Clarification) -> PendingClarification {
    PendingClarification {
        question: c.question.clone(),
        candidates: c
            .candidates
            .iter()
            .map(|x| PendingCandidate {
                label: x.label.clone(),
                return_chunk: match x.action {
                    ClarifyAction::ContinueCurrent => None,
                    ClarifyAction::ReturnTo { chunk_index } => Some(chunk_index),
                },
            })
            .collect(),
    }
}

#[cfg(feature = "embed")]
fn pending_to_clarification(p: &PendingClarification) -> Clarification {
    Clarification {
        question: p.question.clone(),
        candidates: p
            .candidates
            .iter()
            .map(|x| ClarifyCandidate {
                label: x.label.clone(),
                action: match x.return_chunk {
                    None => ClarifyAction::ContinueCurrent,
                    Some(chunk_index) => ClarifyAction::ReturnTo { chunk_index },
                },
            })
            .collect(),
    }
}
