//! [`AgentSession`]: one turn of resolve → compile → chat → tools.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use loco_engine::{with_session_notes, ChatSession, ModelId, ToolCall, ToolHost, GEMMA4_E4B_IT};
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

#[cfg(feature = "inference")]
use loco_engine::{default_tools_json, ensure_tool_call_id, ToolsTurnProgress};

use crate::consent::{AllowUpTo, ConsentBridge, ToolConsent, ToolRisk};
use crate::events::{AgentEvent, ClarifyChoice, TurnOutcome};
use crate::serve_protocol::{merge_tools_json, HostToolSpec, ServeServerMessage};

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
    /// Sandbox root for fs_* tools. Relative paths resolve under this directory.
    pub fs_root: Option<std::path::PathBuf>,
    /// Highest tool risk allowed without an interactive prompt.
    pub consent_ceiling: crate::consent::ToolRisk,
}

/// Progress of a serve/agent turn that may pause for host-executed tools.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentTurnProgress {
    Done(TurnOutcome),
    AwaitingHostTool {
        call_id: String,
        outcome: TurnOutcome,
    },
}

impl AgentTurnProgress {
    pub fn into_serve_message(self) -> ServeServerMessage {
        match self {
            Self::Done(outcome) => ServeServerMessage::done(outcome),
            Self::AwaitingHostTool { call_id, outcome } => {
                ServeServerMessage::awaiting_tool(call_id, outcome)
            }
        }
    }
}

#[cfg(feature = "inference")]
struct PendingHostTool {
    call: ToolCall,
    queued: Vec<ToolCall>,
    rounds_used: usize,
    events: Vec<AgentEvent>,
    memory_user: String,
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
    #[cfg(feature = "inference")]
    host_tools: HashMap<String, HostToolSpec>,
    #[cfg(feature = "inference")]
    call_seq: u64,
    #[cfg(feature = "inference")]
    pending_host: Option<PendingHostTool>,
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
        use loco_engine::model_fully_ready;

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
            Some(ToolHost::with_fs_root(
                layout.notes_path(),
                memory_path.clone(),
                config.fs_root.clone(),
            ))
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
            host_tools: HashMap::new(),
            call_seq: 0,
            pending_host: None,
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

    /// Register surface-owned tools and rebuild the conversation tool list.
    ///
    /// Clears chat history (engine stays warm). Call at session start.
    #[cfg(feature = "inference")]
    pub fn configure_host_tools(&mut self, tools: Vec<HostToolSpec>) -> Result<()> {
        if self.pending_host.is_some() {
            bail!("cannot configure host tools while awaiting a tool_result");
        }
        self.host_tools = tools.into_iter().map(|t| (t.name.clone(), t)).collect();

        if self.config.no_tools {
            return Ok(());
        }

        let host: Vec<_> = self.host_tools.values().cloned().collect();
        let merged = merge_tools_json(&default_tools_json(), &host).context("merge tools json")?;
        let notes = if self.config.no_memory {
            None
        } else {
            self.memory.system_preamble()
        };
        self.chat
            .replace_conversation(notes.as_deref(), Some(&merged))
            .context("rebuild conversation with host tools")?;
        Ok(())
    }

    /// Run one user turn; returns structured events for any UI shell.
    ///
    /// If host tools are configured and the model calls one, this returns
    /// [`AgentTurnProgress::AwaitingHostTool`] instead of blocking.
    #[cfg(feature = "inference")]
    pub fn turn(&mut self, user: &str) -> Result<TurnOutcome> {
        match self.turn_progress(user)? {
            AgentTurnProgress::Done(o) => Ok(o),
            AgentTurnProgress::AwaitingHostTool { call_id, .. } => {
                bail!("host tool {call_id} requires tool_result (use serve protocol)")
            }
        }
    }

    /// Serve-oriented turn that may pause for host tools.
    #[cfg(feature = "inference")]
    pub fn turn_progress(&mut self, user: &str) -> Result<AgentTurnProgress> {
        let mut consent = AllowUpTo {
            max: self.config.consent_ceiling,
        };
        self.turn_progress_with_consent(user, &mut consent)
    }

    #[cfg(feature = "inference")]
    pub fn turn_progress_with_consent(
        &mut self,
        user: &str,
        consent: &mut dyn ToolConsent,
    ) -> Result<AgentTurnProgress> {
        if self.pending_host.is_some() {
            bail!("awaiting tool_result for a previous host tool call");
        }

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
                            return Ok(AgentTurnProgress::Done(TurnOutcome {
                                events,
                                reply_text: question,
                            }));
                        }
                        ResolveApply::Ack(msg) => {
                            events.push(AgentEvent::Ack { text: msg.clone() });
                            events.push(AgentEvent::Done { text: msg.clone() });
                            return Ok(AgentTurnProgress::Done(TurnOutcome {
                                events,
                                reply_text: msg,
                            }));
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
        if let Some(host) = self.tools.as_ref() {
            let host_names: std::collections::HashSet<String> =
                self.host_tools.keys().cloned().collect();
            let mut bridge = ConsentBridge { inner: consent };
            let progress = self.chat.reply_with_tools_routed(
                &payload,
                host,
                |name| host_names.contains(name),
                &mut bridge,
                |name, args, allowed| {
                    events.push(AgentEvent::ToolRequest {
                        name: name.to_string(),
                        arguments: args.clone(),
                        risk: ToolRisk::for_tool(name).as_str().to_string(),
                        call_id: None,
                    });
                    events.push(AgentEvent::ToolResult {
                        name: name.to_string(),
                        ok: allowed,
                        call_id: None,
                    });
                },
            )?;
            self.progress_from_tools(progress, events, user.to_string())
        } else {
            let reply = self.chat.reply(&payload)?;
            events.push(AgentEvent::Done {
                text: reply.clone(),
            });
            Ok(AgentTurnProgress::Done(TurnOutcome {
                events,
                reply_text: reply,
            }))
        }
    }

    /// Fulfill a paused host tool call from the surface.
    #[cfg(feature = "inference")]
    pub fn resume_host_tool(
        &mut self,
        call_id: &str,
        ok: bool,
        content: serde_json::Value,
    ) -> Result<AgentTurnProgress> {
        let pending = self
            .pending_host
            .take()
            .context("no host tool is awaiting a result")?;
        let expected = pending.call.id.as_deref().unwrap_or("");
        if expected != call_id {
            // put back so the client can retry with the right id
            let expected_owned = expected.to_string();
            self.pending_host = Some(pending);
            bail!("call_id mismatch: expected {expected_owned}, got {call_id}");
        }

        let mut events = pending.events;
        events.push(AgentEvent::ToolResult {
            name: pending.call.name.clone(),
            ok,
            call_id: Some(call_id.to_string()),
        });

        let response = if ok {
            content
        } else {
            serde_json::json!({
                "error": "tool denied or failed on host",
                "tool": pending.call.name,
                "detail": content,
            })
        };

        let Some(host) = self.tools.as_ref() else {
            bail!("tools disabled");
        };
        let host_names: std::collections::HashSet<String> =
            self.host_tools.keys().cloned().collect();
        let mut consent = AllowUpTo {
            max: self.config.consent_ceiling,
        };
        let mut bridge = ConsentBridge {
            inner: &mut consent,
        };
        let progress = self.chat.resume_with_host_tool_result(
            loco_engine::HostToolResume {
                call: &pending.call,
                response,
                queued: pending.queued,
                rounds_used: pending.rounds_used,
            },
            host,
            |name| host_names.contains(name),
            &mut bridge,
            |name, args, allowed| {
                events.push(AgentEvent::ToolRequest {
                    name: name.to_string(),
                    arguments: args.clone(),
                    risk: ToolRisk::for_tool(name).as_str().to_string(),
                    call_id: None,
                });
                events.push(AgentEvent::ToolResult {
                    name: name.to_string(),
                    ok: allowed,
                    call_id: None,
                });
            },
        )?;
        self.progress_from_tools(progress, events, pending.memory_user)
    }

    #[cfg(feature = "inference")]
    fn progress_from_tools(
        &mut self,
        progress: ToolsTurnProgress,
        mut events: Vec<AgentEvent>,
        memory_user: String,
    ) -> Result<AgentTurnProgress> {
        match progress {
            ToolsTurnProgress::Done(reply) => {
                events.push(AgentEvent::Done {
                    text: reply.clone(),
                });
                Ok(AgentTurnProgress::Done(TurnOutcome {
                    events,
                    reply_text: reply,
                }))
            }
            ToolsTurnProgress::NeedHostTool {
                mut call,
                queued,
                rounds_used,
            } => {
                self.call_seq += 1;
                let fallback = format!("host-{}", self.call_seq);
                ensure_tool_call_id(&mut call, fallback);
                let call_id = call.id.clone().unwrap_or_default();
                let risk = self
                    .host_tools
                    .get(&call.name)
                    .map(|s| s.risk.clone())
                    .unwrap_or_else(|| ToolRisk::for_tool(&call.name).as_str().to_string());
                events.push(AgentEvent::ToolRequest {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    risk,
                    call_id: Some(call_id.clone()),
                });
                self.pending_host = Some(PendingHostTool {
                    call,
                    queued,
                    rounds_used,
                    events: events.clone(),
                    memory_user,
                });
                Ok(AgentTurnProgress::AwaitingHostTool {
                    call_id,
                    outcome: TurnOutcome {
                        events,
                        reply_text: String::new(),
                    },
                })
            }
        }
    }

    /// Last user text for a paused host tool (for memory append on Done).
    #[cfg(feature = "inference")]
    pub fn pending_memory_user(&self) -> Option<&str> {
        self.pending_host.as_ref().map(|p| p.memory_user.as_str())
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
