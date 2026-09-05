//! Thin context compiler (chatstream-inspired, in-tree).
//!
//! Resident = rolling summary + active topic (+ optional recent turns).
//! Dynamic = returned topic chunk turns, attached only on [`TopicSwitch::Return`].

use crate::session::SessionMemory;
use crate::summary::simple_summary;
use crate::turn::Turn;

/// How the latest S1 decision should shape the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopicSwitch {
    Continue,
    New,
    Return { chunk_index: usize },
}

/// Tunables for compilation / rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompilerConfig {
    /// Recent turns to list under resident (0 = omit turn dump; summary still used).
    pub recent_turn_window: usize,
    /// Max chars per user/assistant side in rendered turns.
    pub max_chars_per_side: usize,
    /// Max turns copied from a returned chunk into dynamic context.
    pub max_dynamic_turns: usize,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            recent_turn_window: 4,
            max_chars_per_side: 120,
            max_dynamic_turns: 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentContext {
    pub state_summary: String,
    pub active_topic: Option<String>,
    pub recent_turns: Vec<Turn>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicChunk {
    pub chunk_index: usize,
    pub summary: String,
    pub turns: Vec<Turn>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledContext {
    pub resident: ResidentContext,
    pub dynamic: Option<DynamicChunk>,
}

impl CompiledContext {
    pub fn is_empty(&self) -> bool {
        self.resident.state_summary.is_empty()
            && self.resident.active_topic.is_none()
            && self.resident.recent_turns.is_empty()
            && self.dynamic.is_none()
    }

    /// Plain-text notes for in-band / system preamble injection.
    pub fn render_notes(&self, cfg: &CompilerConfig) -> Option<String> {
        let mut parts = Vec::new();

        if let Some(topic) = &self.resident.active_topic {
            if !topic.is_empty() {
                parts.push(format!("Active topic: {topic}"));
            }
        }

        if !self.resident.state_summary.is_empty() {
            parts.push(format!(
                "Session summary (recent):\n{}",
                self.resident.state_summary
            ));
        }

        if !self.resident.recent_turns.is_empty() {
            let body = format_turns(&self.resident.recent_turns, cfg.max_chars_per_side);
            parts.push(format!("Recent turns:\n{body}"));
        }

        if let Some(dyn_chunk) = &self.dynamic {
            let mut block = format!(
                "Returned topic context [chunk {}]: {}",
                dyn_chunk.chunk_index, dyn_chunk.summary
            );
            if !dyn_chunk.turns.is_empty() {
                block.push('\n');
                block.push_str(&format_turns(&dyn_chunk.turns, cfg.max_chars_per_side));
            }
            parts.push(block);
        }

        if parts.is_empty() {
            return None;
        }
        Some(format!(
            "You are loco-bot, a local on-device assistant.\n\n{}",
            parts.join("\n\n")
        ))
    }
}

/// Assemble resident (+ optional dynamic) context for this turn.
pub fn compile(
    memory: &SessionMemory,
    switch: TopicSwitch,
    cfg: &CompilerConfig,
) -> CompiledContext {
    let window = cfg.recent_turn_window.min(memory.recent_n);
    let recent_turns = if window == 0 {
        Vec::new()
    } else {
        let start = memory.turns.len().saturating_sub(window);
        memory.turns[start..].to_vec()
    };

    let active_topic = memory
        .current_chunk
        .and_then(|i| memory.chunks.get(i))
        .map(|c| c.summary.clone())
        .filter(|s| !s.is_empty());

    let state_summary = if memory.summary.is_empty() && !memory.turns.is_empty() {
        simple_summary(&memory.turns, memory.recent_n, cfg.max_chars_per_side)
    } else {
        memory.summary.clone()
    };

    let resident = ResidentContext {
        state_summary,
        active_topic,
        recent_turns,
    };

    let dynamic = match switch {
        TopicSwitch::Return { chunk_index } => memory.chunks.get(chunk_index).map(|chunk| {
            let slice = memory
                .turns
                .get(chunk.turn_start..chunk.turn_end)
                .unwrap_or(&[]);
            let start = slice.len().saturating_sub(cfg.max_dynamic_turns);
            DynamicChunk {
                chunk_index,
                summary: chunk.summary.clone(),
                turns: slice[start..].to_vec(),
            }
        }),
        TopicSwitch::Continue | TopicSwitch::New => None,
    };

    CompiledContext { resident, dynamic }
}

fn format_turns(turns: &[Turn], max_chars: usize) -> String {
    turns
        .iter()
        .enumerate()
        .map(|(i, t)| {
            format!(
                "{}. user: {}\n   assistant: {}",
                i + 1,
                truncate(&t.user, max_chars),
                truncate(&t.assistant, max_chars)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn truncate(s: &str, max: usize) -> String {
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i >= max {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_with_two_topics() -> SessionMemory {
        let mut mem = SessionMemory::new(8);
        mem.open_chunk("subway", vec![1.0, 0.0]);
        mem.append("地下鉄の話", "都市の動脈");
        mem.append("切符は？", "ICカード");
        mem.open_chunk("cooking", vec![0.0, 1.0]);
        mem.append("カレーの作り方", "ルーを炒める");
        mem
    }

    #[test]
    fn continue_is_resident_only() {
        let mem = mem_with_two_topics();
        let ctx = compile(&mem, TopicSwitch::Continue, &CompilerConfig::default());
        assert!(ctx.dynamic.is_none());
        assert_eq!(ctx.resident.active_topic.as_deref(), Some("cooking"));
        assert!(!ctx.resident.state_summary.is_empty());
        assert!(!ctx.resident.recent_turns.is_empty());
        let notes = ctx.render_notes(&CompilerConfig::default()).unwrap();
        assert!(notes.contains("Active topic: cooking"));
        assert!(!notes.contains("Returned topic"));
    }

    #[test]
    fn return_adds_dynamic_chunk_turns() {
        let mem = mem_with_two_topics();
        let ctx = compile(
            &mem,
            TopicSwitch::Return { chunk_index: 0 },
            &CompilerConfig::default(),
        );
        let dyn_chunk = ctx.dynamic.as_ref().expect("dynamic");
        assert_eq!(dyn_chunk.chunk_index, 0);
        assert_eq!(dyn_chunk.summary, "subway");
        assert_eq!(dyn_chunk.turns.len(), 2);
        assert!(dyn_chunk.turns[0].user.contains("地下鉄"));
        let notes = ctx.render_notes(&CompilerConfig::default()).unwrap();
        assert!(notes.contains("Returned topic context [chunk 0]"));
        assert!(notes.contains("ICカード"));
    }

    #[test]
    fn empty_memory_renders_none() {
        let mem = SessionMemory::default();
        let ctx = compile(&mem, TopicSwitch::New, &CompilerConfig::default());
        assert!(ctx.is_empty());
        assert!(ctx.render_notes(&CompilerConfig::default()).is_none());
    }

    #[test]
    fn dynamic_turn_cap_keeps_tail() {
        let mut mem = SessionMemory::new(8);
        mem.open_chunk("long", vec![1.0]);
        for i in 0..5 {
            mem.append(format!("u{i}"), format!("a{i}"));
        }
        let cfg = CompilerConfig {
            max_dynamic_turns: 2,
            ..CompilerConfig::default()
        };
        let ctx = compile(&mem, TopicSwitch::Return { chunk_index: 0 }, &cfg);
        let turns = ctx.dynamic.unwrap().turns;
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].user, "u3");
        assert_eq!(turns[1].user, "u4");
    }
}
