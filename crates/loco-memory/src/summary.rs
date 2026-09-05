//! Rolling summary without an LLM call.

use crate::turn::Turn;

/// Build a short plain-text summary from the most recent turns.
///
/// Truncates each side so the preamble stays small enough for a system prompt.
pub fn simple_summary(turns: &[Turn], max_turns: usize, max_chars_per_side: usize) -> String {
    if turns.is_empty() {
        return String::new();
    }
    let start = turns.len().saturating_sub(max_turns);
    let mut lines = Vec::new();
    for (i, turn) in turns[start..].iter().enumerate() {
        lines.push(format!(
            "{}. user: {}\n   assistant: {}",
            start + i + 1,
            truncate(&turn.user, max_chars_per_side),
            truncate(&turn.assistant, max_chars_per_side)
        ));
    }
    lines.join("\n")
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

    #[test]
    fn empty_turns_yield_empty_summary() {
        assert!(simple_summary(&[], 4, 40).is_empty());
    }

    #[test]
    fn keeps_only_recent_turns_and_truncates() {
        let turns: Vec<_> = (0..5)
            .map(|i| Turn::new(format!("user-{i}-{}", "x".repeat(80)), format!("asst-{i}")))
            .collect();
        let s = simple_summary(&turns, 2, 10);
        assert!(s.contains("4."));
        assert!(s.contains("5."));
        assert!(!s.contains("1."));
        assert!(s.contains('…'));
    }
}
