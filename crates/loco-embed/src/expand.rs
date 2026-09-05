//! Short / deictic query expansion from recent user turns.

/// Expand a short user utterance with the previous user message so the
/// embedding is less ambiguous (chatstream S1 lesson).
pub fn expand_query(user: &str, previous_user: Option<&str>) -> String {
    let trimmed = user.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let short = trimmed.chars().count() < 24;
    let deictic = looks_deictic(trimmed);
    match previous_user {
        Some(prev) if (short || deictic) && !prev.trim().is_empty() => {
            format!("{}\n{}", prev.trim(), trimmed)
        }
        _ => trimmed.to_string(),
    }
}

fn looks_deictic(s: &str) -> bool {
    let lower = s.to_lowercase();
    const MARKERS: &[&str] = &[
        "それ",
        "これ",
        "あれ",
        "その",
        "この",
        "あの",
        "that",
        "this",
        "it ",
        "it's",
        "what about",
        "and the",
    ];
    MARKERS.iter().any(|m| lower.contains(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_query_unchanged() {
        let q = "Tell me about the history of the Tokyo subway system in detail";
        assert_eq!(expand_query(q, Some("prior")), q);
    }

    #[test]
    fn short_query_gets_prior() {
        let out = expand_query("なぜ？", Some("地下鉄の話"));
        assert!(out.contains("地下鉄"));
        assert!(out.contains("なぜ"));
    }

    #[test]
    fn deictic_expands() {
        let out = expand_query("それについて詳しく", Some("Gemma 4 E4B"));
        assert!(out.starts_with("Gemma"));
    }
}
