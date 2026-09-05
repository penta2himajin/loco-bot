//! Short / deictic query expansion from recent user turns.

/// Expand a deictic / underspecified utterance with the previous user message
/// so the embedding is less ambiguous (chatstream S1 lesson).
///
/// Length alone is not enough: short but topical lines (e.g. cooking) must not
/// inherit the prior topic’s embedding.
pub fn expand_query(user: &str, previous_user: Option<&str>) -> String {
    let trimmed = user.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    match previous_user {
        Some(prev) if needs_expansion(trimmed) && !prev.trim().is_empty() => {
            format!("{}\n{}", prev.trim(), trimmed)
        }
        _ => trimmed.to_string(),
    }
}

fn needs_expansion(s: &str) -> bool {
    // Topic-return cues must stay clean so S1 can match the named past chunk.
    if looks_topic_return(s) {
        return false;
    }
    looks_deictic(s) || looks_bare_followup(s)
}

fn looks_topic_return(s: &str) -> bool {
    let lower = s.to_lowercase();
    const MARKERS: &[&str] = &["戻って", "さっきの", "前の話", "go back", "back to"];
    MARKERS.iter().any(|m| lower.contains(m))
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
        "なぜ",
        "なんで",
        "どうして",
        "that",
        "this",
        "it ",
        "it's",
        "what about",
        "and the",
    ];
    MARKERS.iter().any(|m| lower.contains(m))
}

/// Very short follow-ups with little topical content (e.g. "なぜ？", "more").
fn looks_bare_followup(s: &str) -> bool {
    let chars = s.chars().count();
    if chars > 12 {
        return false;
    }
    const BARE: &[&str] = &[
        "なぜ",
        "なんで",
        "どうして",
        "もっと",
        "詳しく",
        "続き",
        "more",
        "why",
        "how",
        "and",
        "ok",
        "yes",
        "no",
    ];
    let lower = s.to_lowercase();
    BARE.iter().any(|m| lower.contains(m)) || chars <= 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topical_short_query_unchanged() {
        let q = "カレーの作り方を一言";
        assert_eq!(expand_query(q, Some("地下鉄について一言")), q);
    }

    #[test]
    fn long_query_unchanged() {
        let q = "Tell me about the history of the Tokyo subway system in detail";
        assert_eq!(expand_query(q, Some("prior")), q);
    }

    #[test]
    fn bare_why_gets_prior() {
        let out = expand_query("なぜ？", Some("地下鉄の話"));
        assert!(out.contains("地下鉄"));
        assert!(out.contains("なぜ"));
    }

    #[test]
    fn deictic_expands() {
        let out = expand_query("それについて詳しく", Some("Gemma 4 E4B"));
        assert!(out.starts_with("Gemma"));
    }

    #[test]
    fn return_phrase_stays_clean() {
        let q = "さっきの地下鉄の話に戻って";
        assert_eq!(expand_query(q, Some("カレーの作り方を一言")), q);
    }
}
