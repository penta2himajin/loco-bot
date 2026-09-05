//! Short / deictic query expansion from recent user turns.

use crate::deixis::{classify_deixis, DeixisKind};

/// Expand the embedding query based on deixis class.
///
/// - [`DeixisKind::ContinueHint`]: prepend previous user turn.
/// - Return deixis (named or not): keep clean (no prior pollution).
/// - Plain topical: unchanged.
pub fn expand_query(user: &str, previous_user: Option<&str>) -> String {
    let trimmed = user.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    match classify_deixis(trimmed) {
        DeixisKind::ContinueHint => match previous_user {
            Some(prev) if !prev.trim().is_empty() => {
                format!("{}\n{}", prev.trim(), trimmed)
            }
            _ => trimmed.to_string(),
        },
        DeixisKind::ReturnNamed | DeixisKind::ReturnUnspecified | DeixisKind::Plain => {
            trimmed.to_string()
        }
    }
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
        let q2 = "さっきの話";
        assert_eq!(expand_query(q2, Some("カレーの作り方を一言")), q2);
    }
}
