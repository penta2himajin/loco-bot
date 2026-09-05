//! Rule-based deixis classification for S1 query shaping / topic resolve.

/// How an utterance uses (or does not use) underspecified reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeixisKind {
    /// Normal topical input — embed as-is.
    Plain,
    /// Continues the current topic via deixis ("それ", "詳しく", …).
    ContinueHint,
    /// Returns to a *named* past topic ("さっきの地下鉄の話").
    ReturnNamed,
    /// Returns without naming the topic ("さっきの話", "前の話に戻って").
    ReturnUnspecified,
}

/// Classify deixis for topic resolution (lexicon only — no embedding).
pub fn classify_deixis(user: &str) -> DeixisKind {
    let trimmed = user.trim();
    if trimmed.is_empty() {
        return DeixisKind::Plain;
    }
    if looks_topic_return(trimmed) {
        if has_topic_name_cue(trimmed) {
            DeixisKind::ReturnNamed
        } else {
            DeixisKind::ReturnUnspecified
        }
    } else if looks_continue_deixis(trimmed) || looks_bare_followup(trimmed) {
        DeixisKind::ContinueHint
    } else {
        DeixisKind::Plain
    }
}

pub(crate) fn looks_topic_return(s: &str) -> bool {
    let lower = s.to_lowercase();
    const MARKERS: &[&str] = &[
        "戻って",
        "さっきの",
        "前の話",
        "前のやつ",
        "go back",
        "back to",
        "earlier",
        "回到",
        "刚才的话",
        "之前的话题",
        "上一个话题",
    ];
    MARKERS.iter().any(|m| lower.contains(m))
}

/// Strip return scaffolding; leftover content is treated as a topic name cue.
pub(crate) fn topic_cue_residue(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut t = lower;
    const STRIP: &[&str] = &[
        "さっきの",
        "前の話",
        "前のやつ",
        "前の",
        "話に戻って",
        "に戻って",
        "戻って",
        "go back to",
        "go back",
        "back to",
        "earlier",
        "なんだけど",
        "だけど",
        "けれど",
        "の話",
        "話",
        "について",
        "教えて",
        "回到刚才的",
        "回到刚才",
        "回到",
        "刚才的话",
        "刚才的",
        "刚才",
        "之前的话题",
        "上一个话题",
        "的话题",
        "话题",
    ];
    for m in STRIP {
        t = t.replace(m, "");
    }
    t.chars()
        .filter(|c| !c.is_whitespace() && !"、。？?!,.：:".contains(*c))
        .collect()
}

/// True when residue looks like a topic name, not leftover particles.
pub(crate) fn has_topic_name_cue(s: &str) -> bool {
    let r = topic_cue_residue(s);
    if r.chars().count() < 2 {
        return false;
    }
    const FILLERS: &[&str] = &["けど", "なの", "です", "ます", "して", "どう", "なに", "何"];
    if FILLERS.iter().any(|f| r == *f) {
        return false;
    }
    r.chars().any(|c| {
        c.is_ascii_alphanumeric()
            || ('一'..='龯').contains(&c)
            || ('ァ'..='ヶ').contains(&c)
            || ('ぁ'..='ん').contains(&c)
    })
}

fn looks_continue_deixis(s: &str) -> bool {
    let lower = s.to_lowercase();
    const MARKERS: &[&str] = &[
        "それ",
        "これ",
        "あれ",
        "その",
        "この",
        "あの",
        "それについて",
        "これについて",
        "that",
        "this",
        "it ",
        "it's",
        "what about",
        "and the",
        "那个",
        "这个",
        "关于这个",
        "关于那个",
    ];
    MARKERS.iter().any(|m| lower.contains(m))
}

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
        "ok",
        "yes",
        "no",
        "为什么",
        "详细一点",
        "再说详细",
        "继续",
    ];
    let lower = s.to_lowercase();
    BARE.iter().any(|m| lower.contains(m)) || chars <= 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_return() {
        assert_eq!(
            classify_deixis("さっきの地下鉄の話に戻って、切符はどう買う？"),
            DeixisKind::ReturnNamed
        );
    }

    #[test]
    fn unspecified_return() {
        assert_eq!(classify_deixis("さっきの話"), DeixisKind::ReturnUnspecified);
        assert_eq!(
            classify_deixis("さっきの話に戻って"),
            DeixisKind::ReturnUnspecified
        );
        assert_eq!(
            classify_deixis("前の話なんだけど"),
            DeixisKind::ReturnUnspecified
        );
    }

    #[test]
    fn continue_hint() {
        assert_eq!(classify_deixis("それについて"), DeixisKind::ContinueHint);
        assert_eq!(classify_deixis("なぜ？"), DeixisKind::ContinueHint);
        assert_eq!(classify_deixis("もっと詳しく"), DeixisKind::ContinueHint);
    }

    #[test]
    fn plain_topical() {
        assert_eq!(classify_deixis("カレーの作り方を一言"), DeixisKind::Plain);
        assert_eq!(classify_deixis("地下鉄について一言"), DeixisKind::Plain);
    }

    #[test]
    fn en_and_zh_return_and_continue() {
        assert_eq!(
            classify_deixis("go back to the subway topic"),
            DeixisKind::ReturnNamed
        );
        assert_eq!(classify_deixis("earlier"), DeixisKind::ReturnUnspecified);
        assert_eq!(classify_deixis("why?"), DeixisKind::ContinueHint);
        assert_eq!(
            classify_deixis("回到刚才地铁的话题"),
            DeixisKind::ReturnNamed
        );
        assert_eq!(
            classify_deixis("回到刚才的话题"),
            DeixisKind::ReturnUnspecified
        );
        assert_eq!(classify_deixis("为什么？"), DeixisKind::ContinueHint);
        assert_eq!(classify_deixis("再说详细一点"), DeixisKind::ContinueHint);
    }
}
