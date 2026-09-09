//! Built-in tool implementations (clock, notes, session stats).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("unknown tool `{0}`")]
    Unknown(String),
    #[error("missing argument `{0}`")]
    MissingArg(String),
    #[error("path escapes sandbox: {0}")]
    PathEscape(String),
    #[error("fs root not configured")]
    NoFsRoot,
    #[error("not configured: {0}")]
    NotConfigured(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Executes built-in tools against paths under the loco cache.
#[derive(Debug, Clone)]
pub struct ToolHost {
    notes_path: PathBuf,
    memory_path: PathBuf,
    /// Optional sandbox root for fs_* tools. Relative tool paths resolve here.
    fs_root: Option<PathBuf>,
}

impl ToolHost {
    pub fn new(notes_path: impl Into<PathBuf>, memory_path: impl Into<PathBuf>) -> Self {
        Self::with_fs_root(notes_path, memory_path, None)
    }

    pub fn with_fs_root(
        notes_path: impl Into<PathBuf>,
        memory_path: impl Into<PathBuf>,
        fs_root: Option<PathBuf>,
    ) -> Self {
        Self {
            notes_path: notes_path.into(),
            memory_path: memory_path.into(),
            fs_root,
        }
    }

    pub fn execute(&self, name: &str, args: &Map<String, Value>) -> Result<Value, ToolError> {
        match name {
            "get_current_time" => Ok(self.get_current_time()),
            "note_write" => self.note_write(args),
            "note_read" => self.note_read(args),
            "session_stats" => Ok(self.session_stats()),
            "fs_list" => self.fs_list(args),
            "fs_move" | "fs_rename" => self.fs_relocate(args),
            "web_search" => Err(ToolError::NotConfigured(
                "web_search provider (set LOCO_WEB_SEARCH later)".into(),
            )),
            "mail_list" => Err(ToolError::NotConfigured("mail OAuth not wired yet".into())),
            "drive_list" => Err(ToolError::NotConfigured("drive OAuth not wired yet".into())),
            other => Err(ToolError::Unknown(other.to_string())),
        }
    }

    fn get_current_time(&self) -> Value {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // UTC ISO-8601 without chrono dependency (good enough for P5).
        let (y, mo, d, h, mi, s) = civil_from_unix(secs);
        json!({
            "utc": format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z"),
            "unix_secs": secs,
        })
    }

    fn note_write(&self, args: &Map<String, Value>) -> Result<Value, ToolError> {
        let key = arg_str(args, "key")?;
        let text = arg_str(args, "text")?;
        let mut notes = load_notes(&self.notes_path)?;
        notes.insert(key.to_string(), text.to_string());
        save_notes(&self.notes_path, &notes)?;
        Ok(json!({ "ok": true, "key": key, "bytes": text.len() }))
    }

    fn note_read(&self, args: &Map<String, Value>) -> Result<Value, ToolError> {
        let key = arg_str(args, "key")?;
        let notes = load_notes(&self.notes_path)?;
        Ok(json!({
            "key": key,
            "text": notes.get(key).cloned().unwrap_or_default(),
            "found": notes.contains_key(key),
        }))
    }

    fn session_stats(&self) -> Value {
        match fs::read_to_string(&self.memory_path) {
            Ok(raw) => match serde_json::from_str::<Value>(&raw) {
                Ok(v) => {
                    let turns = v
                        .get("turns")
                        .and_then(|t| t.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0);
                    let chunks = v
                        .get("chunks")
                        .and_then(|t| t.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0);
                    json!({
                        "turns": turns,
                        "chunks": chunks,
                        "path": self.memory_path.display().to_string(),
                    })
                }
                Err(err) => json!({ "error": err.to_string() }),
            },
            Err(_) => json!({
                "turns": 0,
                "chunks": 0,
                "path": self.memory_path.display().to_string(),
                "empty": true,
            }),
        }
    }

    fn fs_list(&self, args: &Map<String, Value>) -> Result<Value, ToolError> {
        let rel = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let dir = self.resolve_under_root(rel)?;
        let mut entries = Vec::new();
        for ent in fs::read_dir(&dir)? {
            let ent = ent?;
            let meta = ent.metadata()?;
            entries.push(json!({
                "name": ent.file_name().to_string_lossy(),
                "is_dir": meta.is_dir(),
                "bytes": if meta.is_file() { Some(meta.len()) } else { None },
            }));
        }
        entries.sort_by(|a, b| {
            a["name"]
                .as_str()
                .unwrap_or("")
                .cmp(b["name"].as_str().unwrap_or(""))
        });
        Ok(json!({
            "path": rel,
            "entries": entries,
        }))
    }

    fn fs_relocate(&self, args: &Map<String, Value>) -> Result<Value, ToolError> {
        let from = arg_str(args, "from")?;
        let to = arg_str(args, "to")?;
        let dry_run = args
            .get("dry_run")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let src = self.resolve_under_root(from)?;
        let dst = self.resolve_under_root(to)?;
        if !src.exists() {
            return Err(ToolError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{} not found", src.display()),
            )));
        }
        if dry_run {
            return Ok(json!({
                "ok": true,
                "dry_run": true,
                "from": from,
                "to": to,
            }));
        }
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::rename(&src, &dst)?;
        Ok(json!({
            "ok": true,
            "dry_run": false,
            "from": from,
            "to": to,
        }))
    }

    fn resolve_under_root(&self, relative: &str) -> Result<PathBuf, ToolError> {
        use std::path::{Component, Path};

        let root = self.fs_root.as_ref().ok_or(ToolError::NoFsRoot)?;
        let root = root.canonicalize().map_err(ToolError::Io)?;
        let rel = Path::new(relative);
        if rel.is_absolute() {
            return Err(ToolError::PathEscape(relative.to_string()));
        }

        let mut parts: Vec<std::ffi::OsString> = Vec::new();
        for c in rel.components() {
            match c {
                Component::CurDir => {}
                Component::Normal(s) => parts.push(s.to_owned()),
                Component::ParentDir => {
                    if parts.pop().is_none() {
                        return Err(ToolError::PathEscape(relative.to_string()));
                    }
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(ToolError::PathEscape(relative.to_string()));
                }
            }
        }

        let mut out = root.clone();
        for p in &parts {
            out.push(p);
        }
        Ok(out)
    }
}

fn arg_str<'a>(args: &'a Map<String, Value>, key: &str) -> Result<&'a str, ToolError> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::MissingArg(key.to_string()))
}

fn load_notes(path: &Path) -> Result<BTreeMap<String, String>, ToolError> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let raw = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw)?)
}

fn save_notes(path: &Path, notes: &BTreeMap<String, String>) -> Result<(), ToolError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(notes)?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, raw)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Convert unix seconds to naive UTC civil components (proleptic Gregorian).
fn civil_from_unix(secs: u64) -> (i32, u32, u32, u32, u32, u32) {
    let s = (secs % 86400) as u32;
    let h = s / 3600;
    let mi = (s % 3600) / 60;
    let sec = s % 60;
    let mut days = (secs / 86400) as i64;
    // 1970-01-01 was a Thursday; algorithm from civil_from_days (Howard Hinnant).
    days += 719468; // shift to civil epoch
    let era = if days >= 0 { days } else { days - 146096 } / 146097;
    let doe = (days - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64 + era * 400) as i32;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d, h, mi, sec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn note_round_trip() {
        let dir = tempdir().unwrap();
        let host = ToolHost::new(dir.path().join("notes.json"), dir.path().join("mem.json"));
        let mut args = Map::new();
        args.insert("key".into(), json!("todo"));
        args.insert("text".into(), json!("buy milk"));
        host.execute("note_write", &args).unwrap();
        let read = host.execute("note_read", &args).unwrap();
        assert_eq!(read["text"], "buy milk");
        assert_eq!(read["found"], true);
    }

    #[test]
    fn time_has_utc_field() {
        let dir = tempdir().unwrap();
        let host = ToolHost::new(dir.path().join("n.json"), dir.path().join("m.json"));
        let v = host.execute("get_current_time", &Map::new()).unwrap();
        assert!(v["utc"].as_str().unwrap().ends_with('Z'));
    }

    #[test]
    fn civil_epoch_smoke() {
        let (y, m, d, h, mi, s) = civil_from_unix(0);
        assert_eq!((y, m, d, h, mi, s), (1970, 1, 1, 0, 0, 0));
    }

    #[test]
    fn fs_list_stays_inside_root() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("sandbox");
        fs::create_dir_all(root.join("sub")).unwrap();
        fs::write(root.join("a.txt"), b"a").unwrap();
        fs::write(root.join("sub/b.txt"), b"b").unwrap();
        let host = ToolHost::with_fs_root(
            dir.path().join("n.json"),
            dir.path().join("m.json"),
            Some(root.clone()),
        );
        let mut args = Map::new();
        args.insert("path".into(), json!("."));
        let v = host.execute("fs_list", &args).unwrap();
        let names = v["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["name"].as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(names.contains(&"a.txt".into()));
        assert!(names.contains(&"sub".into()));
    }

    #[test]
    fn fs_rejects_escape_from_root() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("sandbox");
        fs::create_dir_all(&root).unwrap();
        let host = ToolHost::with_fs_root(
            dir.path().join("n.json"),
            dir.path().join("m.json"),
            Some(root),
        );
        let mut args = Map::new();
        args.insert("path".into(), json!("../outside"));
        let err = host.execute("fs_list", &args).unwrap_err();
        assert!(matches!(err, ToolError::PathEscape(_)));
    }

    #[test]
    fn fs_move_dry_run_does_not_touch_disk() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("sandbox");
        fs::create_dir_all(&root).unwrap();
        let src = root.join("old.txt");
        fs::write(&src, b"x").unwrap();
        let host = ToolHost::with_fs_root(
            dir.path().join("n.json"),
            dir.path().join("m.json"),
            Some(root.clone()),
        );
        let mut args = Map::new();
        args.insert("from".into(), json!("old.txt"));
        args.insert("to".into(), json!("new.txt"));
        args.insert("dry_run".into(), json!(true));
        let v = host.execute("fs_move", &args).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["dry_run"], true);
        assert!(src.exists());
        assert!(!root.join("new.txt").exists());
    }

    #[test]
    fn fs_rename_moves_file() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("sandbox");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("old.txt"), b"x").unwrap();
        let host = ToolHost::with_fs_root(
            dir.path().join("n.json"),
            dir.path().join("m.json"),
            Some(root.clone()),
        );
        let mut args = Map::new();
        args.insert("from".into(), json!("old.txt"));
        args.insert("to".into(), json!("new.txt"));
        let v = host.execute("fs_rename", &args).unwrap();
        assert_eq!(v["ok"], true);
        assert!(!root.join("old.txt").exists());
        assert!(root.join("new.txt").exists());
    }

    #[test]
    fn web_search_requires_provider() {
        let dir = tempdir().unwrap();
        let host = ToolHost::new(dir.path().join("n.json"), dir.path().join("m.json"));
        let err = host.execute("web_search", &Map::new()).unwrap_err();
        assert!(matches!(err, ToolError::NotConfigured(_)));
    }
}
