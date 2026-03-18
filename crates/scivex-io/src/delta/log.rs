//! Delta Lake transaction log parsing.
//!
//! Parses the `_delta_log/` directory's newline-delimited JSON files to
//! reconstruct the set of active Parquet data files at a given table version.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::error::{IoError, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single action from a Delta transaction log entry.
#[derive(Debug, Clone, PartialEq)]
pub enum DeltaAction {
    /// A file was added to the table.
    Add { path: String },
    /// A file was removed from the table.
    Remove { path: String },
    /// Table metadata was recorded (we only track that it exists).
    MetaData,
}

// ---------------------------------------------------------------------------
// Minimal JSON helpers
// ---------------------------------------------------------------------------

/// Skip whitespace and return the remaining slice.
fn skip_ws(s: &str) -> &str {
    s.trim_start()
}

/// Extract a JSON string value.
/// Input: slice starting at the opening `"`. Returns `(value, rest)` where
/// `rest` begins after the closing `"`.
fn extract_json_string(input: &str) -> Option<(String, &str)> {
    if !input.starts_with('"') {
        return None;
    }
    let s = &input[1..]; // skip opening quote
    let mut result = String::new();
    let mut i = 0;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                return Some((result, &s[i + 1..]));
            }
            b'\\' => {
                i += 1;
                if i >= bytes.len() {
                    return None;
                }
                match bytes[i] {
                    b'"' => result.push('"'),
                    b'\\' => result.push('\\'),
                    b'/' => result.push('/'),
                    b'n' => result.push('\n'),
                    b't' => result.push('\t'),
                    b'r' => result.push('\r'),
                    other => {
                        result.push('\\');
                        result.push(other as char);
                    }
                }
            }
            other => result.push(other as char),
        }
        i += 1;
    }
    None // unterminated string
}

/// Find the value of a string-typed key inside a JSON object.
/// `input` should start at or before `{`. Only looks at the top level of
/// the first object encountered.
fn find_string_value(input: &str, key: &str) -> Option<String> {
    // Locate the key as `"key"` followed by `:` then a `"value"`.
    let search = format!("\"{key}\"");
    let mut remaining = input;
    loop {
        let pos = remaining.find(&search)?;
        remaining = &remaining[pos + search.len()..];
        let remaining_trimmed = skip_ws(remaining);
        if let Some(after_colon) = remaining_trimmed.strip_prefix(':') {
            let after_colon = skip_ws(after_colon);
            if after_colon.starts_with('"') {
                let (val, _) = extract_json_string(after_colon)?;
                return Some(val);
            }
            // Value is not a string — skip and keep searching for another
            // occurrence of the same key (shouldn't happen in well-formed
            // Delta logs, but be safe).
        }
    }
}

/// Check whether a JSON line starts with `{"<action_key>":{...}}`.
/// Returns the inner object substring (everything between the outer `{` and
/// the matching `}`).
fn extract_action_object<'a>(line: &'a str, action: &str) -> Option<&'a str> {
    let trimmed = skip_ws(line);
    let inner = skip_ws(trimmed.strip_prefix('{')?);
    let search = format!("\"{action}\"");
    if !inner.starts_with(&search) {
        return None;
    }
    let after_key = skip_ws(&inner[search.len()..]);
    let after_colon = skip_ws(after_key.strip_prefix(':')?);
    // The value is an object — find the matching `}`.
    if !after_colon.starts_with('{') {
        return None;
    }
    // Find matching closing brace (handle nested braces).
    let mut depth = 0u32;
    let mut in_string = false;
    let mut prev_backslash = false;
    let bytes = after_colon.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if b == b'\\' && !prev_backslash {
                prev_backslash = true;
                continue;
            }
            if b == b'"' && !prev_backslash {
                in_string = false;
            }
            prev_backslash = false;
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&after_colon[..=i]);
                }
            }
            _ => {}
        }
        prev_backslash = false;
    }
    None
}

// ---------------------------------------------------------------------------
// Action parsing
// ---------------------------------------------------------------------------

/// Parse a single line from a Delta transaction log into a [`DeltaAction`].
///
/// Returns `None` for lines that don't match `add`, `remove`, or `metaData`
/// actions (e.g., `commitInfo`, `protocol`).
pub fn parse_log_line(line: &str) -> Option<DeltaAction> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Try "add"
    if let Some(obj) = extract_action_object(trimmed, "add") {
        let path = find_string_value(obj, "path")?;
        return Some(DeltaAction::Add { path });
    }

    // Try "remove"
    if let Some(obj) = extract_action_object(trimmed, "remove") {
        let path = find_string_value(obj, "path")?;
        return Some(DeltaAction::Remove { path });
    }

    // Try "metaData"
    if extract_action_object(trimmed, "metaData").is_some() {
        return Some(DeltaAction::MetaData);
    }

    None
}

// ---------------------------------------------------------------------------
// Log file enumeration
// ---------------------------------------------------------------------------

/// Return the Delta log directory path, verifying it exists.
pub fn log_dir(table_path: &Path) -> Result<std::path::PathBuf> {
    let dir = table_path.join("_delta_log");
    if !dir.is_dir() {
        return Err(IoError::FormatError(format!(
            "Delta log directory not found: {}",
            dir.display()
        )));
    }
    Ok(dir)
}

/// List all JSON log files in `_delta_log/`, sorted by version number.
/// Returns `(version, path)` pairs.
pub fn list_log_files(delta_log_dir: &Path) -> Result<Vec<(u64, std::path::PathBuf)>> {
    let mut entries: Vec<(u64, std::path::PathBuf)> = Vec::new();
    for entry in fs::read_dir(delta_log_dir)? {
        let entry = entry?;
        let path = entry.path();
        let is_json = path.extension().is_some_and(|e| e == "json");
        if !is_json {
            continue;
        }
        let version = path
            .file_stem()
            .and_then(|s| s.to_string_lossy().parse::<u64>().ok());
        if let Some(v) = version {
            entries.push((v, path));
        }
    }
    entries.sort_by_key(|(v, _)| *v);
    Ok(entries)
}

// ---------------------------------------------------------------------------
// Snapshot replay
// ---------------------------------------------------------------------------

/// Replay transaction logs up to (and including) `max_version` and return the
/// set of active file paths.
pub fn replay_logs(delta_log_dir: &Path, max_version: Option<u64>) -> Result<(u64, Vec<String>)> {
    let log_files = list_log_files(delta_log_dir)?;
    if log_files.is_empty() {
        return Err(IoError::FormatError(
            "no Delta log files found in _delta_log/".to_string(),
        ));
    }

    let mut active_files: HashSet<String> = HashSet::new();
    let mut latest_version: u64 = 0;

    for (version, path) in &log_files {
        if max_version.is_some_and(|max| *version > max) {
            break;
        }
        latest_version = *version;
        let content = fs::read_to_string(path)?;
        for line in content.lines() {
            match parse_log_line(line) {
                Some(DeltaAction::Add { path }) => {
                    active_files.insert(path);
                }
                Some(DeltaAction::Remove { path }) => {
                    active_files.remove(&path);
                }
                _ => {}
            }
        }
    }

    let mut files: Vec<String> = active_files.into_iter().collect();
    files.sort();
    Ok((latest_version, files))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_parse_add_action() {
        let line = r#"{"add":{"path":"part-00000.parquet","size":1234,"modificationTime":1234567890,"dataChange":true}}"#;
        let action = parse_log_line(line).unwrap();
        assert_eq!(
            action,
            DeltaAction::Add {
                path: "part-00000.parquet".to_string()
            }
        );
    }

    #[test]
    fn test_parse_remove_action() {
        let line = r#"{"remove":{"path":"part-00001.parquet","deletionTimestamp":1234567891,"dataChange":true}}"#;
        let action = parse_log_line(line).unwrap();
        assert_eq!(
            action,
            DeltaAction::Remove {
                path: "part-00001.parquet".to_string()
            }
        );
    }

    #[test]
    fn test_parse_metadata_action() {
        let line = r#"{"metaData":{"id":"abc-123","format":{"provider":"parquet"},"schemaString":"{}","partitionColumns":[]}}"#;
        let action = parse_log_line(line).unwrap();
        assert_eq!(action, DeltaAction::MetaData);
    }

    #[test]
    fn test_parse_unknown_action() {
        let line = r#"{"commitInfo":{"timestamp":1234567890,"operation":"WRITE"}}"#;
        assert!(parse_log_line(line).is_none());
    }

    #[test]
    fn test_parse_empty_line() {
        assert!(parse_log_line("").is_none());
        assert!(parse_log_line("   ").is_none());
    }

    #[test]
    fn test_extract_json_string_basic() {
        let (val, rest) = extract_json_string(r#""hello world""#).unwrap();
        assert_eq!(val, "hello world");
        assert!(rest.is_empty());
    }

    #[test]
    fn test_extract_json_string_escaped() {
        let (val, _) = extract_json_string(r#""path\/to\/file""#).unwrap();
        assert_eq!(val, "path/to/file");
    }

    #[test]
    fn test_snapshot_replay_add_then_remove() {
        let dir = std::env::temp_dir().join("scivex_delta_replay_test");
        let log_dir_path = dir.join("_delta_log");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&log_dir_path).unwrap();

        // Version 0: add two files
        fs::write(
            log_dir_path.join("00000000000000000000.json"),
            r#"{"add":{"path":"part-00000.parquet","size":100,"modificationTime":1000,"dataChange":true}}
{"add":{"path":"part-00001.parquet","size":200,"modificationTime":1000,"dataChange":true}}
"#,
        )
        .unwrap();

        // Version 1: remove one, add another
        fs::write(
            log_dir_path.join("00000000000000000001.json"),
            r#"{"remove":{"path":"part-00000.parquet","deletionTimestamp":2000,"dataChange":true}}
{"add":{"path":"part-00002.parquet","size":300,"modificationTime":2000,"dataChange":true}}
"#,
        )
        .unwrap();

        // Replay all
        let (version, files) = replay_logs(&log_dir_path, None).unwrap();
        assert_eq!(version, 1);
        assert_eq!(files, vec!["part-00001.parquet", "part-00002.parquet"]);

        // Replay only version 0
        let (version, files) = replay_logs(&log_dir_path, Some(0)).unwrap();
        assert_eq!(version, 0);
        assert_eq!(files, vec!["part-00000.parquet", "part-00001.parquet"]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_missing_log_dir() {
        let dir = std::env::temp_dir().join("scivex_delta_missing_log_dir");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // No _delta_log directory
        let result = log_dir(&dir);
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_empty_log_dir() {
        let dir = std::env::temp_dir().join("scivex_delta_empty_log");
        let log_dir_path = dir.join("_delta_log");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&log_dir_path).unwrap();
        let result = replay_logs(&log_dir_path, None);
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_path_with_special_chars() {
        let line = r#"{"add":{"path":"year=2024/month=01/part-00000.parquet","size":500,"modificationTime":3000,"dataChange":true}}"#;
        let action = parse_log_line(line).unwrap();
        assert_eq!(
            action,
            DeltaAction::Add {
                path: "year=2024/month=01/part-00000.parquet".to_string()
            }
        );
    }
}
