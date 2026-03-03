//! Type inference and series-building utilities shared between CSV and JSON
//! readers.

use scivex_frame::{AnySeries, Series, StringSeries};

use crate::error::Result;

/// The inferred data type for a column of text values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferredType {
    /// 64-bit signed integer.
    I64,
    /// 64-bit float.
    F64,
    /// Boolean (stored as `Series<u8>`: 0 = false, 1 = true).
    Bool,
    /// String (fallback).
    Str,
}

/// Returns `true` if `s` is a recognised null sentinel.
///
/// Recognised values (case-insensitive where noted):
/// `""`, `"NA"`, `"N/A"`, `"null"`, `"NULL"`, `"None"`, `"NaN"`, `"nan"`,
/// `"."`, `"-"`.
pub fn is_null_sentinel(s: &str) -> bool {
    matches!(
        s,
        "" | "NA"
            | "N/A"
            | "null"
            | "NULL"
            | "None"
            | "NaN"
            | "nan"
            | "."
            | "-"
            | "na"
            | "n/a"
            | "none"
            | "Null"
    )
}

/// Try to parse `s` as an `i64`.
#[inline]
pub fn try_parse_i64(s: &str) -> Option<i64> {
    s.parse::<i64>().ok()
}

/// Try to parse `s` as an `f64`.
#[inline]
pub fn try_parse_f64(s: &str) -> Option<f64> {
    s.parse::<f64>().ok()
}

/// Try to parse `s` as a boolean.
///
/// Recognised truthy values: `"true"`, `"True"`, `"TRUE"`, `"1"`, `"yes"`,
/// `"Yes"`, `"YES"`.
/// Recognised falsy values: `"false"`, `"False"`, `"FALSE"`, `"0"`, `"no"`,
/// `"No"`, `"NO"`.
pub fn try_parse_bool(s: &str) -> Option<bool> {
    match s {
        "true" | "True" | "TRUE" | "1" | "yes" | "Yes" | "YES" => Some(true),
        "false" | "False" | "FALSE" | "0" | "no" | "No" | "NO" => Some(false),
        _ => None,
    }
}

/// Infer the most specific type that can represent all non-null values.
///
/// Priority: `I64` > `F64` > `Bool` > `Str`.
pub fn infer_column_type(values: &[&str]) -> InferredType {
    let non_null: Vec<&str> = values
        .iter()
        .copied()
        .filter(|s| !is_null_sentinel(s))
        .collect();

    if non_null.is_empty() {
        return InferredType::Str;
    }

    // Try I64
    if non_null.iter().all(|s| try_parse_i64(s).is_some()) {
        return InferredType::I64;
    }

    // Try F64
    if non_null.iter().all(|s| try_parse_f64(s).is_some()) {
        return InferredType::F64;
    }

    // Try Bool
    if non_null.iter().all(|s| try_parse_bool(s).is_some()) {
        return InferredType::Bool;
    }

    InferredType::Str
}

/// Build a typed [`AnySeries`] from a vector of string values.
///
/// Null sentinels and parse failures are marked as null in the resulting
/// series.
pub fn build_series_from_strings(
    name: &str,
    values: &[String],
    dtype: InferredType,
) -> Result<Box<dyn AnySeries>> {
    match dtype {
        InferredType::I64 => build_i64_series(name, values),
        InferredType::F64 => build_f64_series(name, values),
        InferredType::Bool => build_bool_series(name, values),
        InferredType::Str => build_string_series(name, values),
    }
}

fn build_i64_series(name: &str, values: &[String]) -> Result<Box<dyn AnySeries>> {
    let mut data = Vec::with_capacity(values.len());
    let mut nulls = Vec::with_capacity(values.len());
    let mut has_nulls = false;

    for v in values {
        if is_null_sentinel(v) {
            data.push(0_i64);
            nulls.push(true);
            has_nulls = true;
        } else if let Some(parsed) = try_parse_i64(v) {
            data.push(parsed);
            nulls.push(false);
        } else {
            data.push(0);
            nulls.push(true);
            has_nulls = true;
        }
    }

    if has_nulls {
        Ok(Box::new(Series::with_nulls(name, data, nulls)?))
    } else {
        Ok(Box::new(Series::new(name, data)))
    }
}

fn build_f64_series(name: &str, values: &[String]) -> Result<Box<dyn AnySeries>> {
    let mut data = Vec::with_capacity(values.len());
    let mut nulls = Vec::with_capacity(values.len());
    let mut has_nulls = false;

    for v in values {
        if is_null_sentinel(v) {
            data.push(0.0_f64);
            nulls.push(true);
            has_nulls = true;
        } else if let Some(parsed) = try_parse_f64(v) {
            data.push(parsed);
            nulls.push(false);
        } else {
            data.push(0.0);
            nulls.push(true);
            has_nulls = true;
        }
    }

    if has_nulls {
        Ok(Box::new(Series::with_nulls(name, data, nulls)?))
    } else {
        Ok(Box::new(Series::new(name, data)))
    }
}

fn build_bool_series(name: &str, values: &[String]) -> Result<Box<dyn AnySeries>> {
    let mut data = Vec::with_capacity(values.len());
    let mut nulls = Vec::with_capacity(values.len());
    let mut has_nulls = false;

    for v in values {
        if is_null_sentinel(v) {
            data.push(0_u8);
            nulls.push(true);
            has_nulls = true;
        } else if let Some(parsed) = try_parse_bool(v) {
            data.push(u8::from(parsed));
            nulls.push(false);
        } else {
            data.push(0);
            nulls.push(true);
            has_nulls = true;
        }
    }

    if has_nulls {
        Ok(Box::new(Series::with_nulls(name, data, nulls)?))
    } else {
        Ok(Box::new(Series::new(name, data)))
    }
}

fn build_string_series(name: &str, values: &[String]) -> Result<Box<dyn AnySeries>> {
    let mut data = Vec::with_capacity(values.len());
    let mut nulls = Vec::with_capacity(values.len());
    let mut has_nulls = false;

    for v in values {
        if is_null_sentinel(v) {
            data.push(String::new());
            nulls.push(true);
            has_nulls = true;
        } else {
            data.push(v.clone());
            nulls.push(false);
        }
    }

    if has_nulls {
        Ok(Box::new(StringSeries::with_nulls(name, data, nulls)?))
    } else {
        Ok(Box::new(StringSeries::new(name, data)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_sentinels() {
        assert!(is_null_sentinel(""));
        assert!(is_null_sentinel("NA"));
        assert!(is_null_sentinel("N/A"));
        assert!(is_null_sentinel("null"));
        assert!(is_null_sentinel("NULL"));
        assert!(is_null_sentinel("None"));
        assert!(is_null_sentinel("NaN"));
        assert!(is_null_sentinel("nan"));
        assert!(is_null_sentinel("."));
        assert!(is_null_sentinel("-"));
        assert!(!is_null_sentinel("hello"));
        assert!(!is_null_sentinel("42"));
    }

    #[test]
    fn test_parse_bool() {
        assert_eq!(try_parse_bool("true"), Some(true));
        assert_eq!(try_parse_bool("True"), Some(true));
        assert_eq!(try_parse_bool("FALSE"), Some(false));
        assert_eq!(try_parse_bool("0"), Some(false));
        assert_eq!(try_parse_bool("1"), Some(true));
        assert_eq!(try_parse_bool("yes"), Some(true));
        assert_eq!(try_parse_bool("nope"), None);
    }

    #[test]
    fn test_infer_column_type_i64() {
        assert_eq!(infer_column_type(&["1", "2", "3"]), InferredType::I64);
        assert_eq!(infer_column_type(&["1", "NA", "3"]), InferredType::I64);
    }

    #[test]
    fn test_infer_column_type_f64() {
        assert_eq!(infer_column_type(&["1.5", "2.0", "3"]), InferredType::F64);
    }

    #[test]
    fn test_infer_column_type_bool() {
        assert_eq!(
            infer_column_type(&["true", "false", "True"]),
            InferredType::Bool
        );
    }

    #[test]
    fn test_infer_column_type_str() {
        assert_eq!(infer_column_type(&["hello", "world"]), InferredType::Str);
    }

    #[test]
    fn test_infer_column_type_all_null() {
        assert_eq!(infer_column_type(&["NA", "", "null"]), InferredType::Str);
    }

    #[test]
    fn test_build_i64_series() {
        let vals: Vec<String> = vec!["1", "2", "NA", "4"]
            .into_iter()
            .map(String::from)
            .collect();
        let s = build_series_from_strings("col", &vals, InferredType::I64).unwrap();
        assert_eq!(s.len(), 4);
        assert!(!s.is_null(0));
        assert!(s.is_null(2));
        assert_eq!(s.display_value(0), "1");
    }

    #[test]
    fn test_build_f64_series() {
        let vals: Vec<String> = vec!["1.5", "2.0", "null"]
            .into_iter()
            .map(String::from)
            .collect();
        let s = build_series_from_strings("col", &vals, InferredType::F64).unwrap();
        assert_eq!(s.len(), 3);
        assert!(s.is_null(2));
    }

    #[test]
    fn test_build_bool_series() {
        let vals: Vec<String> = vec!["true", "false", "NA"]
            .into_iter()
            .map(String::from)
            .collect();
        let s = build_series_from_strings("col", &vals, InferredType::Bool).unwrap();
        assert_eq!(s.len(), 3);
        assert!(s.is_null(2));
        assert_eq!(s.display_value(0), "1");
        assert_eq!(s.display_value(1), "0");
    }

    #[test]
    fn test_build_string_series() {
        let vals: Vec<String> = vec!["hello", "world", ""]
            .into_iter()
            .map(String::from)
            .collect();
        let s = build_series_from_strings("col", &vals, InferredType::Str).unwrap();
        assert_eq!(s.len(), 3);
        assert!(s.is_null(2)); // empty string is null sentinel
        assert_eq!(s.display_value(0), "hello");
    }
}
