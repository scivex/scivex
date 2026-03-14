//! String column type for non-numeric data.

use std::any::Any;
use std::fmt;

use crate::dtype::DType;
use crate::error::{FrameError, Result};
use crate::series::AnySeries;

/// A named column of `String` values with optional null tracking.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct StringSeries {
    pub(crate) name: String,
    pub(crate) data: Vec<String>,
    pub(crate) null_mask: Option<Vec<bool>>,
}

impl StringSeries {
    /// Create a new string series.
    pub fn new(name: impl Into<String>, data: Vec<String>) -> Self {
        Self {
            name: name.into(),
            data,
            null_mask: None,
        }
    }

    /// Create from string slices.
    pub fn from_strs(name: impl Into<String>, data: &[&str]) -> Self {
        Self::new(name, data.iter().map(|s| (*s).to_string()).collect())
    }

    /// Create with explicit nulls.
    pub fn with_nulls(
        name: impl Into<String>,
        data: Vec<String>,
        null_mask: Vec<bool>,
    ) -> Result<Self> {
        if data.len() != null_mask.len() {
            return Err(FrameError::RowCountMismatch {
                expected: data.len(),
                got: null_mask.len(),
            });
        }
        Ok(Self {
            name: name.into(),
            data,
            null_mask: Some(null_mask),
        })
    }

    // -- Accessors ----------------------------------------------------------

    /// Column name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the series is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the element at `index`.
    pub fn get(&self, index: usize) -> Option<&str> {
        if index >= self.data.len() || self.is_null_at(index) {
            return None;
        }
        Some(&self.data[index])
    }

    /// Whether the element at `index` is null.
    #[inline]
    pub fn is_null_at(&self, index: usize) -> bool {
        self.null_mask
            .as_ref()
            .is_some_and(|m| m.get(index).copied().unwrap_or(false))
    }

    /// Number of null entries.
    pub fn null_count(&self) -> usize {
        self.null_mask
            .as_ref()
            .map_or(0, |m| m.iter().filter(|&&v| v).count())
    }

    /// Rename in place.
    pub fn rename(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Underlying data as a slice.
    pub fn as_slice(&self) -> &[String] {
        &self.data
    }

    // -- String-specific operations -----------------------------------------

    /// Convert all values to uppercase.
    pub fn to_uppercase(&self) -> StringSeries {
        StringSeries {
            name: self.name.clone(),
            data: self.data.iter().map(|s| s.to_uppercase()).collect(),
            null_mask: self.null_mask.clone(),
        }
    }

    /// Convert all values to lowercase.
    pub fn to_lowercase(&self) -> StringSeries {
        StringSeries {
            name: self.name.clone(),
            data: self.data.iter().map(|s| s.to_lowercase()).collect(),
            null_mask: self.null_mask.clone(),
        }
    }

    /// Boolean mask: which values contain the given pattern.
    pub fn contains(&self, pat: &str) -> Vec<bool> {
        self.data.iter().map(|s| s.contains(pat)).collect()
    }

    /// Boolean mask: which values start with the given prefix.
    pub fn starts_with(&self, prefix: &str) -> Vec<bool> {
        self.data.iter().map(|s| s.starts_with(prefix)).collect()
    }

    /// Character length of each value.
    pub fn len_chars(&self) -> Vec<usize> {
        self.data.iter().map(|s| s.chars().count()).collect()
    }

    /// Strip leading and trailing whitespace from each value.
    pub fn strip(&self) -> StringSeries {
        StringSeries {
            name: self.name.clone(),
            data: self.data.iter().map(|s| s.trim().to_string()).collect(),
            null_mask: self.null_mask.clone(),
        }
    }

    /// Replace all occurrences of `old` with `new_val` in each value.
    pub fn replace_all(&self, old: &str, new_val: &str) -> StringSeries {
        StringSeries {
            name: self.name.clone(),
            data: self.data.iter().map(|s| s.replace(old, new_val)).collect(),
            null_mask: self.null_mask.clone(),
        }
    }

    /// Boolean mask: which values end with the given suffix.
    pub fn ends_with(&self, suffix: &str) -> Vec<bool> {
        self.data.iter().map(|s| s.ends_with(suffix)).collect()
    }

    // -- Regex-powered methods (behind `regex` feature) -----------------------

    /// Boolean mask: which values match the given regex pattern.
    #[cfg(feature = "regex")]
    pub fn regex_contains(&self, pattern: &str) -> Result<Vec<bool>> {
        let re = regex::Regex::new(pattern).map_err(|_| FrameError::InvalidValue {
            reason: "invalid regex pattern".into(),
        })?;
        Ok(self.data.iter().map(|s| re.is_match(s)).collect())
    }

    /// Extract the first capture group from each value, or empty string if no match.
    #[cfg(feature = "regex")]
    pub fn regex_extract(&self, pattern: &str) -> Result<StringSeries> {
        let re = regex::Regex::new(pattern).map_err(|_| FrameError::InvalidValue {
            reason: "invalid regex pattern".into(),
        })?;
        let data: Vec<String> = self
            .data
            .iter()
            .map(|s| {
                re.captures(s)
                    .and_then(|c| c.get(1).or_else(|| c.get(0)))
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default()
            })
            .collect();
        Ok(StringSeries {
            name: self.name.clone(),
            data,
            null_mask: self.null_mask.clone(),
        })
    }

    /// Replace all regex matches with the replacement string.
    #[cfg(feature = "regex")]
    pub fn regex_replace(&self, pattern: &str, replacement: &str) -> Result<StringSeries> {
        let re = regex::Regex::new(pattern).map_err(|_| FrameError::InvalidValue {
            reason: "invalid regex pattern".into(),
        })?;
        let data: Vec<String> = self
            .data
            .iter()
            .map(|s| re.replace_all(s, replacement).into_owned())
            .collect();
        Ok(StringSeries {
            name: self.name.clone(),
            data,
            null_mask: self.null_mask.clone(),
        })
    }

    /// Count the number of non-overlapping regex matches in each value.
    #[cfg(feature = "regex")]
    pub fn regex_count(&self, pattern: &str) -> Result<Vec<usize>> {
        let re = regex::Regex::new(pattern).map_err(|_| FrameError::InvalidValue {
            reason: "invalid regex pattern".into(),
        })?;
        Ok(self.data.iter().map(|s| re.find_iter(s).count()).collect())
    }
}

impl fmt::Display for StringSeries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StringSeries({:?}, len={})", self.name, self.data.len())
    }
}

impl AnySeries for StringSeries {
    fn name(&self) -> &str {
        &self.name
    }

    fn dtype(&self) -> DType {
        DType::Str
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn null_count(&self) -> usize {
        self.null_count()
    }

    fn is_null(&self, index: usize) -> bool {
        self.is_null_at(index)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn AnySeries> {
        Box::new(self.clone())
    }

    fn display_value(&self, index: usize) -> String {
        if self.is_null_at(index) {
            "null".to_string()
        } else if index < self.data.len() {
            self.data[index].clone()
        } else {
            String::new()
        }
    }

    fn filter_mask(&self, mask: &[bool]) -> Box<dyn AnySeries> {
        let mut data = Vec::new();
        let mut new_nulls: Option<Vec<bool>> = self.null_mask.as_ref().map(|_| Vec::new());
        for (i, &keep) in mask.iter().enumerate() {
            if keep && i < self.data.len() {
                data.push(self.data[i].clone());
                if let Some(ref mut nm) = new_nulls {
                    nm.push(
                        self.null_mask
                            .as_ref()
                            .expect("null_mask present when has_nulls is true")[i],
                    );
                }
            }
        }
        Box::new(StringSeries {
            name: self.name.clone(),
            data,
            null_mask: new_nulls,
        })
    }

    fn take_indices(&self, indices: &[usize]) -> Box<dyn AnySeries> {
        let data: Vec<String> = indices.iter().map(|&i| self.data[i].clone()).collect();
        let null_mask = self
            .null_mask
            .as_ref()
            .map(|m| indices.iter().map(|&i| m[i]).collect());
        Box::new(StringSeries {
            name: self.name.clone(),
            data,
            null_mask,
        })
    }

    fn slice(&self, offset: usize, length: usize) -> Box<dyn AnySeries> {
        let end = (offset + length).min(self.data.len());
        let data = self.data[offset..end].to_vec();
        let null_mask = self.null_mask.as_ref().map(|m| m[offset..end].to_vec());
        Box::new(StringSeries {
            name: self.name.clone(),
            data,
            null_mask,
        })
    }

    fn rename_box(&self, name: &str) -> Box<dyn AnySeries> {
        let mut cloned = self.clone();
        cloned.name = name.to_string();
        Box::new(cloned)
    }

    fn drop_nulls(&self) -> Box<dyn AnySeries> {
        if self.null_mask.is_none() {
            return self.clone_box();
        }
        let mask = self
            .null_mask
            .as_ref()
            .expect("null_mask present when has_nulls is true");
        let keep: Vec<bool> = mask.iter().map(|&is_null| !is_null).collect();
        self.filter_mask(&keep)
    }

    fn null_mask_vec(&self) -> Vec<bool> {
        self.null_mask
            .clone()
            .unwrap_or_else(|| vec![false; self.data.len()])
    }

    fn null_series(&self, name: &str, len: usize) -> Box<dyn AnySeries> {
        Box::new(StringSeries {
            name: name.to_string(),
            data: vec![String::new(); len],
            null_mask: Some(vec![true; len]),
        })
    }

    fn take_optional(&self, indices: &[Option<usize>]) -> Box<dyn AnySeries> {
        let mut data = Vec::with_capacity(indices.len());
        let mut nulls = Vec::with_capacity(indices.len());
        for opt in indices {
            if let Some(i) = opt {
                data.push(self.data[*i].clone());
                nulls.push(self.is_null_at(*i));
            } else {
                data.push(String::new());
                nulls.push(true);
            }
        }
        let has_nulls = nulls.iter().any(|&v| v);
        Box::new(StringSeries {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(nulls) } else { None },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_series_new() {
        let s = StringSeries::from_strs("name", &["Alice", "Bob", "Charlie"]);
        assert_eq!(s.name(), "name");
        assert_eq!(s.len(), 3);
        assert_eq!(s.get(0), Some("Alice"));
        assert_eq!(s.get(3), None);
    }

    #[test]
    fn test_string_series_uppercase_lowercase() {
        let s = StringSeries::from_strs("x", &["Hello", "World"]);
        let upper = s.to_uppercase();
        assert_eq!(upper.get(0), Some("HELLO"));
        let lower = s.to_lowercase();
        assert_eq!(lower.get(1), Some("world"));
    }

    #[test]
    fn test_string_series_contains() {
        let s = StringSeries::from_strs("x", &["apple", "banana", "apricot"]);
        assert_eq!(s.contains("ap"), vec![true, false, true]);
    }

    #[test]
    fn test_string_series_starts_with() {
        let s = StringSeries::from_strs("x", &["apple", "banana", "apricot"]);
        assert_eq!(s.starts_with("ap"), vec![true, false, true]);
    }

    #[test]
    fn test_string_series_len_chars() {
        let s = StringSeries::from_strs("x", &["hi", "hello"]);
        assert_eq!(s.len_chars(), vec![2, 5]);
    }

    #[test]
    fn test_string_series_any_series() {
        let s = StringSeries::from_strs("name", &["a", "b", "c"]);
        let boxed: Box<dyn AnySeries> = Box::new(s);
        assert_eq!(boxed.dtype(), DType::Str);
        assert_eq!(boxed.display_value(1), "b");
        let downcasted = boxed.as_any().downcast_ref::<StringSeries>().unwrap();
        assert_eq!(downcasted.get(0), Some("a"));
    }

    // -- Edge-case tests -------------------------------------------------------

    #[test]
    fn test_string_series_empty() {
        let s = StringSeries::from_strs("empty", &[]);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.get(0), None);
    }

    #[test]
    fn test_string_series_empty_strings() {
        let s = StringSeries::from_strs("x", &["", "", ""]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.get(0), Some(""));
        assert_eq!(s.contains("a"), vec![false, false, false]);
        assert_eq!(s.len_chars(), vec![0, 0, 0]);
    }

    #[test]
    fn test_string_series_contains_empty_pattern() {
        let s = StringSeries::from_strs("x", &["hello", "world"]);
        // Empty pattern matches everything
        assert_eq!(s.contains(""), vec![true, true]);
    }

    #[test]
    fn test_string_series_starts_with_empty() {
        let s = StringSeries::from_strs("x", &["hello", "world"]);
        assert_eq!(s.starts_with(""), vec![true, true]);
    }

    #[test]
    fn test_string_series_uppercase_lowercase_empty() {
        let s = StringSeries::from_strs("x", &[""]);
        assert_eq!(s.to_uppercase().get(0), Some(""));
        assert_eq!(s.to_lowercase().get(0), Some(""));
    }

    #[test]
    fn test_string_series_with_nulls_display() {
        let s =
            StringSeries::with_nulls("x", vec!["hello".into(), String::new()], vec![false, true])
                .unwrap();
        let boxed: Box<dyn AnySeries> = Box::new(s);
        assert_eq!(boxed.display_value(0), "hello");
        assert_eq!(boxed.display_value(1), "null");
    }

    #[test]
    fn test_string_series_null_count_no_nulls() {
        let s = StringSeries::from_strs("x", &["a", "b", "c"]);
        assert_eq!(s.null_count(), 0);
    }

    #[test]
    fn test_string_series_drop_nulls() {
        let s = StringSeries::with_nulls(
            "x",
            vec!["a".into(), String::new(), "c".into()],
            vec![false, true, false],
        )
        .unwrap();
        let boxed: Box<dyn AnySeries> = Box::new(s);
        let dropped = boxed.drop_nulls();
        assert_eq!(dropped.len(), 2);
        assert_eq!(dropped.null_count(), 0);
    }

    #[test]
    fn test_string_series_take_indices() {
        let s = StringSeries::from_strs("x", &["a", "b", "c", "d"]);
        let boxed: Box<dyn AnySeries> = Box::new(s);
        let taken = boxed.take_indices(&[3, 0]);
        assert_eq!(taken.len(), 2);
        assert_eq!(taken.display_value(0), "d");
        assert_eq!(taken.display_value(1), "a");
    }

    #[test]
    fn test_string_series_slice() {
        let s = StringSeries::from_strs("x", &["a", "b", "c", "d"]);
        let boxed: Box<dyn AnySeries> = Box::new(s);
        let sliced = boxed.slice(1, 2);
        assert_eq!(sliced.len(), 2);
        assert_eq!(sliced.display_value(0), "b");
        assert_eq!(sliced.display_value(1), "c");
    }

    #[test]
    fn test_string_series_with_nulls_length_mismatch() {
        let result = StringSeries::with_nulls("x", vec!["a".into()], vec![false, true]);
        assert!(result.is_err());
    }

    #[test]
    fn test_string_series_rename() {
        let mut s = StringSeries::from_strs("old", &["a"]);
        s.rename("new");
        assert_eq!(s.name(), "new");
    }

    #[test]
    fn test_string_series_contains_no_match() {
        let s = StringSeries::from_strs("x", &["hello", "world"]);
        assert_eq!(s.contains("xyz"), vec![false, false]);
    }

    #[test]
    fn test_string_series_strip() {
        let s = StringSeries::from_strs("x", &["  hello  ", "world", " hi "]);
        let stripped = s.strip();
        assert_eq!(stripped.get(0), Some("hello"));
        assert_eq!(stripped.get(1), Some("world"));
        assert_eq!(stripped.get(2), Some("hi"));
    }

    #[test]
    fn test_string_series_replace_all() {
        let s = StringSeries::from_strs("x", &["hello world", "foo bar"]);
        let replaced = s.replace_all("o", "0");
        assert_eq!(replaced.get(0), Some("hell0 w0rld"));
        assert_eq!(replaced.get(1), Some("f00 bar"));
    }

    #[test]
    fn test_string_series_ends_with() {
        let s = StringSeries::from_strs("x", &["hello.csv", "world.txt", "data.csv"]);
        assert_eq!(s.ends_with(".csv"), vec![true, false, true]);
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_contains() {
        let s = StringSeries::from_strs("x", &["hello123", "world", "abc456"]);
        let result = s.regex_contains(r"\d+").unwrap();
        assert_eq!(result, vec![true, false, true]);
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_extract() {
        let s = StringSeries::from_strs("x", &["hello123", "world", "abc456"]);
        let extracted = s.regex_extract(r"(\d+)").unwrap();
        assert_eq!(extracted.get(0), Some("123"));
        assert_eq!(extracted.get(1), Some(""));
        assert_eq!(extracted.get(2), Some("456"));
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_replace() {
        let s = StringSeries::from_strs("x", &["hello 123 world 456"]);
        let replaced = s.regex_replace(r"\d+", "NUM").unwrap();
        assert_eq!(replaced.get(0), Some("hello NUM world NUM"));
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_count() {
        let s = StringSeries::from_strs("x", &["aaa", "abab", "xyz"]);
        let counts = s.regex_count(r"a").unwrap();
        assert_eq!(counts, vec![3, 2, 0]);
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_invalid_pattern() {
        let s = StringSeries::from_strs("x", &["hello"]);
        assert!(s.regex_contains(r"[invalid").is_err());
    }
}
