//! Dictionary-encoded categorical column type.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;

use crate::dtype::DType;
use crate::error::{FrameError, Result};
use crate::series::AnySeries;
use crate::series::string::StringSeries;

/// A dictionary-encoded categorical column.
///
/// Stores unique category strings in a dictionary and references them by index,
/// making repeated string values much more memory-efficient.
#[derive(Debug, Clone)]
pub struct CategoricalSeries {
    name: String,
    categories: Vec<String>,
    codes: Vec<u32>,
    null_mask: Option<Vec<bool>>,
}

impl CategoricalSeries {
    /// Create from string slices, automatically building the dictionary.
    pub fn from_strs(name: impl Into<String>, data: &[&str]) -> Self {
        let mut categories = Vec::new();
        let mut cat_map: HashMap<String, u32> = HashMap::new();
        let mut codes = Vec::with_capacity(data.len());

        for &s in data {
            let code = if let Some(&c) = cat_map.get(s) {
                c
            } else {
                let c = categories.len() as u32;
                categories.push(s.to_string());
                cat_map.insert(s.to_string(), c);
                c
            };
            codes.push(code);
        }

        Self {
            name: name.into(),
            categories,
            codes,
            null_mask: None,
        }
    }

    /// Create with explicit categories and codes.
    pub fn new(name: impl Into<String>, categories: Vec<String>, codes: Vec<u32>) -> Result<Self> {
        let n_cats = categories.len() as u32;
        for &c in &codes {
            if c >= n_cats {
                return Err(FrameError::IndexOutOfBounds {
                    index: c as usize,
                    length: categories.len(),
                });
            }
        }
        Ok(Self {
            name: name.into(),
            categories,
            codes,
            null_mask: None,
        })
    }

    /// Create with explicit null positions.
    pub fn with_nulls(
        name: impl Into<String>,
        categories: Vec<String>,
        codes: Vec<u32>,
        null_mask: Vec<bool>,
    ) -> Result<Self> {
        if codes.len() != null_mask.len() {
            return Err(FrameError::RowCountMismatch {
                expected: codes.len(),
                got: null_mask.len(),
            });
        }
        Ok(Self {
            name: name.into(),
            categories,
            codes,
            null_mask: Some(null_mask),
        })
    }

    /// The unique category values.
    pub fn categories(&self) -> &[String] {
        &self.categories
    }

    /// The integer codes indexing into categories.
    pub fn codes(&self) -> &[u32] {
        &self.codes
    }

    /// Number of unique categories.
    pub fn n_categories(&self) -> usize {
        self.categories.len()
    }

    /// Get the decoded string at `index`.
    pub fn get(&self, index: usize) -> Option<&str> {
        if index >= self.codes.len() || self.is_null_at(index) {
            return None;
        }
        Some(&self.categories[self.codes[index] as usize])
    }

    /// Column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Whether the series is empty.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Whether the element at `index` is null.
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

    /// Convert to a [`StringSeries`] (decoding all values).
    pub fn to_string_series(&self) -> StringSeries {
        let data: Vec<String> = self
            .codes
            .iter()
            .map(|&c| self.categories[c as usize].clone())
            .collect();
        if let Some(ref mask) = self.null_mask {
            StringSeries::with_nulls(self.name.clone(), data, mask.clone()).unwrap()
        } else {
            StringSeries::new(self.name.clone(), data)
        }
    }

    /// Add a new category to the dictionary.
    pub fn add_category(&mut self, category: &str) {
        if !self.categories.iter().any(|c| c == category) {
            self.categories.push(category.to_string());
        }
    }

    /// Rename an existing category.
    pub fn rename_category(&mut self, old: &str, new: &str) -> Result<()> {
        let pos =
            self.categories
                .iter()
                .position(|c| c == old)
                .ok_or(FrameError::InvalidArgument {
                    reason: "category not found",
                })?;
        self.categories[pos] = new.to_string();
        Ok(())
    }

    /// Reorder categories to match the given order.
    pub fn reorder_categories(&mut self, new_order: &[&str]) -> Result<()> {
        if new_order.len() != self.categories.len() {
            return Err(FrameError::InvalidArgument {
                reason: "new_order length must match category count",
            });
        }
        // Build old→new index mapping
        let mut old_to_new = vec![0u32; self.categories.len()];
        let mut new_cats = Vec::with_capacity(new_order.len());
        for (new_idx, &cat) in new_order.iter().enumerate() {
            let old_idx = self.categories.iter().position(|c| c == cat).ok_or(
                FrameError::InvalidArgument {
                    reason: "category in new_order not found in existing categories",
                },
            )?;
            old_to_new[old_idx] = new_idx as u32;
            new_cats.push(cat.to_string());
        }
        // Remap codes
        for code in &mut self.codes {
            *code = old_to_new[*code as usize];
        }
        self.categories = new_cats;
        Ok(())
    }
}

impl fmt::Display for CategoricalSeries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CategoricalSeries({:?}, len={}, categories={})",
            self.name,
            self.codes.len(),
            self.categories.len()
        )
    }
}

impl AnySeries for CategoricalSeries {
    fn name(&self) -> &str {
        &self.name
    }

    fn dtype(&self) -> DType {
        DType::Categorical
    }

    fn len(&self) -> usize {
        self.codes.len()
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
        } else if index < self.codes.len() {
            self.categories[self.codes[index] as usize].clone()
        } else {
            String::new()
        }
    }

    fn filter_mask(&self, mask: &[bool]) -> Box<dyn AnySeries> {
        let mut codes = Vec::new();
        let mut new_nulls: Option<Vec<bool>> = self.null_mask.as_ref().map(|_| Vec::new());
        for (i, &keep) in mask.iter().enumerate() {
            if keep && i < self.codes.len() {
                codes.push(self.codes[i]);
                if let Some(ref mut nm) = new_nulls {
                    nm.push(self.null_mask.as_ref().unwrap()[i]);
                }
            }
        }
        Box::new(CategoricalSeries {
            name: self.name.clone(),
            categories: self.categories.clone(),
            codes,
            null_mask: new_nulls,
        })
    }

    fn take_indices(&self, indices: &[usize]) -> Box<dyn AnySeries> {
        let codes: Vec<u32> = indices.iter().map(|&i| self.codes[i]).collect();
        let null_mask = self
            .null_mask
            .as_ref()
            .map(|m| indices.iter().map(|&i| m[i]).collect());
        Box::new(CategoricalSeries {
            name: self.name.clone(),
            categories: self.categories.clone(),
            codes,
            null_mask,
        })
    }

    fn slice(&self, offset: usize, length: usize) -> Box<dyn AnySeries> {
        let end = (offset + length).min(self.codes.len());
        let codes = self.codes[offset..end].to_vec();
        let null_mask = self.null_mask.as_ref().map(|m| m[offset..end].to_vec());
        Box::new(CategoricalSeries {
            name: self.name.clone(),
            categories: self.categories.clone(),
            codes,
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
        let mask = self.null_mask.as_ref().unwrap();
        let keep: Vec<bool> = mask.iter().map(|&is_null| !is_null).collect();
        self.filter_mask(&keep)
    }

    fn null_mask_vec(&self) -> Vec<bool> {
        self.null_mask
            .clone()
            .unwrap_or_else(|| vec![false; self.codes.len()])
    }

    fn null_series(&self, name: &str, len: usize) -> Box<dyn AnySeries> {
        Box::new(CategoricalSeries {
            name: name.to_string(),
            categories: self.categories.clone(),
            codes: vec![0; len],
            null_mask: Some(vec![true; len]),
        })
    }

    fn take_optional(&self, indices: &[Option<usize>]) -> Box<dyn AnySeries> {
        let mut codes = Vec::with_capacity(indices.len());
        let mut nulls = Vec::with_capacity(indices.len());
        for opt in indices {
            if let Some(i) = opt {
                codes.push(self.codes[*i]);
                nulls.push(self.is_null_at(*i));
            } else {
                codes.push(0);
                nulls.push(true);
            }
        }
        let has_nulls = nulls.iter().any(|&v| v);
        Box::new(CategoricalSeries {
            name: self.name.clone(),
            categories: self.categories.clone(),
            codes,
            null_mask: if has_nulls { Some(nulls) } else { None },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_strs() {
        let cat = CategoricalSeries::from_strs("color", &["red", "blue", "red", "green", "blue"]);
        assert_eq!(cat.n_categories(), 3);
        assert_eq!(cat.len(), 5);
        assert_eq!(cat.get(0), Some("red"));
        assert_eq!(cat.get(1), Some("blue"));
        assert_eq!(cat.codes(), &[0, 1, 0, 2, 1]);
    }

    #[test]
    fn test_get() {
        let cat = CategoricalSeries::from_strs("x", &["a", "b", "c"]);
        assert_eq!(cat.get(0), Some("a"));
        assert_eq!(cat.get(2), Some("c"));
        assert_eq!(cat.get(5), None);
    }

    #[test]
    fn test_to_string_series() {
        let cat = CategoricalSeries::from_strs("x", &["a", "b", "a"]);
        let ss = cat.to_string_series();
        assert_eq!(ss.get(0), Some("a"));
        assert_eq!(ss.get(1), Some("b"));
        assert_eq!(ss.get(2), Some("a"));
    }

    #[test]
    fn test_any_series_trait() {
        let cat = CategoricalSeries::from_strs("x", &["a", "b", "c"]);
        let boxed: Box<dyn AnySeries> = Box::new(cat);
        assert_eq!(boxed.dtype(), DType::Categorical);
        assert_eq!(boxed.display_value(1), "b");
        assert_eq!(boxed.len(), 3);

        let filtered = boxed.filter_mask(&[true, false, true]);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered.display_value(0), "a");
        assert_eq!(filtered.display_value(1), "c");
    }

    #[test]
    fn test_rename_category() {
        let mut cat = CategoricalSeries::from_strs("x", &["a", "b", "a"]);
        cat.rename_category("a", "alpha").unwrap();
        assert_eq!(cat.get(0), Some("alpha"));
        assert_eq!(cat.get(2), Some("alpha"));
    }

    #[test]
    fn test_filter_categorical() {
        let cat = CategoricalSeries::from_strs("x", &["a", "b", "c", "d"]);
        let boxed: Box<dyn AnySeries> = Box::new(cat);
        let taken = boxed.take_indices(&[3, 1]);
        assert_eq!(taken.display_value(0), "d");
        assert_eq!(taken.display_value(1), "b");
    }

    #[test]
    fn test_reorder_categories() {
        let mut cat = CategoricalSeries::from_strs("x", &["a", "b", "c", "a"]);
        cat.reorder_categories(&["c", "b", "a"]).unwrap();
        assert_eq!(cat.categories(), &["c", "b", "a"]);
        // "a" was code 0, now should be code 2
        assert_eq!(cat.get(0), Some("a"));
        assert_eq!(cat.get(1), Some("b"));
    }
}
