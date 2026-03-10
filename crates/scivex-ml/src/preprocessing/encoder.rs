use scivex_core::Float;

use crate::error::{MlError, Result};

/// Maps unique label values to contiguous integers `0..n_classes`.
#[derive(Debug, Clone)]
pub struct LabelEncoder<T: Float> {
    classes: Option<Vec<T>>,
}

impl<T: Float> Default for LabelEncoder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> LabelEncoder<T> {
    /// Create a new, unfitted label encoder.
    pub fn new() -> Self {
        Self { classes: None }
    }

    /// Learn the unique class labels from `y`.
    pub fn fit(&mut self, y: &[T]) -> Result<()> {
        if y.is_empty() {
            return Err(MlError::EmptyInput);
        }
        let mut classes: Vec<T> = Vec::new();
        for &v in y {
            if !classes.iter().any(|&c| (c - v).abs() < T::epsilon()) {
                classes.push(v);
            }
        }
        // Sort for deterministic ordering
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.classes = Some(classes);
        Ok(())
    }

    /// Encode labels to integers.
    pub fn transform(&self, y: &[T]) -> Result<Vec<usize>> {
        let classes = self.classes.as_ref().ok_or(MlError::NotFitted)?;
        y.iter()
            .map(|&v| {
                classes
                    .iter()
                    .position(|&c| (c - v).abs() < T::epsilon())
                    .ok_or(MlError::InvalidParameter {
                        name: "label",
                        reason: "unknown label encountered during transform",
                    })
            })
            .collect()
    }

    /// Decode integers back to original labels.
    pub fn inverse_transform(&self, indices: &[usize]) -> Result<Vec<T>> {
        let classes = self.classes.as_ref().ok_or(MlError::NotFitted)?;
        indices
            .iter()
            .map(|&i| {
                classes.get(i).copied().ok_or(MlError::InvalidParameter {
                    name: "index",
                    reason: "index out of range for known classes",
                })
            })
            .collect()
    }

    /// Number of classes learned. Returns `None` if not fitted.
    pub fn n_classes(&self) -> Option<usize> {
        self.classes.as_ref().map(Vec::len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let y = [3.0_f64, 1.0, 2.0, 1.0, 3.0];
        let mut enc = LabelEncoder::new();
        enc.fit(&y).unwrap();
        assert_eq!(enc.n_classes(), Some(3));

        let encoded = enc.transform(&y).unwrap();
        // Sorted classes: [1.0, 2.0, 3.0] → indices [2, 0, 1, 0, 2]
        assert_eq!(encoded, vec![2, 0, 1, 0, 2]);

        let decoded = enc.inverse_transform(&encoded).unwrap();
        assert_eq!(decoded, vec![3.0, 1.0, 2.0, 1.0, 3.0]);
    }

    #[test]
    fn test_not_fitted() {
        let enc = LabelEncoder::<f64>::new();
        assert!(enc.transform(&[1.0]).is_err());
    }
}
