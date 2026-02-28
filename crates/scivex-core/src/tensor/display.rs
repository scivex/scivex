//! `Display` and `Debug` formatting for [`Tensor`].

use core::fmt;

use crate::Scalar;

use super::Tensor;

impl<T: Scalar> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "tensor([], shape={:?})", self.shape);
        }

        match self.ndim() {
            0 => write!(f, "tensor({})", self.data[0]),
            1 => {
                write!(f, "tensor([")?;
                for (i, v) in self.data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "])")
            }
            2 => {
                let rows = self.shape[0];
                let cols = self.shape[1];
                writeln!(f, "tensor([")?;
                for r in 0..rows {
                    write!(f, "  [")?;
                    for c in 0..cols {
                        if c > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", self.data[r * cols + c])?;
                    }
                    if r < rows - 1 {
                        writeln!(f, "],")?;
                    } else {
                        writeln!(f, "]")?;
                    }
                }
                write!(f, "])")
            }
            _ => {
                // For 3-D+ tensors, show shape and flat data summary
                write!(
                    f,
                    "tensor(shape={:?}, data=[{}, {}, ..., {}])",
                    self.shape,
                    self.data[0],
                    self.data[1],
                    self.data[self.data.len() - 1]
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_scalar() {
        let t = Tensor::scalar(42_i32);
        assert_eq!(format!("{t}"), "tensor(42)");
    }

    #[test]
    fn test_display_1d() {
        let t = Tensor::from_vec(vec![1, 2, 3], vec![3]).unwrap();
        assert_eq!(format!("{t}"), "tensor([1, 2, 3])");
    }

    #[test]
    fn test_display_2d() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let s = format!("{t}");
        assert!(s.contains("tensor("));
        assert!(s.contains("[1, 2]"));
        assert!(s.contains("[3, 4]"));
    }

    #[test]
    fn test_display_empty() {
        let t = Tensor::<f64>::zeros(vec![0]);
        let s = format!("{t}");
        assert!(s.contains("[]"));
    }

    #[test]
    fn test_display_3d() {
        let t = Tensor::<i32>::arange(24).reshape(vec![2, 3, 4]).unwrap();
        let s = format!("{t}");
        assert!(s.contains("shape=[2, 3, 4]"));
    }
}
