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

// ---------------------------------------------------------------------------
// HTML rendering for Jupyter / evcxr
// ---------------------------------------------------------------------------

/// Maximum elements to show for 1-D/flat tensors in HTML.
const MAX_HTML_ELEMENTS: usize = 100;

impl<T: Scalar> Tensor<T> {
    /// Render the tensor as an HTML string.
    ///
    /// - Scalars and 1-D tensors are shown as formatted values.
    /// - 2-D tensors are rendered as `<table>`.
    /// - Higher-dimensional tensors show shape and a data summary.
    pub fn to_html(&self) -> String {
        use fmt::Write;

        if self.is_empty() {
            return format!("<pre>tensor([], shape={:?})</pre>", self.shape);
        }

        match self.ndim() {
            0 => format!("<pre>tensor({})</pre>", self.data[0]),
            1 => {
                let n = self.data.len();
                let truncated = n > MAX_HTML_ELEMENTS;
                let show = if truncated { MAX_HTML_ELEMENTS } else { n };
                let mut s = String::from("<pre>tensor([");
                for (i, v) in self.data.iter().take(show).enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    let _ = write!(s, "{v}");
                }
                if truncated {
                    let _ = write!(s, ", ... ({n} elements)");
                }
                s.push_str("])</pre>");
                s
            }
            2 => {
                let rows = self.shape[0];
                let cols = self.shape[1];
                let mut html = String::with_capacity(64 + rows * cols * 12);
                let _ = writeln!(
                    html,
                    "<div><strong>Tensor</strong> shape=[{rows}, {cols}]</div>"
                );
                html.push_str("<table style=\"border-collapse:collapse;\">\n<tbody>\n");
                let max_rows = 20;
                let truncated = rows > max_rows;
                for r in 0..rows.min(max_rows) {
                    html.push_str("<tr>");
                    for c in 0..cols {
                        html.push_str("<td style=\"border:1px solid #ddd;padding:2px 6px;text-align:right;\">");
                        let _ = write!(html, "{}", self.data[r * cols + c]);
                        html.push_str("</td>");
                    }
                    html.push_str("</tr>\n");
                }
                if truncated {
                    html.push_str("<tr>");
                    for _ in 0..cols {
                        html.push_str("<td style=\"border:1px solid #ddd;padding:2px 6px;text-align:center;\">…</td>");
                    }
                    html.push_str("</tr>\n");
                }
                html.push_str("</tbody>\n</table>\n");
                html
            }
            _ => {
                format!(
                    "<pre>tensor(shape={:?}, numel={})</pre>",
                    self.shape,
                    self.data.len()
                )
            }
        }
    }

    /// Display this tensor in an evcxr Jupyter notebook.
    ///
    /// Auto-detected by the evcxr kernel for rich HTML output.
    pub fn evcxr_display(&self) {
        println!("EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT", self.to_html());
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

    #[test]
    fn test_to_html_scalar() {
        let t = Tensor::scalar(42_i32);
        let html = t.to_html();
        assert!(html.contains("tensor(42)"));
        assert!(html.contains("<pre>"));
    }

    #[test]
    fn test_to_html_1d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let html = t.to_html();
        assert!(html.contains("tensor(["));
        assert!(html.contains('1'));
        assert!(html.contains('3'));
    }

    #[test]
    fn test_to_html_2d() {
        let t = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let html = t.to_html();
        assert!(html.contains("<table"));
        assert!(html.contains("<tbody>"));
        assert!(html.contains("shape=[2, 2]"));
        assert!(html.contains(">1</td>"));
        assert!(html.contains(">4</td>"));
    }

    #[test]
    fn test_to_html_empty() {
        let t = Tensor::<f64>::zeros(vec![0]);
        let html = t.to_html();
        assert!(html.contains("[]"));
    }

    #[test]
    fn test_to_html_3d() {
        let t = Tensor::<i32>::arange(24).reshape(vec![2, 3, 4]).unwrap();
        let html = t.to_html();
        assert!(html.contains("shape=[2, 3, 4]"));
        assert!(html.contains("numel=24"));
    }
}
