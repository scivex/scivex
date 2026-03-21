//! Pretty-print table display for [`DataFrame`].

use core::fmt;

use super::DataFrame;

/// Maximum rows to show before truncating.
const MAX_DISPLAY_ROWS: usize = 20;
/// Rows to show from the head when truncated.
const TRUNC_HEAD: usize = 10;
/// Rows to show from the tail when truncated.
const TRUNC_TAIL: usize = 10;

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "(empty DataFrame)");
        }

        let nrows = self.nrows();
        let ncols = self.ncols();
        let truncated = nrows > MAX_DISPLAY_ROWS;

        // Collect header names.
        let headers: Vec<&str> = self.columns.iter().map(|c| c.name()).collect();

        // Determine which rows to display.
        let row_indices: Vec<usize> = if truncated {
            let mut v: Vec<usize> = (0..TRUNC_HEAD).collect();
            v.push(usize::MAX); // sentinel for "..."
            v.extend(nrows - TRUNC_TAIL..nrows);
            v
        } else {
            (0..nrows).collect()
        };

        // Build cell values.
        let mut cells: Vec<Vec<String>> = Vec::with_capacity(row_indices.len());
        for &ri in &row_indices {
            if ri == usize::MAX {
                cells.push(vec!["...".to_string(); ncols]);
            } else {
                let row: Vec<String> = self.columns.iter().map(|c| c.display_value(ri)).collect();
                cells.push(row);
            }
        }

        // Compute column widths.
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        for row in &cells {
            for (j, cell) in row.iter().enumerate() {
                widths[j] = widths[j].max(cell.len());
            }
        }

        // Print header.
        write!(f, "|")?;
        for (j, header) in headers.iter().enumerate() {
            write!(f, " {header:>w$} |", w = widths[j])?;
        }
        writeln!(f)?;

        // Print separator.
        write!(f, "|")?;
        for w in &widths {
            write!(f, "-{}-|", "-".repeat(*w))?;
        }
        writeln!(f)?;

        // Print rows.
        for row in &cells {
            write!(f, "|")?;
            for (j, cell) in row.iter().enumerate() {
                write!(f, " {cell:>w$} |", w = widths[j])?;
            }
            writeln!(f)?;
        }

        if truncated {
            writeln!(f, "({nrows} rows x {ncols} columns)")?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HTML table rendering (for Jupyter / evcxr notebooks)
// ---------------------------------------------------------------------------

impl DataFrame {
    /// Render the DataFrame as an HTML `<table>`.
    ///
    /// Large DataFrames are truncated the same way as the text display
    /// (first 10 + last 10 rows with an ellipsis row in between).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::{DataFrame, Series};
    /// let df = DataFrame::builder()
    ///     .add_column("x", vec![1_i32, 2])
    ///     .build()
    ///     .unwrap();
    /// let html = df.to_html();
    /// assert!(html.contains("<table"));
    /// ```
    pub fn to_html(&self) -> String {
        if self.is_empty() {
            return "<p><em>(empty DataFrame)</em></p>".to_string();
        }

        let nrows = self.nrows();
        let ncols = self.ncols();
        let truncated = nrows > MAX_DISPLAY_ROWS;

        let row_indices: Vec<usize> = if truncated {
            let mut v: Vec<usize> = (0..TRUNC_HEAD).collect();
            v.push(usize::MAX);
            v.extend(nrows - TRUNC_TAIL..nrows);
            v
        } else {
            (0..nrows).collect()
        };

        let mut html = String::with_capacity(256 + nrows * ncols * 20);
        html.push_str("<table style=\"border-collapse:collapse;\">\n<thead>\n<tr>");

        // Index column header
        html.push_str("<th style=\"border:1px solid #ddd;padding:4px 8px;\"></th>");

        // Column headers
        for col in &self.columns {
            html.push_str(
                "<th style=\"border:1px solid #ddd;padding:4px 8px;background:#f4f4f4;\">",
            );
            html_escape_into(&mut html, col.name());
            html.push_str("</th>");
        }
        html.push_str("</tr>\n</thead>\n<tbody>\n");

        // Data rows
        for &ri in &row_indices {
            html.push_str("<tr>");
            if ri == usize::MAX {
                html.push_str(
                    "<td style=\"border:1px solid #ddd;padding:4px 8px;text-align:center;\">…</td>",
                );
                for _ in 0..ncols {
                    html.push_str("<td style=\"border:1px solid #ddd;padding:4px 8px;text-align:center;\">…</td>");
                }
            } else {
                // Row index
                html.push_str(
                    "<td style=\"border:1px solid #ddd;padding:4px 8px;background:#f9f9f9;\">",
                );
                html.push_str(&ri.to_string());
                html.push_str("</td>");

                for col in &self.columns {
                    html.push_str("<td style=\"border:1px solid #ddd;padding:4px 8px;\">");
                    if col.is_null(ri) {
                        html.push_str("<em>null</em>");
                    } else {
                        html_escape_into(&mut html, &col.display_value(ri));
                    }
                    html.push_str("</td>");
                }
            }
            html.push_str("</tr>\n");
        }

        html.push_str("</tbody>\n</table>\n");

        if truncated {
            use std::fmt::Write;
            let _ = writeln!(html, "<p>{nrows} rows × {ncols} columns</p>");
        }

        html
    }

    /// Display this DataFrame in an evcxr Jupyter notebook.
    ///
    /// This method is auto-detected by the evcxr kernel and used to render
    /// rich HTML output in Jupyter cells.
    pub fn evcxr_display(&self) {
        println!(
            "EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT",
            self.to_html()
        );
    }
}

/// Escape HTML special characters into `out`.
fn html_escape_into(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::series::Series;

    use super::*;

    #[test]
    fn test_display_basic() {
        let df = DataFrame::new(vec![
            Box::new(Series::new("name", vec![1_i32, 2])),
            Box::new(Series::new("val", vec![10_i32, 20])),
        ])
        .unwrap();
        let output = format!("{df}");
        assert!(output.contains("name"));
        assert!(output.contains("val"));
        assert!(output.contains('1'));
        assert!(output.contains("20"));
    }

    #[test]
    fn test_display_empty() {
        let df = DataFrame::empty();
        assert_eq!(format!("{df}"), "(empty DataFrame)");
    }

    #[test]
    fn test_to_html_basic() {
        let df = DataFrame::new(vec![
            Box::new(Series::new("x", vec![1_i32, 2, 3])),
            Box::new(Series::new("y", vec![10.0_f64, 20.0, 30.0])),
        ])
        .unwrap();
        let html = df.to_html();
        assert!(html.contains("<table"));
        assert!(html.contains("<thead>"));
        assert!(html.contains("<tbody>"));
        assert!(html.contains(">x</th>"));
        assert!(html.contains(">y</th>"));
        assert!(html.contains(">1</td>"));
        assert!(html.contains(">30</td>"));
    }

    #[test]
    fn test_to_html_empty() {
        let df = DataFrame::empty();
        let html = df.to_html();
        assert!(html.contains("empty DataFrame"));
    }

    #[test]
    fn test_to_html_escapes() {
        let df = DataFrame::new(vec![Box::new(crate::StringSeries::from_strs(
            "text",
            &["<b>bold</b>", "a & b"],
        )) as _])
        .unwrap();
        let html = df.to_html();
        assert!(html.contains("&lt;b&gt;bold&lt;/b&gt;"));
        assert!(html.contains("a &amp; b"));
    }

    #[test]
    fn test_to_html_truncation() {
        let data: Vec<i32> = (0..50).collect();
        let df = DataFrame::new(vec![Box::new(Series::new("n", data))]).unwrap();
        let html = df.to_html();
        assert!(html.contains("…")); // ellipsis row
        assert!(html.contains("50 rows"));
    }
}
