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
}
