//! Excel (.xlsx) writer for [`DataFrame`](scivex_frame::DataFrame).

use std::path::Path;

use rust_xlsxwriter::Workbook;
use scivex_frame::DataFrame;

use crate::error::{IoError, Result};

/// Builder for writing a [`DataFrame`] to an Excel file.
///
/// # Example
///
/// ```no_run
/// use scivex_io::excel::ExcelWriterBuilder;
/// use scivex_frame::{DataFrame, Series};
///
/// let df = DataFrame::builder()
///     .add_column("x", vec![1_i32, 2, 3])
///     .build()
///     .unwrap();
///
/// ExcelWriterBuilder::new()
///     .sheet_name("Data")
///     .write_path("output.xlsx", &df)
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ExcelWriterBuilder {
    sheet_name: String,
    write_header: bool,
}

impl Default for ExcelWriterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExcelWriterBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            sheet_name: "Sheet1".to_string(),
            write_header: true,
        }
    }

    /// Set the sheet name.
    pub fn sheet_name(mut self, name: &str) -> Self {
        self.sheet_name = name.to_string();
        self
    }

    /// Whether to write column names as the first row (default: true).
    pub fn write_header(mut self, val: bool) -> Self {
        self.write_header = val;
        self
    }

    /// Write a `DataFrame` to an Excel file.
    pub fn write_path<P: AsRef<Path>>(&self, path: P, df: &DataFrame) -> Result<()> {
        let mut workbook = Workbook::new();
        let worksheet = workbook.add_worksheet();
        worksheet
            .set_name(&self.sheet_name)
            .map_err(|e| IoError::FormatError(e.to_string()))?;

        let mut row_offset: u32 = 0;

        // Write header
        if self.write_header {
            for (col_idx, name) in df.column_names().iter().enumerate() {
                worksheet
                    .write_string(0, col_idx as u16, *name)
                    .map_err(|e| IoError::FormatError(e.to_string()))?;
            }
            row_offset = 1;
        }

        // Write data
        for row in 0..df.nrows() {
            for (col_idx, col) in df.columns().iter().enumerate() {
                let excel_row = row as u32 + row_offset;
                let excel_col = col_idx as u16;

                if col.is_null(row) {
                    // Leave cell empty for nulls
                    continue;
                }

                write_cell(worksheet, excel_row, excel_col, col.as_ref(), row)?;
            }
        }

        workbook
            .save(path)
            .map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(())
    }
}

/// Write a `DataFrame` to an Excel file with default settings.
pub fn write_excel<P: AsRef<Path>>(path: P, df: &DataFrame) -> Result<()> {
    ExcelWriterBuilder::new().write_path(path, df)
}

/// Write a single cell value. Tries numeric first, falls back to string.
fn write_cell(
    ws: &mut rust_xlsxwriter::Worksheet,
    row: u32,
    col: u16,
    series: &dyn scivex_frame::AnySeries,
    idx: usize,
) -> Result<()> {
    let val = series.display_value(idx);

    // Try writing as number first
    if let Ok(n) = val.parse::<f64>() {
        ws.write_number(row, col, n)
            .map_err(|e| IoError::FormatError(e.to_string()))?;
    } else if val == "true" || val == "false" {
        ws.write_boolean(row, col, val == "true")
            .map_err(|e| IoError::FormatError(e.to_string()))?;
    } else {
        ws.write_string(row, col, &val)
            .map_err(|e| IoError::FormatError(e.to_string()))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::excel::read_excel;

    #[test]
    fn test_write_read_roundtrip() {
        let df = DataFrame::builder()
            .add_column("id", vec![1_i64, 2, 3])
            .add_column("score", vec![95.5_f64, 87.0, 92.3])
            .build()
            .unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("scivex_excel_test.xlsx");

        write_excel(&path, &df).unwrap();

        let result = read_excel(&path).unwrap();
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
        assert_eq!(result.column_names(), vec!["id", "score"]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_with_custom_sheet() {
        let df = DataFrame::builder()
            .add_column("x", vec![1_i32])
            .build()
            .unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("scivex_excel_sheet_test.xlsx");

        ExcelWriterBuilder::new()
            .sheet_name("MyData")
            .write_path(&path, &df)
            .unwrap();

        let result = read_excel(&path).unwrap();
        assert_eq!(result.nrows(), 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_builder_defaults() {
        let builder = ExcelWriterBuilder::new();
        assert_eq!(builder.sheet_name, "Sheet1");
        assert!(builder.write_header);
    }
}
