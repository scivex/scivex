//! Excel (.xlsx/.xls/.ods) reader that produces a [`DataFrame`].

use std::path::Path;

use calamine::{Data, Range, Reader, Sheets, open_workbook_auto};
use scivex_frame::DataFrame;

use crate::common::{build_series_from_strings, infer_column_type};
use crate::error::{IoError, Result};

/// Builder for reading Excel files into a [`DataFrame`].
///
/// # Example
///
/// ```no_run
/// use scivex_io::excel::ExcelReaderBuilder;
///
/// let df = ExcelReaderBuilder::new()
///     .sheet_name("Sheet1")
///     .read_path("data.xlsx")
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ExcelReaderBuilder {
    sheet: SheetSelector,
    has_header: bool,
    skip_rows: usize,
    max_rows: Option<usize>,
}

#[derive(Debug, Clone)]
enum SheetSelector {
    Index(usize),
    Name(String),
}

impl Default for ExcelReaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExcelReaderBuilder {
    /// Create a new builder with default settings (first sheet, headers on).
    pub fn new() -> Self {
        Self {
            sheet: SheetSelector::Index(0),
            has_header: true,
            skip_rows: 0,
            max_rows: None,
        }
    }

    /// Select sheet by name.
    pub fn sheet_name(mut self, name: &str) -> Self {
        self.sheet = SheetSelector::Name(name.to_string());
        self
    }

    /// Select sheet by index (0-based).
    pub fn sheet_index(mut self, index: usize) -> Self {
        self.sheet = SheetSelector::Index(index);
        self
    }

    /// Whether the first row is a header row (default: true).
    pub fn has_header(mut self, val: bool) -> Self {
        self.has_header = val;
        self
    }

    /// Skip the first `n` rows before reading.
    pub fn skip_rows(mut self, n: usize) -> Self {
        self.skip_rows = n;
        self
    }

    /// Read at most `n` data rows.
    pub fn max_rows(mut self, n: usize) -> Self {
        self.max_rows = Some(n);
        self
    }

    /// Read an Excel file from a path.
    pub fn read_path<P: AsRef<Path>>(&self, path: P) -> Result<DataFrame> {
        let mut workbook: Sheets<_> =
            open_workbook_auto(path).map_err(|e| IoError::FormatError(e.to_string()))?;
        let range = self.get_range(&mut workbook)?;
        self.range_to_dataframe(&range)
    }

    fn get_range(
        &self,
        workbook: &mut Sheets<std::io::BufReader<std::fs::File>>,
    ) -> Result<Range<Data>> {
        match &self.sheet {
            SheetSelector::Index(idx) => {
                let sheets = workbook.sheet_names().clone();
                let name = sheets
                    .get(*idx)
                    .ok_or_else(|| {
                        IoError::FormatError(format!("sheet index {idx} out of bounds"))
                    })?
                    .clone();
                workbook
                    .worksheet_range(&name)
                    .map_err(|e| IoError::FormatError(e.to_string()))
            }
            SheetSelector::Name(name) => workbook
                .worksheet_range(name)
                .map_err(|e| IoError::FormatError(e.to_string())),
        }
    }

    fn range_to_dataframe(&self, range: &Range<Data>) -> Result<DataFrame> {
        let rows: Vec<_> = range.rows().skip(self.skip_rows).collect();
        if rows.is_empty() {
            return Ok(DataFrame::empty());
        }

        let (headers, data_rows) = if self.has_header {
            let header_row = rows[0];
            let headers: Vec<String> = header_row
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let s = cell_to_string(cell);
                    if s.is_empty() {
                        format!("column_{i}")
                    } else {
                        s
                    }
                })
                .collect();
            (headers, &rows[1..])
        } else {
            let ncols = rows[0].len();
            let headers: Vec<String> = (0..ncols).map(|i| format!("column_{i}")).collect();
            (headers, &rows[..])
        };

        let ncols = headers.len();
        let max = self
            .max_rows
            .unwrap_or(data_rows.len())
            .min(data_rows.len());
        let data_rows = &data_rows[..max];

        let mut columns: Vec<Vec<String>> = vec![Vec::with_capacity(data_rows.len()); ncols];
        for row in data_rows {
            for (col_idx, col) in columns.iter_mut().enumerate() {
                let val = row.get(col_idx).map_or_else(String::new, cell_to_string);
                col.push(val);
            }
        }

        let mut series_list: Vec<Box<dyn scivex_frame::AnySeries>> = Vec::with_capacity(ncols);
        for (i, col_data) in columns.into_iter().enumerate() {
            let name = &headers[i];
            let refs: Vec<&str> = col_data.iter().map(String::as_str).collect();
            let dtype = infer_column_type(&refs);
            let series = build_series_from_strings(name, &col_data, dtype)?;
            series_list.push(series);
        }

        Ok(DataFrame::new(series_list)?)
    }
}

/// Read an Excel file with default settings.
pub fn read_excel<P: AsRef<Path>>(path: P) -> Result<DataFrame> {
    ExcelReaderBuilder::new().read_path(path)
}

fn cell_to_string(cell: &Data) -> String {
    match cell {
        Data::Int(i) => i.to_string(),
        Data::Float(f) => f.to_string(),
        Data::String(s) | Data::DateTimeIso(s) | Data::DurationIso(s) => s.clone(),
        Data::Bool(b) => b.to_string(),
        Data::DateTime(dt) => dt.to_string(),
        Data::Error(e) => format!("#ERR:{e:?}"),
        Data::Empty => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let builder = ExcelReaderBuilder::new();
        assert!(builder.has_header);
        assert_eq!(builder.skip_rows, 0);
        assert!(builder.max_rows.is_none());
    }

    #[test]
    fn test_cell_to_string() {
        assert_eq!(cell_to_string(&Data::Int(42)), "42");
        assert_eq!(cell_to_string(&Data::Float(1.23)), "1.23");
        assert_eq!(cell_to_string(&Data::String("hello".into())), "hello");
        assert_eq!(cell_to_string(&Data::Bool(true)), "true");
        assert_eq!(cell_to_string(&Data::Empty), "");
    }

    #[test]
    fn test_range_to_dataframe() {
        let mut range = Range::new((0, 0), (2, 1));
        range.set_value((0, 0), Data::String("name".into()));
        range.set_value((0, 1), Data::String("age".into()));
        range.set_value((1, 0), Data::String("Alice".into()));
        range.set_value((1, 1), Data::Int(30));
        range.set_value((2, 0), Data::String("Bob".into()));
        range.set_value((2, 1), Data::Int(25));

        let builder = ExcelReaderBuilder::new();
        let df = builder.range_to_dataframe(&range).unwrap();
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.ncols(), 2);
        assert_eq!(df.column_names(), vec!["name", "age"]);
    }

    #[test]
    fn test_range_to_dataframe_no_header() {
        let mut range = Range::new((0, 0), (1, 1));
        range.set_value((0, 0), Data::Int(1));
        range.set_value((0, 1), Data::Int(2));
        range.set_value((1, 0), Data::Int(3));
        range.set_value((1, 1), Data::Int(4));

        let builder = ExcelReaderBuilder::new().has_header(false);
        let df = builder.range_to_dataframe(&range).unwrap();
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.column_names(), vec!["column_0", "column_1"]);
    }

    #[test]
    fn test_range_to_dataframe_max_rows() {
        let mut range = Range::new((0, 0), (3, 0));
        range.set_value((0, 0), Data::String("x".into()));
        range.set_value((1, 0), Data::Int(1));
        range.set_value((2, 0), Data::Int(2));
        range.set_value((3, 0), Data::Int(3));

        let builder = ExcelReaderBuilder::new().max_rows(2);
        let df = builder.range_to_dataframe(&range).unwrap();
        assert_eq!(df.nrows(), 2);
    }
}
