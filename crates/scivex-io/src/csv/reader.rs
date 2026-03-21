//! CSV reader that produces a [`DataFrame`].

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use scivex_frame::DataFrame;

use super::parser::RecordParser;
use crate::common::{InferredType, build_series_from_strings, infer_column_type};
use crate::error::{IoError, Result};

/// Builder for reading CSV data into a [`DataFrame`].
///
/// # Examples
///
/// ```
/// # use scivex_io::csv::CsvReaderBuilder;
/// let csv = "name,age\nAlice,30\nBob,25\n";
/// let df = CsvReaderBuilder::new()
///     .has_header(true)
///     .read(csv.as_bytes())
///     .unwrap();
/// assert_eq!(df.nrows(), 2);
/// assert_eq!(df.ncols(), 2);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct CsvReaderBuilder {
    delimiter: u8,
    quote_char: u8,
    has_header: bool,
    skip_rows: usize,
    max_rows: Option<usize>,
    null_values: Vec<String>,
    column_names: Option<Vec<String>>,
    column_types: Vec<(String, InferredType)>,
    infer_sample_size: usize,
    comment_char: Option<u8>,
    trim_whitespace: bool,
}

impl Default for CsvReaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CsvReaderBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            delimiter: b',',
            quote_char: b'"',
            has_header: true,
            skip_rows: 0,
            max_rows: None,
            null_values: Vec::new(),
            column_names: None,
            column_types: Vec::new(),
            infer_sample_size: 1000,
            comment_char: None,
            trim_whitespace: false,
        }
    }

    /// Set the field delimiter (default: `b','`).
    pub fn delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set the quote character (default: `b'"'`).
    pub fn quote_char(mut self, quote_char: u8) -> Self {
        self.quote_char = quote_char;
        self
    }

    /// Whether the first non-skipped row is a header row (default: `true`).
    pub fn has_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Number of rows to skip before reading (default: `0`).
    pub fn skip_rows(mut self, n: usize) -> Self {
        self.skip_rows = n;
        self
    }

    /// Maximum number of data rows to read.
    pub fn max_rows(mut self, n: Option<usize>) -> Self {
        self.max_rows = n;
        self
    }

    /// Additional strings to treat as null values.
    pub fn null_values(mut self, vals: Vec<String>) -> Self {
        self.null_values = vals;
        self
    }

    /// Explicit column names (overrides header row).
    pub fn column_names(mut self, names: Vec<String>) -> Self {
        self.column_names = Some(names);
        self
    }

    /// Explicit column types (overrides inference for named columns).
    pub fn column_types(mut self, types: Vec<(String, InferredType)>) -> Self {
        self.column_types = types;
        self
    }

    /// How many non-null values to sample when inferring types (default: 1000).
    pub fn infer_sample_size(mut self, n: usize) -> Self {
        self.infer_sample_size = n;
        self
    }

    /// Lines starting with this byte are skipped as comments.
    pub fn comment_char(mut self, ch: Option<u8>) -> Self {
        self.comment_char = ch;
        self
    }

    /// Whether to trim leading/trailing whitespace from fields.
    pub fn trim_whitespace(mut self, trim: bool) -> Self {
        self.trim_whitespace = trim;
        self
    }

    /// Read CSV from any [`Read`] implementation.
    pub fn read<R: Read>(self, reader: R) -> Result<DataFrame> {
        let buf = BufReader::new(reader);
        self.read_buffered(buf)
    }

    /// Read CSV from a file path.
    pub fn read_path<P: AsRef<Path>>(self, path: P) -> Result<DataFrame> {
        let file = std::fs::File::open(path)?;
        self.read(file)
    }

    fn read_buffered<R: BufRead>(self, reader: R) -> Result<DataFrame> {
        let raw_lines = self.read_raw_lines(reader)?;
        let (mut column_names, records) = self.parse_lines(&raw_lines)?;
        self.build_dataframe(&mut column_names, &records)
    }

    fn read_raw_lines<R: BufRead>(&self, reader: R) -> Result<Vec<String>> {
        let mut lines = reader.lines();
        for _ in 0..self.skip_rows {
            if lines.next().is_none() {
                return Err(IoError::EmptyInput);
            }
        }
        let mut raw_lines: Vec<String> = Vec::new();
        for line_result in lines {
            let line = line_result?;
            if self
                .comment_char
                .is_some_and(|cc| line.as_bytes().first() == Some(&cc))
            {
                continue;
            }
            raw_lines.push(line);
        }
        if raw_lines.is_empty() {
            return Err(IoError::EmptyInput);
        }
        Ok(raw_lines)
    }

    fn parse_lines(&self, raw_lines: &[String]) -> Result<(Vec<String>, Vec<Vec<String>>)> {
        let (header_names, data_start) = if self.has_header {
            let names: Vec<String> =
                super::parser::parse_record(&raw_lines[0], self.delimiter, self.quote_char)
                    .into_iter()
                    .map(|s| {
                        if self.trim_whitespace {
                            s.trim().to_string()
                        } else {
                            s
                        }
                    })
                    .collect();
            if names.is_empty() {
                return Err(IoError::InvalidHeader {
                    reason: "header row is empty".to_string(),
                });
            }
            (names, 1)
        } else {
            (Vec::new(), 0)
        };

        let column_names = self.column_names.clone().unwrap_or(header_names);
        let mut parser = RecordParser::new(self.delimiter, self.quote_char);
        let mut records: Vec<Vec<String>> = Vec::new();

        for line in &raw_lines[data_start..] {
            if self.max_rows.is_some_and(|max| records.len() >= max) {
                break;
            }
            if let Some(fields) = parser.feed_line(line) {
                let fields = if self.trim_whitespace {
                    fields.into_iter().map(|s| s.trim().to_string()).collect()
                } else {
                    fields
                };
                records.push(fields);
            }
        }
        #[allow(clippy::collapsible_if)]
        if let Some(fields) = parser.finish() {
            if self.max_rows.is_none_or(|max| records.len() < max) {
                records.push(fields);
            }
        }
        if records.is_empty() {
            return Err(IoError::EmptyInput);
        }
        Ok((column_names, records))
    }

    fn build_dataframe(
        &self,
        column_names: &mut Vec<String>,
        records: &[Vec<String>],
    ) -> Result<DataFrame> {
        let ncols = if column_names.is_empty() {
            records.iter().map(Vec::len).max().unwrap_or(0)
        } else {
            column_names.len()
        };
        if column_names.is_empty() {
            *column_names = (0..ncols).map(|i| format!("column_{i}")).collect();
        }

        let mut columns: Vec<Vec<String>> = vec![Vec::with_capacity(records.len()); ncols];
        for record in records {
            for (col_idx, col) in columns.iter_mut().enumerate() {
                col.push(record.get(col_idx).cloned().unwrap_or_default());
            }
        }

        let type_overrides: std::collections::HashMap<&str, InferredType> = self
            .column_types
            .iter()
            .map(|(name, ty)| (name.as_str(), *ty))
            .collect();

        let mut series_vec: Vec<Box<dyn scivex_frame::AnySeries>> = Vec::with_capacity(ncols);
        for (col_idx, col_data) in columns.iter().enumerate() {
            let col_name = &column_names[col_idx];
            let col_data: Vec<String> = col_data
                .iter()
                .map(|v| {
                    if self.null_values.iter().any(|n| n == v) {
                        String::new()
                    } else {
                        v.clone()
                    }
                })
                .collect();

            let dtype = if let Some(&forced) = type_overrides.get(col_name.as_str()) {
                forced
            } else {
                let sample: Vec<&str> = col_data
                    .iter()
                    .map(String::as_str)
                    .take(self.infer_sample_size)
                    .collect();
                infer_column_type(&sample)
            };
            series_vec.push(build_series_from_strings(col_name, &col_data, dtype)?);
        }

        Ok(DataFrame::new(series_vec)?)
    }
}

/// Read CSV from a reader with default settings.
///
/// # Examples
///
/// ```
/// # use scivex_io::csv::read_csv;
/// let csv = "x,y\n1,2\n3,4\n";
/// let df = read_csv(csv.as_bytes()).unwrap();
/// assert_eq!(df.nrows(), 2);
/// assert_eq!(df.column_names(), vec!["x", "y"]);
/// ```
pub fn read_csv<R: Read>(reader: R) -> Result<DataFrame> {
    CsvReaderBuilder::new().read(reader)
}

/// Read CSV from a file path with default settings.
///
/// # Examples
///
/// ```ignore
/// use scivex_io::csv::read_csv_path;
/// let df = read_csv_path("data.csv").unwrap();
/// assert!(df.nrows() > 0);
/// ```
pub fn read_csv_path<P: AsRef<Path>>(path: P) -> Result<DataFrame> {
    CsvReaderBuilder::new().read_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_frame::{DType, Series, StringSeries};

    #[test]
    fn test_read_csv_basic() {
        let csv = "name,age,score\nAlice,30,95.5\nBob,25,87.0\n";
        let df = read_csv(csv.as_bytes()).unwrap();
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.ncols(), 3);
        assert_eq!(df.column_names(), vec!["name", "age", "score"]);
    }

    #[test]
    fn test_read_csv_types() {
        let csv = "name,age,score\nAlice,30,95.5\nBob,25,87.0\n";
        let df = read_csv(csv.as_bytes()).unwrap();

        assert_eq!(df.column("name").unwrap().dtype(), DType::Str);
        assert_eq!(df.column("age").unwrap().dtype(), DType::I64);
        assert_eq!(df.column("score").unwrap().dtype(), DType::F64);
    }

    #[test]
    fn test_read_csv_with_nulls() {
        let csv = "a,b\n1,hello\nNA,world\n3,\n";
        let df = read_csv(csv.as_bytes()).unwrap();

        let col_a = df.column("a").unwrap();
        assert!(col_a.is_null(1)); // "NA" is null
        assert_eq!(col_a.display_value(0), "1");

        let col_b = df.column("b").unwrap();
        assert!(col_b.is_null(2)); // "" is null
    }

    #[test]
    fn test_read_csv_no_header() {
        let csv = "1,2,3\n4,5,6\n";
        let df = CsvReaderBuilder::new()
            .has_header(false)
            .read(csv.as_bytes())
            .unwrap();
        assert_eq!(df.ncols(), 3);
        assert_eq!(df.column_names(), vec!["column_0", "column_1", "column_2"]);
    }

    #[test]
    fn test_read_csv_custom_delimiter() {
        let csv = "a\tb\n1\t2\n3\t4\n";
        let df = CsvReaderBuilder::new()
            .delimiter(b'\t')
            .read(csv.as_bytes())
            .unwrap();
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.ncols(), 2);
    }

    #[test]
    fn test_read_csv_skip_rows() {
        let csv = "# comment\n# another\nname,val\nAlice,1\n";
        let df = CsvReaderBuilder::new()
            .skip_rows(2)
            .read(csv.as_bytes())
            .unwrap();
        assert_eq!(df.nrows(), 1);
    }

    #[test]
    fn test_read_csv_max_rows() {
        let csv = "x\n1\n2\n3\n4\n5\n";
        let df = CsvReaderBuilder::new()
            .max_rows(Some(3))
            .read(csv.as_bytes())
            .unwrap();
        assert_eq!(df.nrows(), 3);
    }

    #[test]
    fn test_read_csv_comment_char() {
        let csv = "x\n1\n# skip me\n2\n";
        let df = CsvReaderBuilder::new()
            .comment_char(Some(b'#'))
            .read(csv.as_bytes())
            .unwrap();
        assert_eq!(df.nrows(), 2);
    }

    #[test]
    fn test_read_csv_quoted_fields() {
        let csv = "a,b\n\"hello, world\",1\n\"test\",2\n";
        let df = read_csv(csv.as_bytes()).unwrap();
        let col_a = df.column("a").unwrap();
        assert_eq!(col_a.display_value(0), "hello, world");
    }

    #[test]
    fn test_read_csv_trim_whitespace() {
        let csv = " name , age \n Alice , 30 \n Bob , 25 \n";
        let df = CsvReaderBuilder::new()
            .trim_whitespace(true)
            .read(csv.as_bytes())
            .unwrap();
        assert_eq!(df.column_names(), vec!["name", "age"]);
        let name_col = df.column("name").unwrap();
        assert_eq!(name_col.display_value(0), "Alice");
    }

    #[test]
    fn test_read_csv_explicit_column_names() {
        let csv = "1,2\n3,4\n";
        let df = CsvReaderBuilder::new()
            .has_header(false)
            .column_names(vec!["x".into(), "y".into()])
            .read(csv.as_bytes())
            .unwrap();
        assert_eq!(df.column_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_read_csv_explicit_column_types() {
        let csv = "a,b\n1,2\n3,4\n";
        let df = CsvReaderBuilder::new()
            .column_types(vec![("a".into(), InferredType::F64)])
            .read(csv.as_bytes())
            .unwrap();
        assert_eq!(df.column("a").unwrap().dtype(), DType::F64);
    }

    #[test]
    fn test_read_csv_empty_input() {
        let csv = "";
        let result = read_csv(csv.as_bytes());
        assert!(result.is_err());
    }

    #[test]
    fn test_read_csv_column_values() {
        let csv = "x\n10\n20\n30\n";
        let df = read_csv(csv.as_bytes()).unwrap();
        let col = df.column("x").unwrap();
        let typed = col.as_any().downcast_ref::<Series<i64>>().unwrap();
        assert_eq!(typed.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_read_csv_string_column_values() {
        let csv = "name\nAlice\nBob\n";
        let df = read_csv(csv.as_bytes()).unwrap();
        let col = df.column("name").unwrap();
        let typed = col.as_any().downcast_ref::<StringSeries>().unwrap();
        assert_eq!(typed.get(0), Some("Alice"));
        assert_eq!(typed.get(1), Some("Bob"));
    }
}
