//! CSV writer that serialises a [`DataFrame`] to CSV format.

use std::io::Write;
use std::path::Path;

use scivex_frame::DataFrame;

use crate::error::Result;

/// Controls when fields are quoted in CSV output.
///
/// # Examples
///
/// ```
/// use scivex_io::csv::QuoteStyle;
/// assert_eq!(QuoteStyle::Necessary, QuoteStyle::Necessary);
/// assert_ne!(QuoteStyle::Always, QuoteStyle::Never);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteStyle {
    /// Only quote fields that contain the delimiter, quote character, or
    /// newlines.
    Necessary,
    /// Always quote every field.
    Always,
    /// Never quote (caller is responsible for data not containing
    /// delimiters).
    Never,
}

/// Builder for writing a [`DataFrame`] as CSV.
///
/// # Examples
///
/// ```
/// # use scivex_io::csv::{CsvWriterBuilder, read_csv};
/// let csv = "x,y\n1,2\n3,4\n";
/// let df = read_csv(csv.as_bytes()).unwrap();
/// let mut buf = Vec::new();
/// CsvWriterBuilder::new()
///     .write(&mut buf, &df)
///     .unwrap();
/// let output = String::from_utf8(buf).unwrap();
/// assert!(output.contains("x,y"));
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct CsvWriterBuilder {
    delimiter: u8,
    write_header: bool,
    null_representation: String,
    quote_style: QuoteStyle,
    quote_char: u8,
}

impl Default for CsvWriterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CsvWriterBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            delimiter: b',',
            write_header: true,
            null_representation: String::new(),
            quote_style: QuoteStyle::Necessary,
            quote_char: b'"',
        }
    }

    /// Set the field delimiter (default: `b','`).
    pub fn delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Whether to write a header row (default: `true`).
    pub fn write_header(mut self, write_header: bool) -> Self {
        self.write_header = write_header;
        self
    }

    /// String to write for null values (default: `""`).
    pub fn null_representation(mut self, repr: String) -> Self {
        self.null_representation = repr;
        self
    }

    /// How to quote fields (default: [`QuoteStyle::Necessary`]).
    pub fn quote_style(mut self, style: QuoteStyle) -> Self {
        self.quote_style = style;
        self
    }

    /// Write the `DataFrame` as CSV to a writer.
    pub fn write<W: Write>(self, mut writer: W, df: &DataFrame) -> Result<()> {
        let delim = self.delimiter as char;
        let quote = self.quote_char as char;

        // Header.
        if self.write_header {
            let names = df.column_names();
            for (i, name) in names.iter().enumerate() {
                if i > 0 {
                    write!(writer, "{delim}")?;
                }
                self.write_field(&mut writer, name, delim, quote)?;
            }
            writeln!(writer)?;
        }

        // Data rows.
        let columns = df.columns();
        let nrows = df.nrows();
        let ncols = df.ncols();

        for row in 0..nrows {
            for (col_idx, col) in columns.iter().enumerate() {
                if col_idx > 0 {
                    write!(writer, "{delim}")?;
                }
                if col.is_null(row) {
                    self.write_field(&mut writer, &self.null_representation, delim, quote)?;
                } else {
                    let value = col.display_value(row);
                    self.write_field(&mut writer, &value, delim, quote)?;
                }
            }
            // Don't write trailing newline after the very last row if there
            // are columns — actually, always write newline for consistency.
            if ncols > 0 {
                writeln!(writer)?;
            }
        }

        Ok(())
    }

    /// Write the `DataFrame` as CSV to a file.
    pub fn write_path<P: AsRef<Path>>(self, path: P, df: &DataFrame) -> Result<()> {
        let file = std::fs::File::create(path)?;
        let buf = std::io::BufWriter::new(file);
        self.write(buf, df)
    }

    fn write_field<W: Write>(
        &self,
        writer: &mut W,
        value: &str,
        delim: char,
        quote: char,
    ) -> Result<()> {
        match self.quote_style {
            QuoteStyle::Always => {
                write!(writer, "{quote}")?;
                Self::write_escaped(writer, value, quote)?;
                write!(writer, "{quote}")?;
            }
            QuoteStyle::Never => {
                write!(writer, "{value}")?;
            }
            QuoteStyle::Necessary => {
                let needs_quoting = value.contains(delim)
                    || value.contains(quote)
                    || value.contains('\n')
                    || value.contains('\r');
                if needs_quoting {
                    write!(writer, "{quote}")?;
                    Self::write_escaped(writer, value, quote)?;
                    write!(writer, "{quote}")?;
                } else {
                    write!(writer, "{value}")?;
                }
            }
        }
        Ok(())
    }

    fn write_escaped<W: Write>(writer: &mut W, value: &str, quote: char) -> Result<()> {
        for ch in value.chars() {
            if ch == quote {
                write!(writer, "{quote}{quote}")?;
            } else {
                write!(writer, "{ch}")?;
            }
        }
        Ok(())
    }
}

/// Write a `DataFrame` as CSV to a writer with default settings.
///
/// # Examples
///
/// ```
/// # use scivex_io::csv::{write_csv, read_csv};
/// let df = read_csv("a,b\n1,2\n".as_bytes()).unwrap();
/// let mut buf = Vec::new();
/// write_csv(&mut buf, &df).unwrap();
/// let out = String::from_utf8(buf).unwrap();
/// assert!(out.starts_with("a,b\n"));
/// ```
pub fn write_csv<W: Write>(writer: W, df: &DataFrame) -> Result<()> {
    CsvWriterBuilder::new().write(writer, df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_frame::{DataFrame, Series, StringSeries};

    fn test_df() -> DataFrame {
        let name: Box<dyn scivex_frame::AnySeries> =
            Box::new(StringSeries::from_strs("name", &["Alice", "Bob"]));
        let age: Box<dyn scivex_frame::AnySeries> = Box::new(Series::new("age", vec![30_i64, 25]));
        let score: Box<dyn scivex_frame::AnySeries> =
            Box::new(Series::new("score", vec![95.5_f64, 87.0]));
        DataFrame::new(vec![name, age, score]).unwrap()
    }

    #[test]
    fn test_write_csv_basic() {
        let df = test_df();
        let mut buf = Vec::new();
        write_csv(&mut buf, &df).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.starts_with("name,age,score\n"));
        assert!(output.contains("Alice,30,95.5\n"));
        assert!(output.contains("Bob,25,87\n"));
    }

    #[test]
    fn test_write_csv_no_header() {
        let df = test_df();
        let mut buf = Vec::new();
        CsvWriterBuilder::new()
            .write_header(false)
            .write(&mut buf, &df)
            .unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(!output.contains("name"));
        assert!(output.starts_with("Alice,30,95.5\n"));
    }

    #[test]
    fn test_write_csv_always_quote() {
        let df = test_df();
        let mut buf = Vec::new();
        CsvWriterBuilder::new()
            .quote_style(QuoteStyle::Always)
            .write(&mut buf, &df)
            .unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("\"name\",\"age\",\"score\""));
        assert!(output.contains("\"Alice\",\"30\",\"95.5\""));
    }

    #[test]
    fn test_write_csv_tab_delimiter() {
        let df = test_df();
        let mut buf = Vec::new();
        CsvWriterBuilder::new()
            .delimiter(b'\t')
            .write(&mut buf, &df)
            .unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("name\tage\tscore"));
    }

    #[test]
    fn test_write_csv_null_representation() {
        let col: Box<dyn scivex_frame::AnySeries> =
            Box::new(Series::with_nulls("x", vec![1_i64, 0, 3], vec![false, true, false]).unwrap());
        let df = DataFrame::new(vec![col]).unwrap();
        let mut buf = Vec::new();
        CsvWriterBuilder::new()
            .null_representation("NA".to_string())
            .write(&mut buf, &df)
            .unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("NA"));
    }

    #[test]
    fn test_write_csv_quoting_special_chars() {
        let col: Box<dyn scivex_frame::AnySeries> =
            Box::new(StringSeries::from_strs("x", &["hello, world", "normal"]));
        let df = DataFrame::new(vec![col]).unwrap();
        let mut buf = Vec::new();
        write_csv(&mut buf, &df).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("\"hello, world\""));
    }

    #[test]
    fn test_roundtrip() {
        let csv_in = "name,value\nAlice,10\nBob,20\n";
        let df = crate::csv::read_csv(csv_in.as_bytes()).unwrap();
        let mut buf = Vec::new();
        write_csv(&mut buf, &df).unwrap();
        let csv_out = String::from_utf8(buf).unwrap();
        let df2 = crate::csv::read_csv(csv_out.as_bytes()).unwrap();
        assert_eq!(df.nrows(), df2.nrows());
        assert_eq!(df.ncols(), df2.ncols());
        assert_eq!(df.column_names(), df2.column_names());
    }
}
