//! RFC 4180 CSV parser implemented as a state machine.
//!
//! Handles quoted fields, escaped quotes (`""`), multi-line quoted fields,
//! and configurable delimiters / quote characters.

/// Parse state for the RFC 4180 state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// At the start of a new field.
    FieldStart,
    /// Inside an unquoted field.
    InUnquoted,
    /// Inside a quoted field.
    InQuoted,
    /// Just saw a quote inside a quoted field (could be end or escaped quote).
    AfterQuote,
}

/// Parse a single-line CSV record into fields.
///
/// This does **not** handle multi-line quoted fields — use [`RecordParser`]
/// for that.
pub fn parse_record(input: &str, delimiter: u8, quote_char: u8) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut state = State::FieldStart;
    let delim = delimiter as char;
    let quote = quote_char as char;

    for ch in input.chars() {
        match state {
            State::FieldStart => {
                if ch == quote {
                    state = State::InQuoted;
                } else if ch == delim {
                    fields.push(std::mem::take(&mut field));
                } else {
                    field.push(ch);
                    state = State::InUnquoted;
                }
            }
            State::InUnquoted => {
                if ch == delim {
                    fields.push(std::mem::take(&mut field));
                    state = State::FieldStart;
                } else {
                    field.push(ch);
                }
            }
            State::InQuoted => {
                if ch == quote {
                    state = State::AfterQuote;
                } else {
                    field.push(ch);
                }
            }
            State::AfterQuote => {
                if ch == quote {
                    // Escaped quote inside a quoted field.
                    field.push(quote);
                    state = State::InQuoted;
                } else if ch == delim {
                    fields.push(std::mem::take(&mut field));
                    state = State::FieldStart;
                } else {
                    // Unexpected character after closing quote — treat as
                    // unquoted content.
                    field.push(ch);
                    state = State::InUnquoted;
                }
            }
        }
    }

    // Push the last field.
    fields.push(field);
    fields
}

/// Streaming record parser that handles multi-line quoted fields.
///
/// Feed lines one at a time via [`feed_line`](Self::feed_line). When a
/// complete record has been assembled the method returns `Some(fields)`.
/// If the record spans multiple lines (because of a quoted field containing
/// a newline), `feed_line` returns `None` and you should keep feeding lines.
///
/// Call [`finish`](Self::finish) at EOF to flush any buffered partial record.
#[derive(Debug)]
pub struct RecordParser {
    delimiter: u8,
    quote_char: u8,
    fields: Vec<String>,
    current_field: String,
    state: State,
    /// Whether we are in the middle of a multi-line quoted field.
    in_multiline: bool,
}

impl RecordParser {
    /// Create a new parser with the given delimiter and quote character.
    pub fn new(delimiter: u8, quote_char: u8) -> Self {
        Self {
            delimiter,
            quote_char,
            fields: Vec::new(),
            current_field: String::new(),
            state: State::FieldStart,
            in_multiline: false,
        }
    }

    /// Feed a line of input.
    ///
    /// Returns `Some(fields)` when a complete record has been assembled,
    /// or `None` if we're in the middle of a multi-line quoted field and
    /// need more input.
    pub fn feed_line(&mut self, line: &str) -> Option<Vec<String>> {
        // If we're continuing a multiline quoted field, add the newline that
        // was stripped by the line iterator.
        if self.in_multiline {
            self.current_field.push('\n');
        }

        let delim = self.delimiter as char;
        let quote = self.quote_char as char;

        for ch in line.chars() {
            match self.state {
                State::FieldStart => {
                    if ch == quote {
                        self.state = State::InQuoted;
                    } else if ch == delim {
                        self.fields.push(std::mem::take(&mut self.current_field));
                    } else {
                        self.current_field.push(ch);
                        self.state = State::InUnquoted;
                    }
                }
                State::InUnquoted => {
                    if ch == delim {
                        self.fields.push(std::mem::take(&mut self.current_field));
                        self.state = State::FieldStart;
                    } else {
                        self.current_field.push(ch);
                    }
                }
                State::InQuoted => {
                    if ch == quote {
                        self.state = State::AfterQuote;
                    } else {
                        self.current_field.push(ch);
                    }
                }
                State::AfterQuote => {
                    if ch == quote {
                        self.current_field.push(quote);
                        self.state = State::InQuoted;
                    } else if ch == delim {
                        self.fields.push(std::mem::take(&mut self.current_field));
                        self.state = State::FieldStart;
                    } else {
                        self.current_field.push(ch);
                        self.state = State::InUnquoted;
                    }
                }
            }
        }

        // If we're inside a quoted field, we need more lines.
        if self.state == State::InQuoted {
            self.in_multiline = true;
            return None;
        }

        self.in_multiline = false;
        self.fields.push(std::mem::take(&mut self.current_field));
        self.state = State::FieldStart;
        Some(std::mem::take(&mut self.fields))
    }

    /// Flush any remaining buffered content at EOF.
    ///
    /// Returns `Some(fields)` if there was buffered data, `None` if the
    /// parser was already clean.
    pub fn finish(&mut self) -> Option<Vec<String>> {
        if self.fields.is_empty() && self.current_field.is_empty() && !self.in_multiline {
            return None;
        }
        self.fields.push(std::mem::take(&mut self.current_field));
        self.state = State::FieldStart;
        self.in_multiline = false;
        Some(std::mem::take(&mut self.fields))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_record() {
        let fields = parse_record("a,b,c", b',', b'"');
        assert_eq!(fields, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_quoted_field() {
        let fields = parse_record(r#"a,"hello, world",c"#, b',', b'"');
        assert_eq!(fields, vec!["a", "hello, world", "c"]);
    }

    #[test]
    fn test_escaped_quote() {
        let fields = parse_record(r#"a,"he said ""hi""",c"#, b',', b'"');
        assert_eq!(fields, vec!["a", r#"he said "hi""#, "c"]);
    }

    #[test]
    fn test_empty_fields() {
        let fields = parse_record("a,,c,", b',', b'"');
        assert_eq!(fields, vec!["a", "", "c", ""]);
    }

    #[test]
    fn test_tab_delimiter() {
        let fields = parse_record("a\tb\tc", b'\t', b'"');
        assert_eq!(fields, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_record_parser_single_line() {
        let mut parser = RecordParser::new(b',', b'"');
        let result = parser.feed_line("a,b,c");
        assert_eq!(result, Some(vec!["a".into(), "b".into(), "c".into()]));
    }

    #[test]
    fn test_record_parser_multiline() {
        let mut parser = RecordParser::new(b',', b'"');
        assert_eq!(parser.feed_line(r#"a,"hello"#), None);
        let result = parser.feed_line(r#"world",c"#);
        assert_eq!(
            result,
            Some(vec!["a".into(), "hello\nworld".into(), "c".into()])
        );
    }

    #[test]
    fn test_record_parser_finish() {
        let mut parser = RecordParser::new(b',', b'"');
        assert_eq!(parser.feed_line(r#"a,"unterminated"#), None);
        let result = parser.finish();
        assert!(result.is_some());
    }
}
