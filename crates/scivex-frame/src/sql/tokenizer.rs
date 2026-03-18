//! SQL lexer — splits an input string into a stream of [`Token`]s.

use crate::error::{FrameError, Result};

/// A single token produced by the SQL lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Select,
    From,
    Where,
    Group,
    By,
    Having,
    Order,
    Limit,
    As,
    On,
    Join,
    Inner,
    Left,
    Right,
    Outer,
    And,
    Or,
    Not,
    Asc,
    Desc,
    Null,
    Is,
    In,
    // Aggregate function names (also keywords)
    Sum,
    Avg,
    Min,
    Max,
    Count,
    // Literals
    Integer(i64),
    Float(f64),
    StringLit(String),
    // Identifiers
    Ident(String),
    // Operators & punctuation
    Star,
    Comma,
    Dot,
    LParen,
    RParen,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Slash,
    Eof,
}

/// Tokenize an SQL string into a `Vec<Token>`.
#[allow(clippy::too_many_lines)]
pub fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];

        // Skip whitespace
        if ch.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Single-line comment: -- ...
        if ch == '-' && i + 1 < len && chars[i + 1] == '-' {
            while i < len && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        // String literal: 'value'
        if ch == '\'' {
            i += 1;
            let mut s = String::new();
            while i < len && chars[i] != '\'' {
                // Handle escaped single quote ('')
                if chars[i] == '\'' && i + 1 < len && chars[i + 1] == '\'' {
                    s.push('\'');
                    i += 2;
                } else {
                    s.push(chars[i]);
                    i += 1;
                }
            }
            if i >= len {
                return Err(FrameError::InvalidValue {
                    reason: "unterminated string literal".to_string(),
                });
            }
            i += 1; // closing quote
            tokens.push(Token::StringLit(s));
            continue;
        }

        // Quoted identifier: "col name"
        if ch == '"' {
            i += 1;
            let mut s = String::new();
            while i < len && chars[i] != '"' {
                s.push(chars[i]);
                i += 1;
            }
            if i >= len {
                return Err(FrameError::InvalidValue {
                    reason: "unterminated quoted identifier".to_string(),
                });
            }
            i += 1; // closing quote
            tokens.push(Token::Ident(s));
            continue;
        }

        // Numbers: integer or float
        if ch.is_ascii_digit() {
            let start = i;
            while i < len && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i < len && chars[i] == '.' && i + 1 < len && chars[i + 1].is_ascii_digit() {
                i += 1; // skip dot
                while i < len && chars[i].is_ascii_digit() {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                let val: f64 = s.parse().map_err(|_| FrameError::InvalidValue {
                    reason: format!("invalid float literal: {s}"),
                })?;
                tokens.push(Token::Float(val));
            } else {
                let s: String = chars[start..i].iter().collect();
                let val: i64 = s.parse().map_err(|_| FrameError::InvalidValue {
                    reason: format!("invalid integer literal: {s}"),
                })?;
                tokens.push(Token::Integer(val));
            }
            continue;
        }

        // Identifiers and keywords
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = i;
            while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let token = match word.to_uppercase().as_str() {
                "SELECT" => Token::Select,
                "FROM" => Token::From,
                "WHERE" => Token::Where,
                "GROUP" => Token::Group,
                "BY" => Token::By,
                "HAVING" => Token::Having,
                "ORDER" => Token::Order,
                "LIMIT" => Token::Limit,
                "AS" => Token::As,
                "ON" => Token::On,
                "JOIN" => Token::Join,
                "INNER" => Token::Inner,
                "LEFT" => Token::Left,
                "RIGHT" => Token::Right,
                "OUTER" => Token::Outer,
                "AND" => Token::And,
                "OR" => Token::Or,
                "NOT" => Token::Not,
                "ASC" => Token::Asc,
                "DESC" => Token::Desc,
                "NULL" => Token::Null,
                "IS" => Token::Is,
                "IN" => Token::In,
                "SUM" => Token::Sum,
                "AVG" => Token::Avg,
                "MIN" => Token::Min,
                "MAX" => Token::Max,
                "COUNT" => Token::Count,
                _ => Token::Ident(word),
            };
            tokens.push(token);
            continue;
        }

        // Operators and punctuation
        match ch {
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            '.' => {
                tokens.push(Token::Dot);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '=' => {
                tokens.push(Token::Eq);
                i += 1;
            }
            '!' => {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(Token::NotEq);
                    i += 2;
                } else {
                    return Err(FrameError::InvalidValue {
                        reason: format!("unexpected character: '{ch}'"),
                    });
                }
            }
            '<' => {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(Token::LtEq);
                    i += 2;
                } else if i + 1 < len && chars[i + 1] == '>' {
                    tokens.push(Token::NotEq);
                    i += 2;
                } else {
                    tokens.push(Token::Lt);
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(Token::GtEq);
                    i += 2;
                } else {
                    tokens.push(Token::Gt);
                    i += 1;
                }
            }
            _ => {
                return Err(FrameError::InvalidValue {
                    reason: format!("unexpected character: '{ch}'"),
                });
            }
        }
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select() {
        let tokens = tokenize("SELECT * FROM t").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Select,
                Token::Star,
                Token::From,
                Token::Ident("t".into()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_string_literal() {
        let tokens = tokenize("WHERE name = 'hello'").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Where,
                Token::Ident("name".into()),
                Token::Eq,
                Token::StringLit("hello".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_numbers() {
        let tokens = tokenize("42 2.78").unwrap();
        assert_eq!(
            tokens,
            vec![Token::Integer(42), Token::Float(2.78), Token::Eof]
        );
    }

    #[test]
    fn test_operators() {
        let tokens = tokenize("> >= < <= = != <>").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Gt,
                Token::GtEq,
                Token::Lt,
                Token::LtEq,
                Token::Eq,
                Token::NotEq,
                Token::NotEq,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_case_insensitive_keywords() {
        let tokens = tokenize("select FROM Where").unwrap();
        assert_eq!(
            tokens,
            vec![Token::Select, Token::From, Token::Where, Token::Eof]
        );
    }

    #[test]
    fn test_quoted_identifier() {
        let tokens = tokenize("\"col name\"").unwrap();
        assert_eq!(tokens, vec![Token::Ident("col name".into()), Token::Eof]);
    }

    #[test]
    fn test_unterminated_string() {
        assert!(tokenize("'hello").is_err());
    }
}
