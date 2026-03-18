//! Lightweight LaTeX-like math text parser.
//!
//! Converts simple math notation (`$...$`) into [`LatexSegment`]s that renderers
//! can translate into styled text (e.g. SVG `<tspan>` elements).
//!
//! # Supported Syntax
//!
//! - Greek letters: `\alpha`, `\beta`, `\gamma`, … (full set)
//! - Superscripts: `x^2`, `x^{10}`
//! - Subscripts: `x_i`, `x_{ij}`
//! - Fractions: `\frac{a}{b}` (rendered as `a/b` in inline mode)
//! - Common symbols: `\pm`, `\cdot`, `\times`, `\infty`, `\sqrt`, `\sum`, `\int`, `\partial`
//! - Plain math-italic text for single letters

/// A segment of parsed LaTeX-like text.
#[derive(Debug, Clone, PartialEq)]
pub enum LatexSegment {
    /// Normal (non-math) text.
    Plain(String),
    /// Math-mode text rendered in italic.
    MathText(String),
    /// A superscript segment.
    Superscript(String),
    /// A subscript segment.
    Subscript(String),
    /// A Unicode symbol (result of `\alpha` etc.).
    Symbol(char),
    /// An inline fraction `a/b`.
    Fraction(String, String),
}

/// Parse a string that may contain `$...$` math regions into segments.
///
/// Text outside `$...$` is returned as [`LatexSegment::Plain`].
/// Text inside is parsed for LaTeX commands and structures.
pub fn parse(input: &str) -> Vec<LatexSegment> {
    let mut segments = Vec::new();
    let mut rest = input;

    loop {
        if let Some(start) = rest.find('$') {
            // Plain text before the $.
            if start > 0 {
                segments.push(LatexSegment::Plain(rest[..start].to_string()));
            }
            let after_start = &rest[start + 1..];
            if let Some(end) = after_start.find('$') {
                let math = &after_start[..end];
                parse_math(math, &mut segments);
                rest = &after_start[end + 1..];
            } else {
                // Unmatched $ — treat rest as plain.
                segments.push(LatexSegment::Plain(rest[start..].to_string()));
                break;
            }
        } else {
            if !rest.is_empty() {
                segments.push(LatexSegment::Plain(rest.to_string()));
            }
            break;
        }
    }

    segments
}

/// Check if a string contains any LaTeX math (`$...$`).
#[must_use]
pub fn contains_math(s: &str) -> bool {
    let first = s.find('$');
    if let Some(i) = first {
        s[i + 1..].contains('$')
    } else {
        false
    }
}

/// Render parsed segments back into a plain Unicode string (no styling info).
///
/// This is suitable for terminal output or backends that don't support rich text.
#[must_use]
pub fn to_unicode(segments: &[LatexSegment]) -> String {
    let mut out = String::new();
    for seg in segments {
        match seg {
            LatexSegment::Plain(s) | LatexSegment::MathText(s) => out.push_str(s),
            LatexSegment::Superscript(s) => {
                for ch in s.chars() {
                    out.push(superscript_char(ch));
                }
            }
            LatexSegment::Subscript(s) => {
                for ch in s.chars() {
                    out.push(subscript_char(ch));
                }
            }
            LatexSegment::Symbol(c) => out.push(*c),
            LatexSegment::Fraction(num, den) => {
                out.push_str(num);
                out.push('/');
                out.push_str(den);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Internal parsing
// ---------------------------------------------------------------------------

fn parse_math(math: &str, segments: &mut Vec<LatexSegment>) {
    let chars: Vec<char> = math.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut buf = String::new();

    while i < len {
        match chars[i] {
            '\\' => {
                // Flush buffer.
                if !buf.is_empty() {
                    segments.push(LatexSegment::MathText(buf.clone()));
                    buf.clear();
                }
                i += 1;
                let cmd = read_command(&chars, &mut i);
                if cmd == "frac" {
                    let num = read_braced_group(&chars, &mut i);
                    let den = read_braced_group(&chars, &mut i);
                    segments.push(LatexSegment::Fraction(num, den));
                } else if let Some(sym) = greek_or_symbol(&cmd) {
                    segments.push(LatexSegment::Symbol(sym));
                } else {
                    // Unknown command — emit as text.
                    segments.push(LatexSegment::MathText(format!("\\{cmd}")));
                }
            }
            '^' => {
                if !buf.is_empty() {
                    segments.push(LatexSegment::MathText(buf.clone()));
                    buf.clear();
                }
                i += 1;
                let sup = read_group_or_char(&chars, &mut i);
                segments.push(LatexSegment::Superscript(sup));
            }
            '_' => {
                if !buf.is_empty() {
                    segments.push(LatexSegment::MathText(buf.clone()));
                    buf.clear();
                }
                i += 1;
                let sub = read_group_or_char(&chars, &mut i);
                segments.push(LatexSegment::Subscript(sub));
            }
            ' ' => {
                // Skip whitespace in math mode (LaTeX behavior).
                i += 1;
            }
            ch => {
                buf.push(ch);
                i += 1;
            }
        }
    }

    if !buf.is_empty() {
        segments.push(LatexSegment::MathText(buf));
    }
}

fn read_command(chars: &[char], i: &mut usize) -> String {
    let mut cmd = String::new();
    while *i < chars.len() && chars[*i].is_ascii_alphabetic() {
        cmd.push(chars[*i]);
        *i += 1;
    }
    cmd
}

fn read_braced_group(chars: &[char], i: &mut usize) -> String {
    // Skip whitespace.
    while *i < chars.len() && chars[*i] == ' ' {
        *i += 1;
    }
    if *i < chars.len() && chars[*i] == '{' {
        *i += 1;
        let mut depth = 1;
        let mut group = String::new();
        while *i < chars.len() && depth > 0 {
            if chars[*i] == '{' {
                depth += 1;
            } else if chars[*i] == '}' {
                depth -= 1;
                if depth == 0 {
                    *i += 1;
                    return group;
                }
            }
            group.push(chars[*i]);
            *i += 1;
        }
        group
    } else {
        // Single char fallback.
        read_group_or_char(chars, i)
    }
}

fn read_group_or_char(chars: &[char], i: &mut usize) -> String {
    if *i >= chars.len() {
        return String::new();
    }
    if chars[*i] == '{' {
        *i += 1;
        let mut depth = 1;
        let mut group = String::new();
        while *i < chars.len() && depth > 0 {
            if chars[*i] == '{' {
                depth += 1;
            } else if chars[*i] == '}' {
                depth -= 1;
                if depth == 0 {
                    *i += 1;
                    return group;
                }
            }
            group.push(chars[*i]);
            *i += 1;
        }
        group
    } else {
        let ch = chars[*i];
        *i += 1;
        ch.to_string()
    }
}

fn greek_or_symbol(cmd: &str) -> Option<char> {
    Some(match cmd {
        // Lowercase Greek.
        "alpha" => '\u{03B1}',
        "beta" => '\u{03B2}',
        "gamma" => '\u{03B3}',
        "delta" => '\u{03B4}',
        "epsilon" => '\u{03B5}',
        "zeta" => '\u{03B6}',
        "eta" => '\u{03B7}',
        "theta" => '\u{03B8}',
        "iota" => '\u{03B9}',
        "kappa" => '\u{03BA}',
        "lambda" => '\u{03BB}',
        "mu" => '\u{03BC}',
        "nu" => '\u{03BD}',
        "xi" => '\u{03BE}',
        "pi" => '\u{03C0}',
        "rho" => '\u{03C1}',
        "sigma" => '\u{03C3}',
        "tau" => '\u{03C4}',
        "upsilon" => '\u{03C5}',
        "phi" => '\u{03C6}',
        "chi" => '\u{03C7}',
        "psi" => '\u{03C8}',
        "omega" => '\u{03C9}',
        // Uppercase Greek.
        "Alpha" => '\u{0391}',
        "Beta" => '\u{0392}',
        "Gamma" => '\u{0393}',
        "Delta" => '\u{0394}',
        "Epsilon" => '\u{0395}',
        "Theta" => '\u{0398}',
        "Lambda" => '\u{039B}',
        "Pi" => '\u{03A0}',
        "Sigma" => '\u{03A3}',
        "Phi" => '\u{03A6}',
        "Psi" => '\u{03A8}',
        "Omega" => '\u{03A9}',
        // Common math symbols.
        "pm" => '\u{00B1}',
        "mp" => '\u{2213}',
        "times" => '\u{00D7}',
        "div" => '\u{00F7}',
        "cdot" => '\u{22C5}',
        "infty" => '\u{221E}',
        "sqrt" => '\u{221A}',
        "sum" => '\u{2211}',
        "prod" => '\u{220F}',
        "int" => '\u{222B}',
        "partial" => '\u{2202}',
        "nabla" => '\u{2207}',
        "approx" => '\u{2248}',
        "neq" => '\u{2260}',
        "leq" => '\u{2264}',
        "geq" => '\u{2265}',
        "rightarrow" => '\u{2192}',
        "leftarrow" => '\u{2190}',
        "in" => '\u{2208}',
        "forall" => '\u{2200}',
        "exists" => '\u{2203}',
        "degree" => '\u{00B0}',
        _ => return None,
    })
}

fn superscript_char(ch: char) -> char {
    match ch {
        '0' => '\u{2070}',
        '1' => '\u{00B9}',
        '2' => '\u{00B2}',
        '3' => '\u{00B3}',
        '4' => '\u{2074}',
        '5' => '\u{2075}',
        '6' => '\u{2076}',
        '7' => '\u{2077}',
        '8' => '\u{2078}',
        '9' => '\u{2079}',
        '+' => '\u{207A}',
        '-' => '\u{207B}',
        '=' => '\u{207C}',
        '(' => '\u{207D}',
        ')' => '\u{207E}',
        'n' => '\u{207F}',
        'i' => '\u{2071}',
        _ => ch,
    }
}

fn subscript_char(ch: char) -> char {
    match ch {
        '0' => '\u{2080}',
        '1' => '\u{2081}',
        '2' => '\u{2082}',
        '3' => '\u{2083}',
        '4' => '\u{2084}',
        '5' => '\u{2085}',
        '6' => '\u{2086}',
        '7' => '\u{2087}',
        '8' => '\u{2088}',
        '9' => '\u{2089}',
        '+' => '\u{208A}',
        '-' => '\u{208B}',
        '=' => '\u{208C}',
        '(' => '\u{208D}',
        ')' => '\u{208E}',
        'a' => '\u{2090}',
        'e' => '\u{2091}',
        'o' => '\u{2092}',
        'x' => '\u{2093}',
        'i' => '\u{1D62}',
        'j' => '\u{2C7C}',
        'k' => '\u{2096}',
        'n' => '\u{2099}',
        _ => ch,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_no_math() {
        let segs = parse("Hello World");
        assert_eq!(segs, vec![LatexSegment::Plain("Hello World".into())]);
    }

    #[test]
    fn simple_math_variable() {
        let segs = parse("$x$");
        assert_eq!(segs, vec![LatexSegment::MathText("x".into())]);
    }

    #[test]
    fn superscript_simple() {
        let segs = parse("$x^2$");
        assert_eq!(
            segs,
            vec![
                LatexSegment::MathText("x".into()),
                LatexSegment::Superscript("2".into()),
            ]
        );
    }

    #[test]
    fn superscript_braced() {
        let segs = parse("$x^{10}$");
        assert_eq!(
            segs,
            vec![
                LatexSegment::MathText("x".into()),
                LatexSegment::Superscript("10".into()),
            ]
        );
    }

    #[test]
    fn subscript_simple() {
        let segs = parse("$x_i$");
        assert_eq!(
            segs,
            vec![
                LatexSegment::MathText("x".into()),
                LatexSegment::Subscript("i".into()),
            ]
        );
    }

    #[test]
    fn greek_letter() {
        let segs = parse("$\\alpha$");
        assert_eq!(segs, vec![LatexSegment::Symbol('\u{03B1}')]);
    }

    #[test]
    fn mixed_text_and_math() {
        let segs = parse("Energy: $E = mc^2$");
        assert_eq!(
            segs,
            vec![
                LatexSegment::Plain("Energy: ".into()),
                LatexSegment::MathText("E=mc".into()),
                LatexSegment::Superscript("2".into()),
            ]
        );
    }

    #[test]
    fn fraction() {
        let segs = parse("$\\frac{a}{b}$");
        assert_eq!(segs, vec![LatexSegment::Fraction("a".into(), "b".into())]);
    }

    #[test]
    fn contains_math_true() {
        assert!(contains_math("hello $x^2$ world"));
    }

    #[test]
    fn contains_math_false() {
        assert!(!contains_math("no math here"));
        assert!(!contains_math("only one $"));
    }

    #[test]
    fn to_unicode_superscript() {
        let segs = parse("$x^2$");
        let u = to_unicode(&segs);
        assert_eq!(u, "x\u{00B2}");
    }

    #[test]
    fn to_unicode_subscript() {
        let segs = parse("$x_0$");
        let u = to_unicode(&segs);
        assert_eq!(u, "x\u{2080}");
    }

    #[test]
    fn to_unicode_greek() {
        let segs = parse("$\\alpha + \\beta$");
        let u = to_unicode(&segs);
        assert_eq!(u, "\u{03B1}+\u{03B2}");
    }

    #[test]
    fn to_unicode_fraction() {
        let segs = parse("$\\frac{1}{2}$");
        let u = to_unicode(&segs);
        assert_eq!(u, "1/2");
    }

    #[test]
    fn math_symbols() {
        let segs = parse("$\\pm \\infty$");
        assert_eq!(
            segs,
            vec![
                LatexSegment::Symbol('\u{00B1}'),
                LatexSegment::Symbol('\u{221E}'),
            ]
        );
    }

    #[test]
    fn complex_expression() {
        let segs = parse("$\\sigma^{2}_{i}$");
        assert_eq!(
            segs,
            vec![
                LatexSegment::Symbol('\u{03C3}'),
                LatexSegment::Superscript("2".into()),
                LatexSegment::Subscript("i".into()),
            ]
        );
    }

    #[test]
    fn multiple_math_regions() {
        let segs = parse("$x$ and $y$");
        assert_eq!(
            segs,
            vec![
                LatexSegment::MathText("x".into()),
                LatexSegment::Plain(" and ".into()),
                LatexSegment::MathText("y".into()),
            ]
        );
    }
}
