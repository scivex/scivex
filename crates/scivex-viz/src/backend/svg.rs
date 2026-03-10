use super::Renderer;
use crate::element::{Element, TextAnchor};
use crate::error::Result;
use crate::style::Font;
use std::fmt::Write;

/// Renders elements to SVG 1.1 markup.
#[derive(Debug, Clone)]
pub struct SvgBackend;

impl Renderer for SvgBackend {
    fn render(&self, elements: &[Element], width: f64, height: f64) -> Result<String> {
        let mut svg = String::with_capacity(4096);
        writeln!(
            svg,
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">"#,
        )
        .unwrap();

        for elem in elements {
            render_element(&mut svg, elem, 1);
        }

        svg.push_str("</svg>\n");
        Ok(svg)
    }
}

impl SvgBackend {
    /// Render elements to SVG and write to a file.
    pub fn to_file(&self, elements: &[Element], width: f64, height: f64, path: &str) -> Result<()> {
        let content = self.render(elements, width, height)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

fn indent(buf: &mut String, depth: usize) {
    for _ in 0..depth {
        buf.push_str("  ");
    }
}

fn render_line(buf: &mut String, elem: &Element, depth: usize) {
    if let Element::Line {
        x1,
        y1,
        x2,
        y2,
        stroke,
    } = elem
    {
        indent(buf, depth);
        write!(
            buf,
            r#"<line x1="{x1:.2}" y1="{y1:.2}" x2="{x2:.2}" y2="{y2:.2}" stroke="{}" stroke-width="{:.2}""#,
            stroke.color.to_svg_color(),
            stroke.width,
        )
        .unwrap();
        write_dash(buf, stroke.dash.as_ref());
        buf.push_str("/>\n");
    }
}

fn render_rect(buf: &mut String, elem: &Element, depth: usize) {
    if let Element::Rect {
        x,
        y,
        w,
        h,
        fill,
        stroke,
    } = elem
    {
        indent(buf, depth);
        write!(
            buf,
            r#"<rect x="{x:.2}" y="{y:.2}" width="{w:.2}" height="{h:.2}""#,
        )
        .unwrap();
        write_fill(buf, fill.as_ref());
        write_stroke(buf, stroke.as_ref());
        buf.push_str("/>\n");
    }
}

fn render_circle(buf: &mut String, elem: &Element, depth: usize) {
    if let Element::Circle {
        cx,
        cy,
        r,
        fill,
        stroke,
    } = elem
    {
        indent(buf, depth);
        write!(buf, r#"<circle cx="{cx:.2}" cy="{cy:.2}" r="{r:.2}""#).unwrap();
        write_fill(buf, fill.as_ref());
        write_stroke(buf, stroke.as_ref());
        buf.push_str("/>\n");
    }
}

fn render_text(buf: &mut String, elem: &Element, depth: usize) {
    if let Element::Text {
        x,
        y,
        text,
        font,
        anchor,
    } = elem
    {
        indent(buf, depth);
        let anchor_str = match anchor {
            TextAnchor::Start => "start",
            TextAnchor::Middle => "middle",
            TextAnchor::End => "end",
        };
        write!(
            buf,
            r#"<text x="{x:.2}" y="{y:.2}" text-anchor="{anchor_str}""#,
        )
        .unwrap();
        write_font_attrs(buf, font);
        let escaped = xml_escape(text);
        writeln!(buf, ">{escaped}</text>").unwrap();
    }
}

fn render_polyline(buf: &mut String, elem: &Element, depth: usize) {
    if let Element::Polyline {
        points,
        stroke,
        fill,
    } = elem
    {
        indent(buf, depth);
        let pts: Vec<String> = points
            .iter()
            .map(|(px, py)| format!("{px:.2},{py:.2}"))
            .collect();
        write!(buf, r#"<polyline points="{}""#, pts.join(" ")).unwrap();
        write_fill(buf, fill.as_ref());
        write!(
            buf,
            r#" stroke="{}" stroke-width="{:.2}""#,
            stroke.color.to_svg_color(),
            stroke.width,
        )
        .unwrap();
        write_dash(buf, stroke.dash.as_ref());
        buf.push_str("/>\n");
    }
}

fn render_element(buf: &mut String, elem: &Element, depth: usize) {
    match elem {
        Element::Line { .. } => render_line(buf, elem, depth),
        Element::Rect { .. } => render_rect(buf, elem, depth),
        Element::Circle { .. } => render_circle(buf, elem, depth),
        Element::Text { .. } => render_text(buf, elem, depth),
        Element::Polyline { .. } => render_polyline(buf, elem, depth),
        Element::Group { elements } => {
            indent(buf, depth);
            buf.push_str("<g>\n");
            for child in elements {
                render_element(buf, child, depth + 1);
            }
            indent(buf, depth);
            buf.push_str("</g>\n");
        }
    }
}

fn write_fill(buf: &mut String, fill: Option<&crate::style::Fill>) {
    match fill {
        Some(f) => write!(buf, r#" fill="{}""#, f.color.to_svg_color()).unwrap(),
        None => buf.push_str(r#" fill="none""#),
    }
}

fn write_stroke(buf: &mut String, stroke: Option<&crate::style::Stroke>) {
    if let Some(s) = stroke {
        write!(
            buf,
            r#" stroke="{}" stroke-width="{:.2}""#,
            s.color.to_svg_color(),
            s.width,
        )
        .unwrap();
        write_dash(buf, s.dash.as_ref());
    }
}

fn write_dash(buf: &mut String, dash: Option<&Vec<f64>>) {
    if let Some(d) = dash {
        let dash_str: Vec<String> = d.iter().map(|v| format!("{v:.1}")).collect();
        write!(buf, r#" stroke-dasharray="{}""#, dash_str.join(",")).unwrap();
    }
}

fn write_font_attrs(buf: &mut String, font: &Font) {
    write!(
        buf,
        r#" font-family="{}" font-size="{:.1}" fill="{}""#,
        font.family,
        font.size,
        font.color.to_svg_color(),
    )
    .unwrap();
    if font.bold {
        buf.push_str(r#" font-weight="bold""#);
    }
    if font.italic {
        buf.push_str(r#" font-style="italic""#);
    }
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::style::{Fill, Stroke};

    #[test]
    fn render_basic_elements() {
        let elements = vec![
            Element::Rect {
                x: 10.0,
                y: 20.0,
                w: 100.0,
                h: 50.0,
                fill: Some(Fill::new(Color::RED)),
                stroke: None,
            },
            Element::Line {
                x1: 0.0,
                y1: 0.0,
                x2: 100.0,
                y2: 100.0,
                stroke: Stroke::new(Color::BLACK, 2.0),
            },
            Element::Circle {
                cx: 50.0,
                cy: 50.0,
                r: 10.0,
                fill: Some(Fill::new(Color::BLUE)),
                stroke: None,
            },
        ];
        let svg = SvgBackend.render(&elements, 200.0, 200.0).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<line"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn xml_escape_special_chars() {
        let escaped = xml_escape("<hello & \"world\">");
        assert_eq!(escaped, "&lt;hello &amp; &quot;world&quot;&gt;");
    }
}
