use super::Renderer;
use crate::element::{Element, TextAnchor};
use crate::error::Result;
use std::fmt::Write;

/// Renders elements to a standalone HTML page with Canvas 2D drawing and
/// optional pan/zoom/tooltip interactivity.
///
/// # To register in `backend/mod.rs`
///
/// ```rust,ignore
/// pub mod html;
/// pub use html::HtmlBackend;
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct HtmlBackend {
    /// Enable interactive pan/zoom/tooltips.
    interactive: bool,
    /// Optional page title.
    title: Option<String>,
}

impl HtmlBackend {
    /// Create a new `HtmlBackend` with interactivity enabled by default.
    #[must_use]
    pub fn new() -> Self {
        Self {
            interactive: true,
            title: None,
        }
    }

    /// Enable or disable interactive pan/zoom/tooltips.
    #[must_use]
    pub fn interactive(mut self, v: bool) -> Self {
        self.interactive = v;
        self
    }

    /// Set the HTML page title.
    #[must_use]
    pub fn title(mut self, t: &str) -> Self {
        self.title = Some(t.to_string());
        self
    }

    /// Render elements to HTML and write to a file.
    pub fn to_file(&self, elements: &[Element], width: f64, height: f64, path: &str) -> Result<()> {
        let content = self.render(elements, width, height)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

impl Default for HtmlBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for HtmlBackend {
    fn render(&self, elements: &[Element], width: f64, height: f64) -> Result<String> {
        let mut html = String::with_capacity(8192);
        let page_title = self.title.as_deref().unwrap_or("Scivex Chart");

        write!(
            html,
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{}</title>
<style>
body {{ margin:0; display:flex; justify-content:center; align-items:center; height:100vh; background:#f0f0f0; }}
canvas {{ background:#fff; cursor:grab; }}
canvas:active {{ cursor:grabbing; }}
#tooltip {{ position:fixed; padding:4px 8px; background:rgba(0,0,0,0.8); color:#fff;
  font:12px sans-serif; border-radius:3px; pointer-events:none; display:none; }}
</style>
</head>
<body>
<canvas id="c" width="{}" height="{}"></canvas>
<div id="tooltip"></div>
<script>
"#,
            html_escape(page_title),
            width,
            height,
        )
        .expect("write to String is infallible");

        write!(html, "\"use strict\";\nvar W={width},H={height},elems=")
            .expect("write to String is infallible");

        // Serialize elements to JSON
        html.push('[');
        for (i, elem) in elements.iter().enumerate() {
            if i > 0 {
                html.push(',');
            }
            write_element_json(&mut html, elem);
        }
        html.push_str("];\n");

        // Emit the drawing and interaction JavaScript
        write_js_body(&mut html, self.interactive);
        html.push_str("</script>\n</body>\n</html>\n");
        Ok(html)
    }
}

// ---------------------------------------------------------------------------
// JSON serialization helpers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn write_element_json(buf: &mut String, elem: &Element) {
    match elem {
        Element::Line {
            x1,
            y1,
            x2,
            y2,
            stroke,
        } => {
            write!(
                buf,
                r#"{{"t":"line","x1":{x1},"y1":{y1},"x2":{x2},"y2":{y2},"s":"#,
            )
            .expect("write to String is infallible");
            write_stroke_json(buf, stroke);
            buf.push('}');
        }
        Element::Rect {
            x,
            y,
            w,
            h,
            fill,
            stroke,
        } => {
            write!(buf, r#"{{"t":"rect","x":{x},"y":{y},"w":{w},"h":{h}"#,)
                .expect("write to String is infallible");
            write_optional_fill_json(buf, fill.as_ref());
            write_optional_stroke_json(buf, stroke.as_ref());
            buf.push('}');
        }
        Element::Circle {
            cx,
            cy,
            r,
            fill,
            stroke,
        } => {
            write!(buf, r#"{{"t":"circle","cx":{cx},"cy":{cy},"r":{r}"#)
                .expect("write to String is infallible");
            write_optional_fill_json(buf, fill.as_ref());
            write_optional_stroke_json(buf, stroke.as_ref());
            buf.push('}');
        }
        Element::Text {
            x,
            y,
            text,
            font,
            anchor,
        } => {
            let anchor_str = match anchor {
                TextAnchor::Start => "start",
                TextAnchor::Middle => "center",
                TextAnchor::End => "right",
            };
            write!(
                buf,
                r#"{{"t":"text","x":{x},"y":{y},"txt":"{}","anchor":"{anchor_str}""#,
                json_escape(text),
            )
            .expect("write to String is infallible");
            write!(
                buf,
                r#","ff":"{}","fs":{},"fc":"{}","fb":{},"fi":{}"#,
                json_escape(&font.family),
                font.size,
                font.color.to_svg_color(),
                font.bold,
                font.italic,
            )
            .expect("write to String is infallible");
            buf.push('}');
        }
        Element::Polyline {
            points,
            stroke,
            fill,
        } => {
            buf.push_str(r#"{"t":"polyline","pts":["#);
            for (i, (px, py)) in points.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                write!(buf, "[{px},{py}]").expect("write to String is infallible");
            }
            buf.push_str("],\"s\":");
            write_stroke_json(buf, stroke);
            write_optional_fill_json(buf, fill.as_ref());
            buf.push('}');
        }
        Element::Group { elements } => {
            buf.push_str(r#"{"t":"group","ch":["#);
            for (i, child) in elements.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                write_element_json(buf, child);
            }
            buf.push_str("]}");
        }
    }
}

fn write_stroke_json(buf: &mut String, s: &crate::style::Stroke) {
    write!(buf, r#"{{"c":"{}","w":{}"#, s.color.to_svg_color(), s.width,)
        .expect("write to String is infallible");
    if let Some(ref d) = s.dash {
        buf.push_str(",\"d\":[");
        for (i, v) in d.iter().enumerate() {
            if i > 0 {
                buf.push(',');
            }
            write!(buf, "{v}").expect("write to String is infallible");
        }
        buf.push(']');
    }
    buf.push('}');
}

fn write_optional_fill_json(buf: &mut String, fill: Option<&crate::style::Fill>) {
    if let Some(f) = fill {
        write!(buf, r#","f":"{}""#, f.color.to_svg_color()).expect("write to String is infallible");
    }
}

fn write_optional_stroke_json(buf: &mut String, stroke: Option<&crate::style::Stroke>) {
    if let Some(s) = stroke {
        buf.push_str(",\"s\":");
        write_stroke_json(buf, s);
    }
}

// ---------------------------------------------------------------------------
// JavaScript output
// ---------------------------------------------------------------------------

#[allow(clippy::collapsible_if)]
fn write_js_body(buf: &mut String, interactive: bool) {
    // Core drawing code (always emitted)
    buf.push_str(
        r#"var canvas=document.getElementById("c"),ctx=canvas.getContext("2d");
var panX=0,panY=0,scale=1;
function applyTx(){ctx.setTransform(scale,0,0,scale,panX,panY);}
function drawAll(){
  ctx.save();ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.restore();
  applyTx();
  for(var i=0;i<elems.length;i++) drawElem(elems[i]);
}
function drawElem(e){
  switch(e.t){
  case"line":
    ctx.beginPath();ctx.moveTo(e.x1,e.y1);ctx.lineTo(e.x2,e.y2);
    applyStroke(e.s);ctx.stroke();break;
  case"rect":
    if(e.f){ctx.fillStyle=e.f;ctx.fillRect(e.x,e.y,e.w,e.h);}
    if(e.s){applyStroke(e.s);ctx.strokeRect(e.x,e.y,e.w,e.h);}
    break;
  case"circle":
    ctx.beginPath();ctx.arc(e.cx,e.cy,e.r,0,Math.PI*2);
    if(e.f){ctx.fillStyle=e.f;ctx.fill();}
    if(e.s){applyStroke(e.s);ctx.stroke();}
    break;
  case"text":
    var st=(e.fb?"bold ":"")+(e.fi?"italic ":"")+e.fs+"px "+e.ff;
    ctx.font=st;ctx.fillStyle=e.fc;ctx.textAlign=e.anchor;
    ctx.textBaseline="alphabetic";ctx.fillText(e.txt,e.x,e.y);break;
  case"polyline":
    if(e.pts.length<1)break;
    ctx.beginPath();ctx.moveTo(e.pts[0][0],e.pts[0][1]);
    for(var j=1;j<e.pts.length;j++) ctx.lineTo(e.pts[j][0],e.pts[j][1]);
    if(e.f){ctx.fillStyle=e.f;ctx.fill();}
    applyStroke(e.s);ctx.stroke();break;
  case"group":
    for(var k=0;k<e.ch.length;k++) drawElem(e.ch[k]);break;
  }
}
function applyStroke(s){
  ctx.strokeStyle=s.c;ctx.lineWidth=s.w;
  if(s.d){ctx.setLineDash(s.d);}else{ctx.setLineDash([]);}
}
"#,
    );

    if interactive {
        buf.push_str(
            r#"var dragging=false,lastX=0,lastY=0;
canvas.addEventListener("mousedown",function(ev){dragging=true;lastX=ev.clientX;lastY=ev.clientY;});
window.addEventListener("mouseup",function(){dragging=false;});
window.addEventListener("mousemove",function(ev){
  if(dragging){panX+=ev.clientX-lastX;panY+=ev.clientY-lastY;lastX=ev.clientX;lastY=ev.clientY;drawAll();}
  showTooltip(ev);
});
canvas.addEventListener("wheel",function(ev){
  ev.preventDefault();
  var r=canvas.getBoundingClientRect(),mx=ev.clientX-r.left,my=ev.clientY-r.top;
  var z=ev.deltaY<0?1.1:1/1.1;
  panX=mx-(mx-panX)*z;panY=my-(my-panY)*z;scale*=z;drawAll();
},{passive:false});
var tip=document.getElementById("tooltip");
function showTooltip(ev){
  var r=canvas.getBoundingClientRect(),mx=(ev.clientX-r.left-panX)/scale,my=(ev.clientY-r.top-panY)/scale;
  var found=null,best=Infinity;
  function check(e){
    if(e.t==="circle"){var dx=e.cx-mx,dy=e.cy-my,d=Math.sqrt(dx*dx+dy*dy);if(d<e.r+4&&d<best){best=d;found=e;}}
    if(e.t==="group"){for(var i=0;i<e.ch.length;i++) check(e.ch[i]);}
  }
  for(var i=0;i<elems.length;i++) check(elems[i]);
  if(found){tip.style.display="block";tip.style.left=(ev.clientX+12)+"px";tip.style.top=(ev.clientY+12)+"px";
    tip.textContent="("+found.cx.toFixed(1)+", "+found.cy.toFixed(1)+")";}
  else{tip.style.display="none";}
}
"#,
        );
    }

    buf.push_str("drawAll();\n");
}

// ---------------------------------------------------------------------------
// Escaping helpers
// ---------------------------------------------------------------------------

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::style::{Fill, Font, Stroke};

    #[test]
    fn render_contains_html_structure() {
        let elements = vec![Element::Rect {
            x: 10.0,
            y: 20.0,
            w: 100.0,
            h: 50.0,
            fill: Some(Fill::new(Color::RED)),
            stroke: None,
        }];
        let html = HtmlBackend::new().render(&elements, 800.0, 600.0).unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<canvas"));
        assert!(html.contains("</html>"));
        assert!(html.contains("drawAll"));
    }

    #[test]
    fn render_contains_element_json() {
        let elements = vec![
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
        let html = HtmlBackend::new().render(&elements, 400.0, 300.0).unwrap();
        assert!(html.contains(r#""t":"line""#));
        assert!(html.contains(r#""t":"circle""#));
    }

    #[test]
    fn interactive_flag_controls_pan_zoom() {
        let elements = vec![];
        let interactive = HtmlBackend::new()
            .interactive(true)
            .render(&elements, 200.0, 200.0)
            .unwrap();
        assert!(interactive.contains("mousedown"));
        assert!(interactive.contains("wheel"));
        assert!(interactive.contains("tooltip"));

        let static_html = HtmlBackend::new()
            .interactive(false)
            .render(&elements, 200.0, 200.0)
            .unwrap();
        assert!(!static_html.contains("mousedown"));
    }

    #[test]
    fn title_appears_in_output() {
        let html = HtmlBackend::new()
            .title("My Chart")
            .render(&[], 100.0, 100.0)
            .unwrap();
        assert!(html.contains("<title>My Chart</title>"));
    }

    #[test]
    fn title_escapes_html() {
        let html = HtmlBackend::new()
            .title("<script>alert(1)</script>")
            .render(&[], 100.0, 100.0)
            .unwrap();
        assert!(!html.contains("<script>alert"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn text_element_in_json() {
        let elements = vec![Element::Text {
            x: 10.0,
            y: 20.0,
            text: "Hello \"world\"".to_string(),
            font: Font::default(),
            anchor: TextAnchor::Middle,
        }];
        let html = HtmlBackend::new().render(&elements, 200.0, 200.0).unwrap();
        assert!(html.contains(r#""t":"text""#));
        assert!(html.contains(r#"Hello \"world\""#));
        assert!(html.contains(r#""anchor":"center""#));
    }

    #[test]
    fn polyline_in_json() {
        let elements = vec![Element::Polyline {
            points: vec![(0.0, 0.0), (10.0, 20.0), (30.0, 40.0)],
            stroke: Stroke::new(Color::RED, 1.0),
            fill: None,
        }];
        let html = HtmlBackend::new().render(&elements, 200.0, 200.0).unwrap();
        assert!(html.contains(r#""t":"polyline""#));
        assert!(html.contains(r#""pts""#));
    }

    #[test]
    fn group_in_json() {
        let elements = vec![Element::Group {
            elements: vec![Element::Circle {
                cx: 5.0,
                cy: 5.0,
                r: 2.0,
                fill: None,
                stroke: Some(Stroke::new(Color::BLACK, 1.0)),
            }],
        }];
        let html = HtmlBackend::new().render(&elements, 100.0, 100.0).unwrap();
        assert!(html.contains(r#""t":"group""#));
        assert!(html.contains(r#""ch""#));
    }

    #[test]
    fn default_impl() {
        let backend = HtmlBackend::default();
        assert!(backend.interactive);
        assert!(backend.title.is_none());
    }

    #[test]
    fn json_escape_special_chars() {
        assert_eq!(json_escape(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(json_escape("a\nb"), r"a\nb");
    }

    #[test]
    fn html_escape_special_chars() {
        let escaped = html_escape("<b>\"test\" & 'more'</b>");
        assert_eq!(
            escaped,
            "&lt;b&gt;&quot;test&quot; &amp; &#39;more&#39;&lt;/b&gt;"
        );
    }

    #[test]
    fn stroke_with_dash_in_json() {
        let elements = vec![Element::Line {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
            stroke: Stroke::new(Color::BLACK, 2.0).dashed(vec![5.0, 3.0]),
        }];
        let html = HtmlBackend::new().render(&elements, 200.0, 200.0).unwrap();
        assert!(html.contains(r#""d":[5,3]"#));
    }

    #[test]
    fn canvas_dimensions() {
        let html = HtmlBackend::new().render(&[], 640.0, 480.0).unwrap();
        assert!(html.contains(r#"width="640""#));
        assert!(html.contains(r#"height="480""#));
    }
}
