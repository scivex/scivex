# Visualization

`scivex-viz` is a from-scratch visualization library that produces SVG, HTML,
bitmap (PNG), and terminal braille output with zero external rendering
dependencies. It follows a three-layer architecture:

- **User API** -- `Figure`, `Axes`, plot builders (`LinePlot`, `ScatterPlot`, etc.)
- **Element layer** -- backend-agnostic drawing primitives (`Element`)
- **Renderer layer** -- `SvgBackend`, `TerminalBackend`, `HtmlBackend`, `BitmapBackend`

All examples in this guide use the prelude:

```rust
use scivex_viz::prelude::*;
```

---

## Figure and Axes Basics

A `Figure` is the top-level container. It holds a `Layout` and one or more
`Axes`, each placed in a grid cell. An `Axes` holds plots, labels, annotations,
and styling.

```rust
use scivex_viz::prelude::*;

let fig = Figure::new()
    .size(800.0, 600.0)           // width x height in pixels (default 800x600)
    .plot(                         // shorthand for add_axes(0, 0, ...)
        Axes::new()
            .title("My Plot")
            .x_label("x")
            .y_label("y")
            .grid(true)            // show grid lines (default: true)
            .add_plot(LinePlot::new(
                vec![0.0, 1.0, 2.0, 3.0],
                vec![0.0, 1.0, 0.5, 1.5],
            )),
    );

let svg = fig.to_svg().unwrap();
```

### Manual axis ranges

By default, axes auto-fit to the data. Override with explicit ranges:

```rust
# use scivex_viz::prelude::*;
let axes = Axes::new()
    .x_range(0.0, 10.0)
    .y_range(-1.0, 1.0)
    .add_plot(LinePlot::new(vec![0.0, 5.0, 10.0], vec![0.0, 1.0, -1.0]));
```

### Hiding tick marks

```rust
# use scivex_viz::prelude::*;
let axes = Axes::new()
    .hide_x_ticks(true)
    .hide_y_ticks(true);
```

---

## Line Plots

`LinePlot` connects `(x, y)` points with a stroked line. Customize color,
width, dash pattern, and legend label.

```rust
# use scivex_viz::prelude::*;
let axes = Axes::new()
    .title("Line Styles")
    .add_plot(
        LinePlot::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.5])
            .color(Color::BLUE)
            .width(2.0)
            .label("solid"),
    )
    .add_plot(
        LinePlot::new(vec![0.0, 1.0, 2.0], vec![0.5, 0.8, 0.2])
            .color(Color::RED)
            .width(1.5)
            .dash(vec![6.0, 3.0])   // 6px on, 3px off
            .label("dashed"),
    )
    .annotate(Annotation::legend()); // show the legend

let fig = Figure::new().plot(axes);
```

---

## Scatter Plots

`ScatterPlot` draws markers at each data point. Customize marker color, size,
and shape.

```rust
# use scivex_viz::prelude::*;
let axes = Axes::new()
    .title("Scatter")
    .add_plot(
        ScatterPlot::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.3, 3.1, 2.8, 4.5, 3.9],
        )
        .color(Color::ORANGE)
        .size(5.0)                          // marker radius in pixels
        .shape(MarkerShape::Diamond)
        .label("observations"),
    );
```

Available marker shapes: `Circle`, `Square`, `Triangle`, `Cross`, `Plus`,
`Diamond`.

---

## Bar Charts

`BarPlot` renders categorical data as vertical bars.

```rust
# use scivex_viz::prelude::*;
let axes = Axes::new()
    .title("Sales by Region")
    .x_label("Region")
    .y_label("Revenue ($M)")
    .add_plot(
        BarPlot::new(
            vec!["East".into(), "West".into(), "North".into(), "South".into()],
            vec![12.0, 18.5, 9.3, 14.7],
        )
        .color(Color::rgb(44, 160, 44))
        .bar_width(0.6)                    // fraction of category spacing
        .label("2025"),
    );

let fig = Figure::new().plot(axes);
```

---

## Histograms

`Histogram` bins continuous data into equal-width buckets.

```rust
# use scivex_viz::prelude::*;
let data: Vec<f64> = (0..200)
    .map(|i| (i as f64 * 0.031).sin() * 3.0 + 5.0)
    .collect();

let axes = Axes::new()
    .title("Distribution")
    .x_label("Value")
    .y_label("Count")
    .add_plot(
        Histogram::new(data, 25)           // 25 bins
            .color(Color::rgba(70, 130, 180, 180))
            .label("samples"),
    );

let fig = Figure::new().plot(axes);
```

---

## 3D Surface Plots

`SurfacePlot` renders a 2D grid of z-values as a pseudo-3D surface using
isometric projection with the painter's algorithm for depth ordering.

```rust
# use scivex_viz::prelude::*;
let n = 20;
let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();
let y: Vec<f64> = (0..n).map(|j| j as f64 * 0.3).collect();
let z: Vec<Vec<f64>> = y.iter().map(|yv| {
    x.iter().map(|xv| (xv * 0.5).sin() * (yv * 0.5).cos()).collect()
}).collect();

let axes = Axes::new()
    .title("Surface")
    .add_plot(
        SurfacePlot::new(x, y, z)
            .mode(SurfaceMode::FilledWireframe)  // Wireframe | Filled | FilledWireframe
            .colormap(ColorMap::viridis())
            .wire_color(Color::rgba(40, 40, 40, 200))
            .wire_width(0.5)
            .azimuth(45.0)                       // viewing angle in degrees
            .elevation(30.0)
            .with_label("f(x,y)"),
    );

let fig = Figure::new().size(800.0, 600.0).plot(axes);
```

### Available colormaps

- `ColorMap::viridis()` -- perceptually uniform, blue-green-yellow
- `ColorMap::plasma()` -- purple-orange-yellow
- `ColorMap::inferno()` -- black-red-yellow-white
- `ColorMap::coolwarm()` -- diverging blue-white-red
- `ColorMap::new(colors)` -- custom from a list of color stops

---

## Pair Plots and Joint Plots

### Pair plot

`PairPlot` produces an n-by-n grid of scatter plots with histograms (or KDE
curves) on the diagonal. It is useful for exploring pairwise relationships
between multiple variables.

```rust
# use scivex_viz::prelude::*;
let columns = vec![
    vec![1.0, 2.0, 3.0, 4.0, 5.0],
    vec![5.0, 4.0, 3.0, 2.0, 1.0],
    vec![1.0, 4.0, 9.0, 16.0, 25.0],
];

let fig = PairPlot::new(columns)
    .labels(&["height", "weight", "age"])
    .point_color(Color::rgba(70, 130, 180, 160))
    .point_size(3.0)
    .hist_bins(20)
    .hist_color(Color::rgba(70, 130, 180, 180))
    .diag(DiagMode::Histogram)   // or DiagMode::Kde
    .fig_size(900.0, 900.0)
    .build();

let svg = fig.to_svg().unwrap();
```

### Joint plot

`JointPlot` shows a central scatter plot with marginal histograms on the top
and right edges, using a weighted grid layout.

```rust
# use scivex_viz::prelude::*;
let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 7.0];

let fig = JointPlot::new(x, y)
    .x_label("Feature A")
    .y_label("Feature B")
    .point_color(Color::RED)
    .point_size(4.0)
    .hist_bins(15)
    .hist_color(Color::rgba(70, 130, 180, 180))
    .marginal_ratio(0.2)         // fraction of figure for marginals (clamped to 0.05..0.5)
    .fig_size(700.0, 700.0)
    .build();
```

---

## Subplots and Layouts

### Uniform grid

Use `Layout::grid(rows, cols)` and `Figure::add_axes(row, col, axes)` to
create multi-panel figures.

```rust
# use scivex_viz::prelude::*;
let fig = Figure::new()
    .size(1000.0, 600.0)
    .layout(Layout::grid(2, 2))
    .add_axes(0, 0, Axes::new()
        .title("Top Left")
        .add_plot(LinePlot::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.5])))
    .add_axes(0, 1, Axes::new()
        .title("Top Right")
        .add_plot(ScatterPlot::new(vec![1.0, 2.0, 3.0], vec![3.0, 1.0, 2.0])))
    .add_axes(1, 0, Axes::new()
        .title("Bottom Left")
        .add_plot(Histogram::new(vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0], 5)))
    .add_axes(1, 1, Axes::new()
        .title("Bottom Right")
        .add_plot(BarPlot::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![10.0, 20.0, 15.0],
        )));

let svg = fig.to_svg().unwrap();
```

### Weighted grid

`Layout::weighted_grid` assigns proportional sizes to rows and columns.
Weights are relative -- `vec![1.0, 3.0]` gives the second row three times
the height of the first.

```rust
# use scivex_viz::prelude::*;
// A small panel on top, a large panel below
let layout = Layout::weighted_grid(
    vec![1.0, 3.0],      // row weights
    vec![3.0, 1.0],      // column weights
);

let fig = Figure::new()
    .size(800.0, 600.0)
    .layout(layout)
    .add_axes(0, 0, Axes::new().title("Small top-left"))
    .add_axes(0, 1, Axes::new().title("Narrow top-right"))
    .add_axes(1, 0, Axes::new().title("Wide bottom-left"))
    .add_axes(1, 1, Axes::new().title("Narrow bottom-right"));
```

The `JointPlot` builder uses this internally to create its marginal panels.

---

## Shared Axes

When comparing subplots, shared axes unify the data ranges across rows or
columns and hide redundant tick labels.

```rust
# use scivex_viz::prelude::*;
let fig = Figure::new()
    .layout(Layout::grid(2, 1))
    .share_x(true)   // unify x-ranges within each column; hide x-ticks except bottom
    .share_y(true)   // unify y-ranges within each row; hide y-ticks except leftmost
    .add_axes(0, 0, Axes::new()
        .title("Signal A")
        .add_plot(LinePlot::new(vec![0.0, 1.0, 2.0], vec![0.0, 0.5, 1.0])))
    .add_axes(1, 0, Axes::new()
        .title("Signal B")
        .x_label("Time (s)")
        .add_plot(LinePlot::new(vec![0.0, 1.0, 2.0], vec![1.0, 0.2, 0.8])));
```

With `share_x(true)`, all axes in the same column use the union of their
x-data ranges, and only the bottom-most axes shows x-tick labels.

---

## Colors and Styling

### Color creation

```rust
# use scivex_viz::prelude::*;
// Named constants
let red   = Color::RED;
let blue  = Color::BLUE;
let black = Color::BLACK;
let white = Color::WHITE;
let gray  = Color::GRAY;
let light = Color::LIGHT_GRAY;

// RGB and RGBA constructors
let custom    = Color::rgb(31, 119, 180);
let semi      = Color::rgba(255, 0, 0, 128);    // 50% transparent red

// Hex parsing
let from_hex  = Color::from_hex("#FF6600").unwrap();
let with_alpha = Color::from_hex("#FF660080").unwrap();

// Interpolation
let mid = Color::BLACK.lerp(Color::WHITE, 0.5);  // gray
```

Full list of named constants: `RED`, `GREEN`, `BLUE`, `BLACK`, `WHITE`,
`GRAY`, `LIGHT_GRAY`, `ORANGE`, `PURPLE`, `CYAN`, `YELLOW`, `TRANSPARENT`.

### Colormaps

```rust
# use scivex_viz::prelude::*;
let cm = ColorMap::viridis();
let c = cm.sample(0.5);   // color at the midpoint

// Custom colormap from stops
let custom_cm = ColorMap::new(vec![Color::BLUE, Color::WHITE, Color::RED]).unwrap();
```

### Themes

Two built-in themes control background, foreground, grid, palette, and fonts:

```rust
# use scivex_viz::prelude::*;
let light_fig = Figure::new().theme(Theme::default_light());  // white background (default)
let dark_fig  = Figure::new().theme(Theme::default_dark());   // dark background
```

Themes can be applied at the `Axes` level too:

```rust
# use scivex_viz::prelude::*;
let axes = Axes::new().theme(Theme::default_dark());
```

### Strokes, fills, and markers

```rust
# use scivex_viz::prelude::*;
// Stroke with dash pattern
let stroke = Stroke::new(Color::RED, 2.0).dashed(vec![8.0, 4.0]);

// Fill
let fill = Fill::new(Color::rgba(0, 128, 255, 100));

// Marker (used internally by ScatterPlot)
let marker = Marker {
    shape: MarkerShape::Triangle,
    size: 6.0,
    color: Color::PURPLE,
};
```

### Annotations

Add reference lines, text labels, and legends:

```rust
# use scivex_viz::prelude::*;
let axes = Axes::new()
    .add_plot(LinePlot::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.5]))
    .annotate(Annotation::hline(0.5))         // horizontal reference line
    .annotate(Annotation::vline(1.0))         // vertical reference line
    .annotate(Annotation::text(1.5, 0.8, "peak"))  // text label at data coords
    .annotate(Annotation::legend());          // auto-generated legend (TopRight)
```

Legend positions: `LegendPosition::TopRight`, `TopLeft`, `BottomRight`,
`BottomLeft`.

---

## LaTeX Math in Labels

The `latex` module parses `$...$` regions in strings into styled segments
(Greek letters, superscripts, subscripts, fractions, symbols). The SVG backend
renders these as styled `<tspan>` elements; the terminal backend converts them
to Unicode equivalents.

```rust
use scivex_viz::latex::{parse as parse_latex, to_unicode as latex_to_unicode, contains_math};

// Check if a string contains math
assert!(contains_math("Energy $E = mc^2$"));

// Parse into segments
let segments = parse_latex("$\\alpha^2 + \\beta_i$");

// Convert to plain Unicode (for terminal display)
let text = latex_to_unicode(&segments);
```

### Supported syntax

| Input | Rendered as |
|---|---|
| `\alpha`, `\beta`, `\gamma`, ... | Greek letters |
| `x^2`, `x^{10}` | Superscripts |
| `x_i`, `x_{ij}` | Subscripts |
| `\frac{a}{b}` | Inline fraction `a/b` |
| `\pm`, `\cdot`, `\times` | Math operators |
| `\infty`, `\sqrt`, `\sum`, `\int`, `\partial` | Common symbols |

Use it in axis labels:

```rust
# use scivex_viz::prelude::*;
let axes = Axes::new()
    .x_label("$\\theta$ (radians)")
    .y_label("$\\sin(\\theta)$")
    .title("$f(\\theta) = \\sin(\\theta)$");
```

---

## Animation (GIF Export)

The `Animation` struct builds frame-by-frame animations and encodes them as
GIF89a with LZW compression and median-cut color quantization.

```rust
# use scivex_viz::prelude::*;
# use scivex_viz::animation::Animation;
use scivex_viz::element::Element;
use scivex_viz::style::{Fill, Stroke};

let mut anim = Animation::new(200, 200)
    .loop_count(0);   // 0 = infinite loop

for i in 0..10 {
    let x = 20.0 + i as f64 * 16.0;
    let frame = vec![
        // White background
        Element::Rect {
            x: 0.0, y: 0.0, w: 200.0, h: 200.0,
            fill: Some(Fill::new(Color::WHITE)),
            stroke: None,
        },
        // Moving circle
        Element::Circle {
            cx: x, cy: 100.0, r: 15.0,
            fill: Some(Fill::new(Color::RED)),
            stroke: Some(Stroke::new(Color::BLACK, 1.0)),
        },
    ];
    anim = anim.add_frame(frame, 100);  // 100ms per frame
}

// Encode to GIF bytes
let gif_bytes = anim.to_gif().unwrap();

// Or write directly to a file
// anim.save_gif("animation.gif").unwrap();
```

Each frame is rasterized by the `BitmapBackend`, quantized to 256 colors via
median-cut, and LZW-compressed into the GIF stream.

---

## Rendering Backends

### SVG (default)

The `SvgBackend` produces SVG 1.1 markup. This is the primary output format.

```rust
# use scivex_viz::prelude::*;
let fig = Figure::new().plot(
    Axes::new().add_plot(LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]))
);

// Get SVG string
let svg_string = fig.to_svg().unwrap();

// Save to file
fig.save_svg("plot.svg").unwrap();
```

### Terminal (braille art)

The `TerminalBackend` renders to Unicode braille characters (U+2800..U+28FF),
where each character cell represents a 2x4 dot grid. Default size is 80
columns by 24 rows.

```rust
# use scivex_viz::prelude::*;
let fig = Figure::new().plot(
    Axes::new().add_plot(LinePlot::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.5]))
);

// Get the braille string
let terminal_output = fig.to_terminal().unwrap();

// Print to stdout
fig.show_terminal().unwrap();
```

For custom terminal dimensions:

```rust
# use scivex_viz::prelude::*;
let backend = TerminalBackend::new(120, 40);
// Use via the Renderer trait:
// backend.render(&elements, width, height)
```

### HTML (interactive)

The `HtmlBackend` produces a standalone HTML page with Canvas 2D drawing and
optional pan/zoom/tooltip interactivity.

```rust
# use scivex_viz::prelude::*;
let backend = HtmlBackend::new()
    .interactive(true)    // enable pan/zoom/tooltips (default: true)
    .title("My Chart");

// Use via the Renderer trait to get an HTML string,
// or write directly to a file:
// backend.to_file(&elements, 800.0, 600.0, "chart.html").unwrap();
```

### Bitmap (PNG)

The `BitmapBackend` is a from-scratch software rasterizer that renders
elements to an RGBA pixel buffer with a built-in 5x7 bitmap font. It encodes
PNG with no external crates.

```rust
# use scivex_viz::prelude::*;
let backend = BitmapBackend::default();

// Render to RGBA bytes
// let rgba: Vec<u8> = backend.render_rgba(&elements, 800, 600);

// Render to PNG bytes
// let png: Vec<u8> = backend.render_png(&elements, 800, 600).unwrap();

// Write PNG to file
// backend.to_file(&elements, 800, 600, "plot.png").unwrap();
```

### Jupyter notebook (evcxr)

Figures support inline display in evcxr Jupyter notebooks:

```rust,ignore
let fig = Figure::new().plot(Axes::new().add_plot(
    LinePlot::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.5])
));
fig.evcxr_display();  // renders inline SVG in the notebook cell
```

---

## Saving Figures

A summary of all output methods:

| Method | Output |
|---|---|
| `fig.to_svg()` | `Result<String>` -- SVG markup |
| `fig.save_svg("path.svg")` | Write SVG to file |
| `fig.to_terminal()` | `Result<String>` -- braille art |
| `fig.show_terminal()` | Print braille art to stdout |
| `fig.evcxr_display()` | Display in evcxr Jupyter notebook |
| `SvgBackend.to_file(&elems, w, h, "p.svg")` | Low-level SVG file write |
| `HtmlBackend::new().to_file(&elems, w, h, "p.html")` | HTML file write |
| `BitmapBackend::default().to_file(&elems, w, h, "p.png")` | PNG file write |
| `anim.to_gif()` | `Result<Vec<u8>>` -- GIF bytes |
| `anim.save_gif("anim.gif")` | Write GIF to file |

The high-level `Figure` API (`to_svg`, `save_svg`, `to_terminal`,
`show_terminal`) is the recommended way to produce output. The low-level
backend APIs are available when you need direct control over rendering or when
working with raw `Element` lists.
