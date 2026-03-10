# scivex-viz

Visualization and plotting for Scivex. Create publication-quality SVG charts
or terminal-based plots with a builder API.

## Highlights

- **Plot types** — Line, scatter, bar, histogram, heatmap, box plot
- **Figure/Axes model** — Multi-plot layouts with rows, columns, grids
- **SVG backend** — Clean SVG output for web and publication
- **Terminal backend** — Braille/Unicode art for CLI workflows
- **Styling** — Colors, markers, strokes, themes (default + dark)
- **Scales** — Linear and logarithmic axes
- **Colormaps** — Viridis, Plasma, Inferno, Cividis, Greys, and more

## Usage

```rust
use scivex_viz::prelude::*;

let fig = Figure::new()
    .size(800.0, 600.0)
    .theme(Theme::dark())
    .add_axes(
        Axes::new()
            .title("Scatter Plot")
            .x_label("X")
            .y_label("Y")
            .add_plot(ScatterPlot::new(&x, &y).color(Color::RED).size(4.0))
            .add_plot(LinePlot::new(&x, &trend).color(Color::BLUE).label("Trend"))
    );

let svg = fig.render(&SvgBackend).unwrap();
std::fs::write("plot.svg", svg).unwrap();
```

## License

MIT
