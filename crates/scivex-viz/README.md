# scivex-viz

Visualization and plotting for Scivex. Create publication-quality charts
rendered to SVG or terminal output.

## Highlights

- **Figure/Axes API** — Matplotlib-style composable figure building
- **Plot types** — Line, scatter, bar, histogram, heatmap, boxplot, pie, area
- **Styling** — Colors, line styles, markers, legends, titles, axis labels
- **SVG backend** — Scalable vector graphics output
- **Terminal backend** — ASCII/Unicode charts for terminal display
- **Subplots** — Multiple axes in a single figure
- **Color maps** — Viridis, plasma, inferno, magma, and custom gradients

## Usage

```rust
use scivex_viz::prelude::*;

let fig = Figure::new()
    .size(800.0, 600.0)
    .add_axes(
        Axes::new()
            .title("Sales by Month")
            .x_label("Month")
            .y_label("Revenue ($)")
            .add_plot(BarPlot::new(&months, &revenue).color(Color::BLUE))
    );

let svg = fig.render(&SvgBackend).unwrap();
std::fs::write("chart.svg", svg).unwrap();
```

## License

MIT
