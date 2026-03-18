//! 3D surface plot with isometric projection.
//!
//! Renders a 2D grid of z-values as a wireframe or filled surface using
//! isometric projection and the painter's algorithm for depth ordering.

use crate::color::{Color, ColorMap};
use crate::element::Element;
use crate::layout::Rect;
use crate::plot::PlotBuilder;
use crate::scale::Scale;
use crate::style::{Fill, Stroke};

/// Rendering mode for the surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceMode {
    /// Draw only wireframe lines.
    Wireframe,
    /// Draw filled polygons (no wireframe).
    Filled,
    /// Draw filled polygons with wireframe overlay.
    FilledWireframe,
}

/// A 3D surface plot builder.
///
/// Displays a grid of `(x, y, z)` data as a pseudo-3D surface using isometric
/// projection. Supports wireframe, filled, and combined rendering modes.
///
/// # Example
///
/// ```rust
/// use scivex_viz::surface::{SurfacePlot, SurfaceMode};
///
/// let x = vec![0.0, 1.0, 2.0];
/// let y = vec![0.0, 1.0, 2.0];
/// let z = vec![
///     vec![0.0, 1.0, 0.5],
///     vec![1.0, 2.0, 1.5],
///     vec![0.5, 1.5, 1.0],
/// ];
/// let surface = SurfacePlot::new(x, y, z).mode(SurfaceMode::FilledWireframe);
/// ```
pub struct SurfacePlot {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<Vec<f64>>,
    mode: SurfaceMode,
    colormap: ColorMap,
    wire_color: Color,
    wire_width: f64,
    azimuth: f64,
    elevation: f64,
    label: Option<String>,
}

impl SurfacePlot {
    /// Create a new surface plot from grid coordinates.
    ///
    /// - `x`: x-axis grid values (length `nx`)
    /// - `y`: y-axis grid values (length `ny`)
    /// - `z`: z-values as `ny` rows of `nx` columns
    #[must_use]
    pub fn new(x: Vec<f64>, y: Vec<f64>, z: Vec<Vec<f64>>) -> Self {
        Self {
            x,
            y,
            z,
            mode: SurfaceMode::FilledWireframe,
            colormap: ColorMap::viridis(),
            wire_color: Color::rgba(40, 40, 40, 200),
            wire_width: 0.5,
            azimuth: 45.0_f64.to_radians(),
            elevation: 30.0_f64.to_radians(),
            label: None,
        }
    }

    /// Set the rendering mode.
    #[must_use]
    pub fn mode(mut self, mode: SurfaceMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the colormap for filled surfaces.
    #[must_use]
    pub fn colormap(mut self, cm: ColorMap) -> Self {
        self.colormap = cm;
        self
    }

    /// Set the wireframe line color.
    #[must_use]
    pub fn wire_color(mut self, c: Color) -> Self {
        self.wire_color = c;
        self
    }

    /// Set the wireframe line width.
    #[must_use]
    pub fn wire_width(mut self, w: f64) -> Self {
        self.wire_width = w;
        self
    }

    /// Set the viewing azimuth angle in degrees.
    #[must_use]
    pub fn azimuth(mut self, deg: f64) -> Self {
        self.azimuth = deg.to_radians();
        self
    }

    /// Set the viewing elevation angle in degrees.
    #[must_use]
    pub fn elevation(mut self, deg: f64) -> Self {
        self.elevation = deg.to_radians();
        self
    }

    /// Set a label for legends.
    #[must_use]
    pub fn with_label(mut self, l: &str) -> Self {
        self.label = Some(l.to_string());
        self
    }

    /// Project a 3D point to 2D using isometric-like projection.
    fn project(&self, x: f64, y: f64, z: f64) -> (f64, f64) {
        let cos_a = self.azimuth.cos();
        let sin_a = self.azimuth.sin();
        let cos_e = self.elevation.cos();
        let sin_e = self.elevation.sin();

        // Rotate around Z axis (azimuth), then tilt (elevation).
        let px = -x * sin_a + y * cos_a;
        let py = -x * cos_a * sin_e - y * sin_a * sin_e + z * cos_e;
        (px, py)
    }

    fn z_range(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for row in &self.z {
            for &v in row {
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            return (0.0, 1.0);
        }
        if (hi - lo).abs() < f64::EPSILON {
            lo -= 0.5;
            hi += 0.5;
        }
        (lo, hi)
    }
}

/// A single quad face of the surface grid.
struct Quad {
    corners: [(f64, f64); 4],
    avg_z: f64,
    z_norm: f64,
}

impl PlotBuilder for SurfacePlot {
    fn data_range(&self) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
        // For 3D surface, the projected 2D range depends on viewing angle.
        // Return None so axes use manual ranges or auto-fit.
        (None, None)
    }

    #[allow(clippy::too_many_lines)]
    fn build_elements(
        &self,
        _x_scale: &dyn Scale,
        _y_scale: &dyn Scale,
        area: Rect,
    ) -> Vec<Element> {
        let ny = self.y.len();
        let nx = self.x.len();
        if ny < 2 || nx < 2 {
            return Vec::new();
        }

        // Normalize x, y, z to [0, 1] range.
        let x_min = self.x.first().copied().unwrap_or(0.0);
        let x_max = self.x.last().copied().unwrap_or(1.0);
        let y_min = self.y.first().copied().unwrap_or(0.0);
        let y_max = self.y.last().copied().unwrap_or(1.0);
        let (z_min, z_max) = self.z_range();

        let x_span = if (x_max - x_min).abs() < f64::EPSILON {
            1.0
        } else {
            x_max - x_min
        };
        let y_span = if (y_max - y_min).abs() < f64::EPSILON {
            1.0
        } else {
            y_max - y_min
        };
        let z_span = if (z_max - z_min).abs() < f64::EPSILON {
            1.0
        } else {
            z_max - z_min
        };

        // Project all grid points.
        let mut projected: Vec<Vec<(f64, f64)>> = Vec::with_capacity(ny);
        let mut px_min = f64::INFINITY;
        let mut px_max = f64::NEG_INFINITY;
        let mut py_min = f64::INFINITY;
        let mut py_max = f64::NEG_INFINITY;

        for (j, yv) in self.y.iter().enumerate() {
            let mut row = Vec::with_capacity(nx);
            for (i, xv) in self.x.iter().enumerate() {
                let xn = (xv - x_min) / x_span;
                let yn = (yv - y_min) / y_span;
                let zn = (self.z[j][i] - z_min) / z_span;
                let (px, py) = self.project(xn, yn, zn);
                if px < px_min {
                    px_min = px;
                }
                if px > px_max {
                    px_max = px;
                }
                if py < py_min {
                    py_min = py;
                }
                if py > py_max {
                    py_max = py;
                }
                row.push((px, py));
            }
            projected.push(row);
        }

        // Map projected coords to pixel area.
        let px_span = if (px_max - px_min).abs() < f64::EPSILON {
            1.0
        } else {
            px_max - px_min
        };
        let py_span = if (py_max - py_min).abs() < f64::EPSILON {
            1.0
        } else {
            py_max - py_min
        };

        // Maintain aspect ratio within the plot area.
        let scale = (area.w / px_span).min(area.h / py_span) * 0.9;
        let cx = area.x + area.w / 2.0;
        let cy = area.y + area.h / 2.0;
        let pcx = f64::midpoint(px_min, px_max);
        let pcy = f64::midpoint(py_min, py_max);

        let to_pixel = |px: f64, py: f64| -> (f64, f64) {
            (
                cx + (px - pcx) * scale,
                cy - (py - pcy) * scale, // flip Y
            )
        };

        // Collect quads with average depth for painter's algorithm.
        let mut quads: Vec<Quad> = Vec::with_capacity((ny - 1) * (nx - 1));

        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let p00 = to_pixel(projected[j][i].0, projected[j][i].1);
                let p10 = to_pixel(projected[j][i + 1].0, projected[j][i + 1].1);
                let p11 = to_pixel(projected[j + 1][i + 1].0, projected[j + 1][i + 1].1);
                let p01 = to_pixel(projected[j + 1][i].0, projected[j + 1][i].1);

                let avg_z =
                    (self.z[j][i] + self.z[j][i + 1] + self.z[j + 1][i + 1] + self.z[j + 1][i])
                        / 4.0;
                let z_norm = (avg_z - z_min) / z_span;

                quads.push(Quad {
                    corners: [p00, p10, p11, p01],
                    avg_z,
                    z_norm,
                });
            }
        }

        // Sort by depth (back to front — painter's algorithm).
        // Quads with lower projected depth should be drawn first.
        quads.sort_by(|a, b| {
            a.avg_z
                .partial_cmp(&b.avg_z)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut elements = Vec::with_capacity(quads.len());

        for quad in &quads {
            let points: Vec<(f64, f64)> = quad.corners.to_vec();

            match self.mode {
                SurfaceMode::Wireframe => {
                    let mut pts = points;
                    // Close the quad.
                    pts.push(pts[0]);
                    elements.push(Element::Polyline {
                        points: pts,
                        stroke: Stroke::new(self.wire_color, self.wire_width),
                        fill: None,
                    });
                }
                SurfaceMode::Filled => {
                    let color = self.colormap.sample(quad.z_norm);
                    let mut pts = points;
                    pts.push(pts[0]);
                    elements.push(Element::Polyline {
                        points: pts,
                        stroke: Stroke::new(color, 0.0),
                        fill: Some(Fill::new(color)),
                    });
                }
                SurfaceMode::FilledWireframe => {
                    let color = self.colormap.sample(quad.z_norm);
                    let mut pts = points;
                    pts.push(pts[0]);
                    elements.push(Element::Polyline {
                        points: pts,
                        stroke: Stroke::new(self.wire_color, self.wire_width),
                        fill: Some(Fill::new(color)),
                    });
                }
            }
        }

        elements
    }

    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scale::LinearScale;

    fn sample_data() -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0];
        let z = vec![
            vec![0.0, 1.0, 0.5, 0.2],
            vec![1.0, 2.0, 1.5, 1.0],
            vec![0.5, 1.5, 1.0, 0.5],
        ];
        (x, y, z)
    }

    #[test]
    fn surface_creates_elements() {
        let (x, y, z) = sample_data();
        let s = SurfacePlot::new(x, y, z);
        let xs = LinearScale::new(0.0, 3.0);
        let ys = LinearScale::new(0.0, 2.0);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 600.0,
            h: 400.0,
        };
        let elems = s.build_elements(&xs, &ys, area);
        // 3 rows × 4 cols → (2)(3) = 6 quads
        assert_eq!(elems.len(), 6);
    }

    #[test]
    fn surface_wireframe_mode() {
        let (x, y, z) = sample_data();
        let s = SurfacePlot::new(x, y, z).mode(SurfaceMode::Wireframe);
        let xs = LinearScale::new(0.0, 3.0);
        let ys = LinearScale::new(0.0, 2.0);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 600.0,
            h: 400.0,
        };
        let elems = s.build_elements(&xs, &ys, area);
        // All should be polylines with no fill.
        for e in &elems {
            if let Element::Polyline { fill, .. } = e {
                assert!(fill.is_none());
            }
        }
    }

    #[test]
    fn surface_filled_mode() {
        let (x, y, z) = sample_data();
        let s = SurfacePlot::new(x, y, z).mode(SurfaceMode::Filled);
        let xs = LinearScale::new(0.0, 3.0);
        let ys = LinearScale::new(0.0, 2.0);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 600.0,
            h: 400.0,
        };
        let elems = s.build_elements(&xs, &ys, area);
        for e in &elems {
            if let Element::Polyline { fill, .. } = e {
                assert!(fill.is_some());
            }
        }
    }

    #[test]
    fn surface_empty_grid() {
        let s = SurfacePlot::new(vec![0.0], vec![0.0], vec![vec![1.0]]);
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
        };
        let elems = s.build_elements(&xs, &ys, area);
        assert!(elems.is_empty());
    }

    #[test]
    fn surface_data_range_is_none() {
        let (x, y, z) = sample_data();
        let s = SurfacePlot::new(x, y, z);
        let (xr, yr) = s.data_range();
        assert!(xr.is_none());
        assert!(yr.is_none());
    }

    #[test]
    fn surface_builder_methods() {
        let (x, y, z) = sample_data();
        let s = SurfacePlot::new(x, y, z)
            .mode(SurfaceMode::Wireframe)
            .wire_color(Color::RED)
            .wire_width(2.0)
            .azimuth(60.0)
            .elevation(45.0)
            .with_label("test surface");
        assert_eq!(s.mode, SurfaceMode::Wireframe);
        assert_eq!(s.wire_color, Color::RED);
        assert!((s.wire_width - 2.0).abs() < f64::EPSILON);
        assert_eq!(PlotBuilder::label(&s), Some("test surface"));
    }

    #[test]
    fn surface_projection() {
        let (x, y, z) = sample_data();
        let s = SurfacePlot::new(x, y, z);
        let (px, py) = s.project(0.0, 0.0, 0.0);
        assert!((px - 0.0).abs() < 1e-10);
        assert!((py - 0.0).abs() < 1e-10);
    }

    #[test]
    fn surface_z_range() {
        let (x, y, z) = sample_data();
        let s = SurfacePlot::new(x, y, z);
        let (lo, hi) = s.z_range();
        assert!((lo - 0.0).abs() < 1e-10);
        assert!((hi - 2.0).abs() < 1e-10);
    }
}
