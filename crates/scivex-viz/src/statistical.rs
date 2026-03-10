use crate::color::Color;
use crate::element::Element;
use crate::layout::Rect;
use crate::plot::{AxisRange, PlotBuilder};
use crate::scale::Scale;
use crate::style::{Fill, Stroke};

/// A box plot builder for one or more datasets.
pub struct BoxPlotBuilder {
    datasets: Vec<Vec<f64>>,
    labels: Option<Vec<String>>,
    fill: Option<Fill>,
    stroke: Stroke,
}

impl BoxPlotBuilder {
    /// Create a box plot from one or more datasets.
    #[must_use]
    pub fn new(datasets: Vec<Vec<f64>>) -> Self {
        Self {
            datasets,
            labels: None,
            fill: Some(Fill::new(Color::rgb(31, 119, 180))),
            stroke: Stroke::new(Color::BLACK, 1.0),
        }
    }

    /// Set category labels for each box.
    #[must_use]
    pub fn labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Set the fill color for the IQR boxes.
    #[must_use]
    pub fn fill_color(mut self, c: Color) -> Self {
        self.fill = Some(Fill::new(c));
        self
    }

    /// Remove the fill from the IQR boxes (outline only).
    #[must_use]
    pub fn no_fill(mut self) -> Self {
        self.fill = None;
        self
    }
}

/// Statistics for a single box in a box plot.
struct BoxStats {
    median: f64,
    q1: f64,
    q3: f64,
    whisker_lo: f64,
    whisker_hi: f64,
    outliers: Vec<f64>,
}

fn compute_box_stats(data: &[f64]) -> Option<BoxStats> {
    if data.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();

    let median = percentile(&sorted, 0.5);
    let q1 = percentile(&sorted, 0.25);
    let q3 = percentile(&sorted, 0.75);
    let iqr = q3 - q1;
    let lo_fence = q1 - 1.5 * iqr;
    let hi_fence = q3 + 1.5 * iqr;

    let whisker_lo = sorted
        .iter()
        .copied()
        .find(|&v| v >= lo_fence)
        .unwrap_or(sorted[0]);
    let whisker_hi = sorted
        .iter()
        .rev()
        .copied()
        .find(|&v| v <= hi_fence)
        .unwrap_or(sorted[n - 1]);

    let outliers: Vec<f64> = sorted
        .iter()
        .copied()
        .filter(|&v| v < whisker_lo || v > whisker_hi)
        .collect();

    Some(BoxStats {
        median,
        q1,
        q3,
        whisker_lo,
        whisker_hi,
        outliers,
    })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    let frac = idx - lo as f64;
    if hi >= n {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

impl PlotBuilder for BoxPlotBuilder {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        let n = self.datasets.len();
        let x_range = if n > 0 {
            Some((-0.5, n as f64 - 0.5))
        } else {
            None
        };

        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for ds in &self.datasets {
            for &v in ds {
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
        }
        let y_range = if lo.is_finite() && hi.is_finite() {
            Some((lo, hi))
        } else {
            None
        };
        (x_range, y_range)
    }

    fn build_elements(&self, x_scale: &dyn Scale, y_scale: &dyn Scale, area: Rect) -> Vec<Element> {
        let mut elements = Vec::new();
        let n = self.datasets.len();
        if n == 0 {
            return elements;
        }

        let box_width_data = 0.6;
        for (i, ds) in self.datasets.iter().enumerate() {
            let Some(stats) = compute_box_stats(ds) else {
                continue;
            };

            let cx = area.x + x_scale.transform(i as f64) * area.w;
            let half_w = (x_scale.transform(box_width_data / 2.0)
                - x_scale.transform(-box_width_data / 2.0))
                * area.w
                / 2.0;

            let y_q1 = area.y + area.h - y_scale.transform(stats.q1) * area.h;
            let y_q3 = area.y + area.h - y_scale.transform(stats.q3) * area.h;
            let y_med = area.y + area.h - y_scale.transform(stats.median) * area.h;
            let y_wlo = area.y + area.h - y_scale.transform(stats.whisker_lo) * area.h;
            let y_whi = area.y + area.h - y_scale.transform(stats.whisker_hi) * area.h;

            // IQR box
            elements.push(Element::Rect {
                x: cx - half_w,
                y: y_q3,
                w: half_w * 2.0,
                h: y_q1 - y_q3,
                fill: self.fill,
                stroke: Some(self.stroke.clone()),
            });

            // Median line
            elements.push(Element::Line {
                x1: cx - half_w,
                y1: y_med,
                x2: cx + half_w,
                y2: y_med,
                stroke: Stroke::new(Color::BLACK, 2.0),
            });

            // Lower whisker
            elements.push(Element::Line {
                x1: cx,
                y1: y_q1,
                x2: cx,
                y2: y_wlo,
                stroke: self.stroke.clone(),
            });
            elements.push(Element::Line {
                x1: cx - half_w * 0.5,
                y1: y_wlo,
                x2: cx + half_w * 0.5,
                y2: y_wlo,
                stroke: self.stroke.clone(),
            });

            // Upper whisker
            elements.push(Element::Line {
                x1: cx,
                y1: y_q3,
                x2: cx,
                y2: y_whi,
                stroke: self.stroke.clone(),
            });
            elements.push(Element::Line {
                x1: cx - half_w * 0.5,
                y1: y_whi,
                x2: cx + half_w * 0.5,
                y2: y_whi,
                stroke: self.stroke.clone(),
            });

            // Outliers
            for &out in &stats.outliers {
                let y_out = area.y + area.h - y_scale.transform(out) * area.h;
                elements.push(Element::Circle {
                    cx,
                    cy: y_out,
                    r: 3.0,
                    fill: None,
                    stroke: Some(self.stroke.clone()),
                });
            }
        }

        elements
    }

    fn label(&self) -> Option<&str> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::PlotBuilder;

    #[test]
    fn box_plot_labels() {
        let b = BoxPlotBuilder::new(vec![vec![1.0, 2.0]])
            .labels(vec!["Group A".into()]);
        assert_eq!(b.labels, Some(vec!["Group A".to_string()]));
    }

    #[test]
    fn box_plot_fill_color() {
        let b = BoxPlotBuilder::new(vec![vec![1.0, 2.0]])
            .fill_color(Color::RED);
        assert_eq!(b.fill.unwrap().color, Color::RED);
    }

    #[test]
    fn box_plot_no_fill() {
        let b = BoxPlotBuilder::new(vec![vec![1.0, 2.0]]).no_fill();
        assert!(b.fill.is_none());
    }

    #[test]
    fn box_plot_data_range() {
        let b = BoxPlotBuilder::new(vec![vec![1.0, 5.0], vec![2.0, 8.0]]);
        let (xr, yr) = b.data_range();
        assert_eq!(xr, Some((-0.5, 1.5)));
        assert_eq!(yr, Some((1.0, 8.0)));
    }

    #[test]
    fn box_plot_empty_datasets() {
        let b = BoxPlotBuilder::new(vec![]);
        let (xr, yr) = b.data_range();
        assert!(xr.is_none());
        assert!(yr.is_none());
    }

    #[test]
    fn box_stats_empty() {
        assert!(compute_box_stats(&[]).is_none());
    }

    #[test]
    fn box_stats_single_value() {
        let stats = compute_box_stats(&[5.0]).unwrap();
        assert!((stats.median - 5.0).abs() < 1e-10);
        assert!((stats.q1 - 5.0).abs() < 1e-10);
        assert!((stats.q3 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn percentile_interpolation() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&sorted, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile(&sorted, 1.0) - 4.0).abs() < 1e-10);
        assert!((percentile(&sorted, 0.5) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn box_plot_label_is_none() {
        let b = BoxPlotBuilder::new(vec![vec![1.0]]);
        assert_eq!(PlotBuilder::label(&b), None);
    }

    #[test]
    fn box_stats_known_data() {
        // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let data: Vec<f64> = (1..=10).map(f64::from).collect();
        let stats = compute_box_stats(&data).unwrap();
        assert!((stats.median - 5.5).abs() < 1e-10);
        assert!((stats.q1 - 3.25).abs() < 1e-10);
        assert!((stats.q3 - 7.75).abs() < 1e-10);
        assert!(stats.outliers.is_empty());
    }

    #[test]
    fn box_stats_with_outliers() {
        let mut data: Vec<f64> = (1..=10).map(f64::from).collect();
        data.push(100.0); // outlier
        let stats = compute_box_stats(&data).unwrap();
        assert!(stats.outliers.contains(&100.0));
    }
}
