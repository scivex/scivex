/// Trait for mapping data values to normalized `[0, 1]` space and back.
pub trait Scale {
    /// Map a data value to `[0, 1]`.
    fn transform(&self, value: f64) -> f64;
    /// Map a normalized value in `[0, 1]` back to data space.
    fn inverse(&self, t: f64) -> f64;
    /// Generate approximately `n` nicely-spaced tick values.
    fn nice_ticks(&self, n_ticks: usize) -> Vec<f64>;
}

/// A linear scale mapping `[min, max]` to `[0, 1]`.
#[derive(Debug, Clone)]
pub struct LinearScale {
    /// Minimum data value.
    pub min: f64,
    /// Maximum data value.
    pub max: f64,
}

impl LinearScale {
    /// Create a linear scale spanning `[min, max]`.
    #[must_use]
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }
}

impl Scale for LinearScale {
    fn transform(&self, value: f64) -> f64 {
        if (self.max - self.min).abs() < f64::EPSILON {
            return 0.5;
        }
        (value - self.min) / (self.max - self.min)
    }

    fn inverse(&self, t: f64) -> f64 {
        self.min + t * (self.max - self.min)
    }

    fn nice_ticks(&self, n_ticks: usize) -> Vec<f64> {
        heckbert_nice_ticks(self.min, self.max, n_ticks)
    }
}

/// A logarithmic scale mapping `[min, max]` (both > 0) to `[0, 1]`.
#[derive(Debug, Clone)]
pub struct LogScale {
    /// Minimum data value (must be positive).
    pub min: f64,
    /// Maximum data value (must be positive).
    pub max: f64,
    /// Logarithm base (default 10).
    pub base: f64,
}

impl LogScale {
    /// Create a log scale spanning `[min, max]` with base 10.
    #[must_use]
    pub fn new(min: f64, max: f64) -> Self {
        Self {
            min,
            max,
            base: 10.0,
        }
    }

    /// Set a custom logarithm base (e.g. 2 or `e`).
    #[must_use]
    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
        self
    }
}

impl Scale for LogScale {
    fn transform(&self, value: f64) -> f64 {
        if value <= 0.0 || self.min <= 0.0 || self.max <= 0.0 {
            return 0.0;
        }
        let log_min = self.min.log(self.base);
        let log_max = self.max.log(self.base);
        let range = log_max - log_min;
        if range.abs() < f64::EPSILON {
            return 0.5;
        }
        (value.log(self.base) - log_min) / range
    }

    fn inverse(&self, t: f64) -> f64 {
        let log_min = self.min.log(self.base);
        let log_max = self.max.log(self.base);
        self.base.powf(log_min + t * (log_max - log_min))
    }

    fn nice_ticks(&self, n_ticks: usize) -> Vec<f64> {
        if self.min <= 0.0 || self.max <= 0.0 || n_ticks == 0 {
            return vec![];
        }
        let log_min = self.min.log10().floor() as i32;
        let log_max = self.max.log10().ceil() as i32;
        let mut ticks = Vec::new();
        for exp in log_min..=log_max {
            let val = 10.0_f64.powi(exp);
            if val >= self.min && val <= self.max {
                ticks.push(val);
            }
            if ticks.len() >= n_ticks {
                break;
            }
        }
        ticks
    }
}

// ---------------------------------------------------------------------------
// Heckbert "nice numbers" algorithm
// ---------------------------------------------------------------------------

/// Round `x` to a "nice" number (1, 2, 5 × 10^n).
fn nice_num(x: f64, round: bool) -> f64 {
    let exp = x.log10().floor();
    let frac = x / 10.0_f64.powf(exp);
    let nice = if round {
        if frac < 1.5 {
            1.0
        } else if frac < 3.0 {
            2.0
        } else if frac < 7.0 {
            5.0
        } else {
            10.0
        }
    } else if frac <= 1.0 {
        1.0
    } else if frac <= 2.0 {
        2.0
    } else if frac <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice * 10.0_f64.powf(exp)
}

/// Generate approximately `n` nice tick values spanning `[min, max]`.
#[must_use]
pub fn heckbert_nice_ticks(min: f64, max: f64, n: usize) -> Vec<f64> {
    if n == 0 || min >= max {
        return vec![];
    }
    let range = nice_num(max - min, false);
    let d = nice_num(range / n as f64, true);
    let graph_min = (min / d).floor() * d;
    let graph_max = (max / d).ceil() * d;
    let mut ticks = Vec::new();
    let mut x = graph_min;
    // Safety limit to avoid infinite loops from floating-point edge cases.
    let limit = n * 4 + 2;
    while x <= graph_max + 0.5 * d && ticks.len() < limit {
        // Only include ticks within or very close to the data range.
        if x >= min - 0.5 * d && x <= max + 0.5 * d {
            // Round to avoid floating-point artefacts like 0.30000000000000004.
            let rounded = (x / d).round() * d;
            ticks.push(rounded);
        }
        x += d;
    }
    ticks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_transform_inverse() {
        let s = LinearScale::new(10.0, 20.0);
        assert!((s.transform(10.0)).abs() < 1e-12);
        assert!((s.transform(20.0) - 1.0).abs() < 1e-12);
        assert!((s.transform(15.0) - 0.5).abs() < 1e-12);
        assert!((s.inverse(0.5) - 15.0).abs() < 1e-12);
    }

    #[test]
    fn nice_ticks_reasonable() {
        let ticks = heckbert_nice_ticks(0.0, 10.0, 5);
        assert!(!ticks.is_empty());
        assert!(ticks.len() <= 12);
        // Should include 0 and 10
        assert!(ticks.iter().any(|&t| t.abs() < 1e-10));
        assert!(ticks.iter().any(|&t| (t - 10.0).abs() < 1e-10));
    }

    #[test]
    fn log_scale_transform() {
        let s = LogScale::new(1.0, 1000.0);
        assert!((s.transform(1.0)).abs() < 1e-12);
        assert!((s.transform(1000.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn linear_equal_min_max() {
        let s = LinearScale::new(5.0, 5.0);
        assert!((s.transform(5.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn linear_inverse_roundtrip() {
        let s = LinearScale::new(-10.0, 50.0);
        for &v in &[-10.0, 0.0, 20.0, 50.0] {
            let t = s.transform(v);
            let back = s.inverse(t);
            assert!((back - v).abs() < 1e-10, "{v} roundtrip failed");
        }
    }

    #[test]
    fn log_scale_inverse() {
        let s = LogScale::new(1.0, 1000.0);
        let mid_t = s.transform(31.62); // ~sqrt(1000)
        let back = s.inverse(mid_t);
        assert!((back - 31.62).abs() < 0.1);
    }

    #[test]
    fn log_scale_non_positive_returns_zero() {
        let s = LogScale::new(1.0, 100.0);
        assert!((s.transform(0.0)).abs() < 1e-12);
        assert!((s.transform(-5.0)).abs() < 1e-12);
    }

    #[test]
    fn log_scale_with_base() {
        let s = LogScale::new(1.0, 100.0).with_base(2.0);
        assert!((s.base - 2.0).abs() < f64::EPSILON);
        // transform should still work
        assert!((s.transform(1.0)).abs() < 1e-10);
    }

    #[test]
    fn nice_ticks_empty_when_invalid() {
        assert!(heckbert_nice_ticks(0.0, 0.0, 5).is_empty());
        assert!(heckbert_nice_ticks(10.0, 5.0, 5).is_empty());
        assert!(heckbert_nice_ticks(0.0, 10.0, 0).is_empty());
    }

    #[test]
    fn log_scale_ticks() {
        let s = LogScale::new(1.0, 10000.0);
        let ticks = s.nice_ticks(10);
        assert!(ticks.contains(&1.0));
        assert!(ticks.contains(&10.0));
        assert!(ticks.contains(&100.0));
        assert!(ticks.contains(&1000.0));
    }
}
