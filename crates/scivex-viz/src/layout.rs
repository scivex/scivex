/// Padding around a region.
#[derive(Debug, Clone, Copy)]
pub struct Padding {
    /// Top padding in pixels.
    pub top: f64,
    /// Right padding in pixels.
    pub right: f64,
    /// Bottom padding in pixels.
    pub bottom: f64,
    /// Left padding in pixels.
    pub left: f64,
}

impl Default for Padding {
    fn default() -> Self {
        Self {
            top: 50.0,
            right: 30.0,
            bottom: 60.0,
            left: 70.0,
        }
    }
}

/// A simple bounding rectangle.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    /// X origin (left edge).
    pub x: f64,
    /// Y origin (top edge).
    pub y: f64,
    /// Width.
    pub w: f64,
    /// Height.
    pub h: f64,
}

/// Grid layout for placing multiple axes inside a figure.
#[derive(Debug, Clone)]
pub struct Layout {
    /// Number of rows in the grid.
    pub rows: usize,
    /// Number of columns in the grid.
    pub cols: usize,
    /// Padding around the entire grid.
    pub padding: Padding,
    /// Spacing between adjacent cells in pixels.
    pub spacing: f64,
}

impl Layout {
    /// A single-cell layout (1 row, 1 column).
    #[must_use]
    pub fn single() -> Self {
        Self {
            rows: 1,
            cols: 1,
            padding: Padding::default(),
            spacing: 20.0,
        }
    }

    /// A grid layout with the given number of rows and columns.
    #[must_use]
    pub fn grid(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            padding: Padding::default(),
            spacing: 20.0,
        }
    }

    /// Compute the pixel bounds for the cell at `(row, col)` within a figure
    /// of the given `width` and `height`.
    #[must_use]
    pub fn cell_bounds(&self, row: usize, col: usize, width: f64, height: f64) -> Rect {
        let usable_w = width - self.padding.left - self.padding.right;
        let usable_h = height - self.padding.top - self.padding.bottom;

        let total_h_spacing = if self.cols > 1 {
            self.spacing * (self.cols - 1) as f64
        } else {
            0.0
        };
        let total_v_spacing = if self.rows > 1 {
            self.spacing * (self.rows - 1) as f64
        } else {
            0.0
        };

        let cell_w = (usable_w - total_h_spacing) / self.cols as f64;
        let cell_h = (usable_h - total_v_spacing) / self.rows as f64;

        let x = self.padding.left + col as f64 * (cell_w + self.spacing);
        let y = self.padding.top + row as f64 * (cell_h + self.spacing);

        Rect {
            x,
            y,
            w: cell_w,
            h: cell_h,
        }
    }
}

impl Default for Layout {
    fn default() -> Self {
        Self::single()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_default() {
        let p = Padding::default();
        assert!((p.top - 50.0).abs() < f64::EPSILON);
        assert!((p.right - 30.0).abs() < f64::EPSILON);
        assert!((p.bottom - 60.0).abs() < f64::EPSILON);
        assert!((p.left - 70.0).abs() < f64::EPSILON);
    }

    #[test]
    fn layout_default_is_single() {
        let l = Layout::default();
        assert_eq!(l.rows, 1);
        assert_eq!(l.cols, 1);
    }

    #[test]
    fn grid_layout() {
        let l = Layout::grid(3, 4);
        assert_eq!(l.rows, 3);
        assert_eq!(l.cols, 4);
    }

    #[test]
    fn single_cell_bounds() {
        let l = Layout::single();
        let r = l.cell_bounds(0, 0, 800.0, 600.0);
        // Should fill available space minus padding
        assert!(r.w > 0.0);
        assert!(r.h > 0.0);
        assert!((r.x - l.padding.left).abs() < f64::EPSILON);
        assert!((r.y - l.padding.top).abs() < f64::EPSILON);
    }

    #[test]
    fn grid_cells_positive_size() {
        let l = Layout::grid(3, 3);
        for row in 0..3 {
            for col in 0..3 {
                let r = l.cell_bounds(row, col, 800.0, 600.0);
                assert!(r.w > 0.0, "cell ({row},{col}) has non-positive width");
                assert!(r.h > 0.0, "cell ({row},{col}) has non-positive height");
            }
        }
    }

    #[test]
    fn single_layout() {
        let l = Layout::single();
        assert_eq!(l.rows, 1);
        assert_eq!(l.cols, 1);
    }

    #[test]
    fn grid_cell_bounds_no_overlap() {
        let l = Layout::grid(2, 2);
        let a = l.cell_bounds(0, 0, 800.0, 600.0);
        let b = l.cell_bounds(0, 1, 800.0, 600.0);
        let c = l.cell_bounds(1, 0, 800.0, 600.0);
        // a and b should not overlap horizontally
        assert!(a.x + a.w <= b.x + 0.01);
        // a and c should not overlap vertically
        assert!(a.y + a.h <= c.y + 0.01);
    }
}
