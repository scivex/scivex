//! Aggregation operations on `Series<T>`.

use scivex_core::{Float, Scalar};

use super::Series;

// ---------------------------------------------------------------------------
// Scalar-level aggregations (work on any Scalar)
// ---------------------------------------------------------------------------

impl<T: Scalar> Series<T> {
    /// Sum of all non-null elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![1_i32, 2, 3]);
    /// assert_eq!(s.sum(), 6);
    /// ```
    pub fn sum(&self) -> T {
        self.non_null_iter()
            .copied()
            .fold(T::zero(), |acc, x| acc + x)
    }

    /// Product of all non-null elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![2_i32, 3, 4]);
    /// assert_eq!(s.product(), 24);
    /// ```
    pub fn product(&self) -> T {
        self.non_null_iter()
            .copied()
            .fold(T::one(), |acc, x| acc * x)
    }

    /// Minimum non-null element, or `None` if empty / all null.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![3_i32, 1, 4]);
    /// assert_eq!(s.min(), Some(1));
    /// ```
    pub fn min(&self) -> Option<T> {
        self.non_null_iter()
            .copied()
            .reduce(|a, b| if b < a { b } else { a })
    }

    /// Maximum non-null element, or `None` if empty / all null.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![3_i32, 1, 4]);
    /// assert_eq!(s.max(), Some(4));
    /// ```
    pub fn max(&self) -> Option<T> {
        self.non_null_iter()
            .copied()
            .reduce(|a, b| if b > a { b } else { a })
    }

    /// Apply `f` element-wise, returning a new series.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![1_i32, 2, 3]);
    /// let doubled = s.apply(|v| v * 2);
    /// assert_eq!(doubled.as_slice(), &[2, 4, 6]);
    /// ```
    pub fn apply<F: Fn(T) -> T>(&self, f: F) -> Series<T> {
        let data = self.data.iter().map(|v| f(*v)).collect();
        Series {
            name: self.name.clone(),
            data,
            null_mask: self.null_mask.clone(),
        }
    }

    /// Map each element to a different type, returning a new series.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![1_i32, 2, 3]);
    /// let floats = s.map(|v| v as f64 * 0.5);
    /// assert_eq!(floats.as_slice(), &[0.5, 1.0, 1.5]);
    /// ```
    pub fn map<U: Scalar, F: Fn(T) -> U>(&self, f: F) -> Series<U> {
        let data = self.data.iter().map(|v| f(*v)).collect();
        Series {
            name: self.name.clone(),
            data,
            null_mask: self.null_mask.clone(),
        }
    }

    // -- internal helpers ---------------------------------------------------

    /// Iterator over non-null elements.
    fn non_null_iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().enumerate().filter_map(
            |(i, v)| {
                if self.is_null_at(i) { None } else { Some(v) }
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Float-level aggregations
// ---------------------------------------------------------------------------

impl<T: Float> Series<T> {
    /// Arithmetic mean of non-null elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![1.0_f64, 2.0, 3.0]);
    /// assert!((s.mean() - 2.0).abs() < 1e-10);
    /// ```
    pub fn mean(&self) -> T {
        let count = self.count();
        if count == 0 {
            return T::nan();
        }
        self.sum() / T::from_usize(count)
    }

    /// Population variance of non-null elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    /// assert!(s.var() > 0.0);
    /// ```
    pub fn var(&self) -> T {
        let count = self.count();
        if count == 0 {
            return T::nan();
        }
        let mean = self.mean();
        let sum_sq = self.non_null_iter().copied().fold(T::zero(), |acc, x| {
            let d = x - mean;
            acc + d * d
        });
        sum_sq / T::from_usize(count)
    }

    /// Population standard deviation of non-null elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![1.0_f64, 1.0, 1.0]);
    /// assert!((s.std()).abs() < 1e-10); // all same → std = 0
    /// ```
    pub fn std(&self) -> T {
        self.var().sqrt()
    }

    /// Median of non-null elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::Series;
    /// let s = Series::new("x", vec![3.0_f64, 1.0, 2.0]);
    /// assert!((s.median() - 2.0).abs() < 1e-10);
    /// ```
    pub fn median(&self) -> T {
        let mut vals: Vec<T> = self.non_null_iter().copied().collect();
        if vals.is_empty() {
            return T::nan();
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let n = vals.len();
        if n % 2 == 1 {
            vals[n / 2]
        } else {
            (vals[n / 2 - 1] + vals[n / 2]) / T::from_usize(2)
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_product() {
        let s = Series::new("x", vec![1_i32, 2, 3, 4]);
        assert_eq!(s.sum(), 10);
        assert_eq!(s.product(), 24);
    }

    #[test]
    fn test_sum_with_nulls() {
        let s = Series::with_nulls("x", vec![1.0_f64, 0.0, 3.0], vec![false, true, false]).unwrap();
        assert_eq!(s.sum(), 4.0);
    }

    #[test]
    fn test_min_max() {
        let s = Series::new("x", vec![3_i32, 1, 4, 1, 5]);
        assert_eq!(s.min(), Some(1));
        assert_eq!(s.max(), Some(5));
    }

    #[test]
    fn test_min_max_empty() {
        let s: Series<i32> = Series::new("x", vec![]);
        assert_eq!(s.min(), None);
        assert_eq!(s.max(), None);
    }

    #[test]
    fn test_mean() {
        let s = Series::new("x", vec![2.0_f64, 4.0, 6.0]);
        assert_eq!(s.mean(), 4.0);
    }

    #[test]
    fn test_mean_empty() {
        let s: Series<f64> = Series::new("x", vec![]);
        assert!(s.mean().is_nan());
    }

    #[test]
    fn test_var_std() {
        let s = Series::new("x", vec![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let var = s.var();
        assert!((var - 4.0).abs() < 1e-10);
        assert!((s.std() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_odd() {
        let s = Series::new("x", vec![3.0_f64, 1.0, 2.0]);
        assert_eq!(s.median(), 2.0);
    }

    #[test]
    fn test_median_even() {
        let s = Series::new("x", vec![3.0_f64, 1.0, 2.0, 4.0]);
        assert_eq!(s.median(), 2.5);
    }

    #[test]
    fn test_apply() {
        let s = Series::new("x", vec![1_i32, 2, 3]);
        let doubled = s.apply(|v| v * 2_i32);
        assert_eq!(doubled.as_slice(), &[2, 4, 6]);
    }

    #[test]
    fn test_map() {
        let s = Series::new("x", vec![1_i32, 2, 3]);
        let floats: Series<f64> = s.map(|v| f64::from_usize(v as usize));
        assert_eq!(floats.as_slice(), &[1.0, 2.0, 3.0]);
    }

    // -- Edge-case tests -------------------------------------------------------

    #[test]
    fn test_sum_empty() {
        let s: Series<i32> = Series::new("x", vec![]);
        assert_eq!(s.sum(), 0);
    }

    #[test]
    fn test_product_empty() {
        let s: Series<i32> = Series::new("x", vec![]);
        assert_eq!(s.product(), 1);
    }

    #[test]
    fn test_sum_all_nulls() {
        let s = Series::with_nulls("x", vec![0.0_f64, 0.0, 0.0], vec![true, true, true]).unwrap();
        assert_eq!(s.sum(), 0.0);
    }

    #[test]
    fn test_min_max_all_nulls() {
        let s = Series::with_nulls("x", vec![0_i32, 0, 0], vec![true, true, true]).unwrap();
        assert_eq!(s.min(), None);
        assert_eq!(s.max(), None);
    }

    #[test]
    fn test_mean_single_element() {
        let s = Series::new("x", vec![7.0_f64]);
        assert_eq!(s.mean(), 7.0);
    }

    #[test]
    fn test_var_single_element() {
        let s = Series::new("x", vec![7.0_f64]);
        assert_eq!(s.var(), 0.0);
    }

    #[test]
    fn test_median_single_element() {
        let s = Series::new("x", vec![7.0_f64]);
        assert_eq!(s.median(), 7.0);
    }

    #[test]
    fn test_median_all_nulls() {
        let s = Series::with_nulls("x", vec![0.0_f64, 0.0], vec![true, true]).unwrap();
        assert!(s.median().is_nan());
    }

    #[test]
    fn test_min_max_single_element() {
        let s = Series::new("x", vec![42_i32]);
        assert_eq!(s.min(), Some(42));
        assert_eq!(s.max(), Some(42));
    }

    #[test]
    fn test_var_std_empty() {
        let s: Series<f64> = Series::new("x", vec![]);
        assert!(s.var().is_nan());
        assert!(s.std().is_nan());
    }

    #[test]
    fn test_product_with_nulls() {
        let s = Series::with_nulls("x", vec![2_i32, 0, 3], vec![false, true, false]).unwrap();
        // null is skipped, product = 2 * 3 = 6
        assert_eq!(s.product(), 6);
    }

    #[test]
    fn test_apply_preserves_nulls() {
        let s = Series::with_nulls("x", vec![1_i32, 0, 3], vec![false, true, false]).unwrap();
        let doubled = s.apply(|v| v * 2_i32);
        assert_eq!(doubled.get(0), Some(2));
        assert_eq!(doubled.get(1), None); // null preserved
        assert_eq!(doubled.get(2), Some(6));
    }

    #[test]
    fn test_mean_all_nulls() {
        let s = Series::with_nulls("x", vec![0.0_f64, 0.0], vec![true, true]).unwrap();
        assert!(s.mean().is_nan());
    }

    #[test]
    fn test_sum_single_element() {
        let s = Series::new("x", vec![42_i32]);
        assert_eq!(s.sum(), 42);
    }

    #[test]
    fn test_product_single_element() {
        let s = Series::new("x", vec![7_i32]);
        assert_eq!(s.product(), 7);
    }
}
