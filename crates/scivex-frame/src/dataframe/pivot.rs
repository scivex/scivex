//! Pivot, melt, and crosstab operations for [`DataFrame`].

use std::collections::HashMap;

use crate::dataframe::DataFrame;
use crate::error::{FrameError, Result};
use crate::groupby::AggFunc;
use crate::series::string::StringSeries;
use crate::series::{AnySeries, Series};

impl DataFrame {
    /// Reshape from long to wide format.
    ///
    /// Groups rows by `index` columns, creates one output column per unique
    /// value in `columns`, and fills cells with the aggregated `values` column.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// # use scivex_frame::groupby::AggFunc;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("region", &["East", "West", "East"])))
    ///     .add_boxed(Box::new(StringSeries::from_strs("product", &["A", "A", "B"])))
    ///     .add_column("sales", vec![100.0_f64, 200.0, 150.0])
    ///     .build()
    ///     .unwrap();
    /// let wide = df.pivot(&["region"], "product", "sales", AggFunc::Sum).unwrap();
    /// assert_eq!(wide.nrows(), 2);
    /// ```
    pub fn pivot(
        &self,
        index: &[&str],
        columns: &str,
        values: &str,
        agg_func: AggFunc,
    ) -> Result<DataFrame> {
        // Validate
        for &idx in index {
            self.column(idx)?;
        }
        let pivot_col = self.column(columns)?;
        self.column(values)?;

        // Find unique pivot column values (preserving order)
        let nrows = self.nrows();
        let mut pivot_values: Vec<String> = Vec::new();
        let mut seen: HashMap<String, usize> = HashMap::new();
        for row in 0..nrows {
            let val = pivot_col.display_value(row);
            if !seen.contains_key(&val) {
                seen.insert(val.clone(), pivot_values.len());
                pivot_values.push(val);
            }
        }

        // Build groups: composite index key → HashMap<pivot_value, Vec<row_index>>
        let mut groups: HashMap<String, HashMap<String, Vec<usize>>> = HashMap::new();
        let mut group_order: Vec<(String, Vec<String>)> = Vec::new();

        for row in 0..nrows {
            let idx_key_parts: Vec<String> = index
                .iter()
                .map(|&c| {
                    self.column(c)
                        .expect("pivot index column exists")
                        .display_value(row)
                })
                .collect();
            let idx_key = idx_key_parts.join("\x00");
            let pval = pivot_col.display_value(row);

            let group = groups.entry(idx_key.clone()).or_default();
            if group.is_empty() {
                group_order.push((idx_key, idx_key_parts));
            }
            group.entry(pval).or_default().push(row);
        }

        // Build result: index columns + one column per pivot value
        let mut result_cols: Vec<Box<dyn AnySeries>> = Vec::new();

        // Index columns
        for (ki, &idx_name) in index.iter().enumerate() {
            let data: Vec<String> = group_order
                .iter()
                .map(|(_, parts)| parts[ki].clone())
                .collect();
            result_cols.push(Box::new(StringSeries::new(idx_name.to_string(), data)));
        }

        // Value columns — one per pivot value
        let values_col = self.column(values)?;
        for pval in &pivot_values {
            let col_name = format!("{values}_{pval}");
            let mut agg_data: Vec<f64> = Vec::new();
            let mut null_mask: Vec<bool> = Vec::new();

            for (idx_key, _) in &group_order {
                let group = &groups[idx_key];
                if let Some(rows) = group.get(pval) {
                    let agg_val = aggregate_rows(values_col, rows, agg_func);
                    agg_data.push(agg_val);
                    null_mask.push(false);
                } else {
                    agg_data.push(0.0);
                    null_mask.push(true);
                }
            }

            let has_nulls = null_mask.iter().any(|&v| v);
            if has_nulls {
                result_cols.push(Box::new(
                    Series::with_nulls(col_name, agg_data, null_mask)
                        .expect("valid pivot construction"),
                ));
            } else {
                result_cols.push(Box::new(Series::new(col_name, agg_data)));
            }
        }

        DataFrame::new(result_cols)
    }

    /// Reshape from wide to long format.
    ///
    /// `id_vars` are columns to keep as-is, `value_vars` are columns to unpivot.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("id", &["a", "b"])))
    ///     .add_column("x", vec![1.0_f64, 2.0])
    ///     .add_column("y", vec![3.0_f64, 4.0])
    ///     .build()
    ///     .unwrap();
    /// let long = df.melt(&["id"], &["x", "y"], None, None).unwrap();
    /// assert_eq!(long.nrows(), 4); // 2 rows * 2 value_vars
    /// ```
    pub fn melt(
        &self,
        id_vars: &[&str],
        value_vars: &[&str],
        var_name: Option<&str>,
        value_name: Option<&str>,
    ) -> Result<DataFrame> {
        // Validate
        for &col in id_vars {
            self.column(col)?;
        }
        for &col in value_vars {
            self.column(col)?;
        }

        if value_vars.is_empty() {
            return Err(FrameError::InvalidArgument {
                reason: "value_vars must not be empty",
            });
        }

        let variable_col_name = var_name.unwrap_or("variable");
        let value_col_name = value_name.unwrap_or("value");
        let nrows = self.nrows();
        let n_value_vars = value_vars.len();
        let total_rows = nrows * n_value_vars;

        // Build id columns (repeated n_value_vars times)
        let mut result_cols: Vec<Box<dyn AnySeries>> = Vec::new();
        for &id_col_name in id_vars {
            let src = self.column(id_col_name)?;
            let mut indices = Vec::with_capacity(total_rows);
            for _ in 0..n_value_vars {
                for row in 0..nrows {
                    indices.push(row);
                }
            }
            result_cols.push(src.take_indices(&indices).rename_box(id_col_name));
        }

        // Variable name column
        let mut variable_data: Vec<String> = Vec::with_capacity(total_rows);
        for &vv in value_vars {
            for _ in 0..nrows {
                variable_data.push(vv.to_string());
            }
        }
        result_cols.push(Box::new(StringSeries::new(
            variable_col_name.to_string(),
            variable_data,
        )));

        // Value column — use display_value since dtypes may differ
        let mut value_data: Vec<String> = Vec::with_capacity(total_rows);
        let mut value_nulls: Vec<bool> = Vec::with_capacity(total_rows);
        for &vv in value_vars {
            let col = self.column(vv)?;
            for row in 0..nrows {
                if col.is_null(row) {
                    value_data.push(String::new());
                    value_nulls.push(true);
                } else {
                    value_data.push(col.display_value(row));
                    value_nulls.push(false);
                }
            }
        }

        let has_nulls = value_nulls.iter().any(|&v| v);
        if has_nulls {
            result_cols.push(Box::new(
                StringSeries::with_nulls(value_col_name.to_string(), value_data, value_nulls)
                    .expect("valid pivot construction"),
            ));
        } else {
            result_cols.push(Box::new(StringSeries::new(
                value_col_name.to_string(),
                value_data,
            )));
        }

        DataFrame::new(result_cols)
    }

    /// Cross-tabulation: count occurrences of each combination of two columns.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::prelude::*;
    /// let df = DataFrame::builder()
    ///     .add_boxed(Box::new(StringSeries::from_strs("animal", &["cat", "cat", "dog"])))
    ///     .add_boxed(Box::new(StringSeries::from_strs("color", &["black", "white", "black"])))
    ///     .build()
    ///     .unwrap();
    /// let ct = df.crosstab("animal", "color").unwrap();
    /// assert_eq!(ct.nrows(), 2);
    /// ```
    pub fn crosstab(&self, index: &str, columns: &str) -> Result<DataFrame> {
        self.pivot(&[index], columns, index, AggFunc::Count)
    }
}

/// Aggregate a set of rows from a column into a single f64.
fn aggregate_rows(col: &dyn AnySeries, rows: &[usize], func: AggFunc) -> f64 {
    match func {
        AggFunc::Count => rows.len() as f64,
        AggFunc::Sum => rows
            .iter()
            .filter(|&&r| !col.is_null(r))
            .map(|&r| parse_display_value(col, r))
            .sum(),
        AggFunc::Mean => {
            let vals: Vec<f64> = rows
                .iter()
                .filter(|&&r| !col.is_null(r))
                .map(|&r| parse_display_value(col, r))
                .collect();
            if vals.is_empty() {
                0.0
            } else {
                vals.iter().sum::<f64>() / vals.len() as f64
            }
        }
        AggFunc::Min => rows
            .iter()
            .filter(|&&r| !col.is_null(r))
            .map(|&r| parse_display_value(col, r))
            .fold(f64::INFINITY, f64::min),
        AggFunc::Max => rows
            .iter()
            .filter(|&&r| !col.is_null(r))
            .map(|&r| parse_display_value(col, r))
            .fold(f64::NEG_INFINITY, f64::max),
        AggFunc::First => {
            if let Some(&r) = rows.first() {
                parse_display_value(col, r)
            } else {
                0.0
            }
        }
        AggFunc::Last => {
            if let Some(&r) = rows.last() {
                parse_display_value(col, r)
            } else {
                0.0
            }
        }
    }
}

fn parse_display_value(col: &dyn AnySeries, row: usize) -> f64 {
    col.display_value(row).parse::<f64>().unwrap_or(0.0)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn sales_df() -> DataFrame {
        DataFrame::new(vec![
            Box::new(StringSeries::from_strs(
                "region",
                &["East", "East", "West", "West", "East", "West"],
            )),
            Box::new(StringSeries::from_strs(
                "product",
                &["A", "B", "A", "B", "A", "A"],
            )),
            Box::new(Series::new(
                "sales",
                vec![100.0_f64, 150.0, 200.0, 250.0, 120.0, 180.0],
            )),
        ])
        .unwrap()
    }

    #[test]
    fn test_pivot_sum() {
        let df = sales_df();
        let result = df
            .pivot(&["region"], "product", "sales", AggFunc::Sum)
            .unwrap();
        // Two regions, two products → 2 rows, 3 columns (region, sales_A, sales_B)
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 3);
    }

    #[test]
    fn test_pivot_mean() {
        let df = sales_df();
        let result = df
            .pivot(&["region"], "product", "sales", AggFunc::Mean)
            .unwrap();
        assert_eq!(result.nrows(), 2);
        // East A: (100+120)/2=110, East B: 150
        let col_a = result.column_typed::<f64>("sales_A").unwrap();
        assert!((col_a.get(0).unwrap() - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_pivot_missing_cells() {
        let df = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("g", &["a", "b"])),
            Box::new(StringSeries::from_strs("p", &["x", "y"])),
            Box::new(Series::new("v", vec![1.0_f64, 2.0])),
        ])
        .unwrap();
        let result = df.pivot(&["g"], "p", "v", AggFunc::Sum).unwrap();
        // a only has x, b only has y → missing cells should be null
        assert_eq!(result.nrows(), 2);
        let vx = result.column("v_x").unwrap();
        let vy = result.column("v_y").unwrap();
        assert!(!vx.is_null(0)); // a,x exists
        assert!(vx.is_null(1)); // b,x missing
        assert!(vy.is_null(0)); // a,y missing
        assert!(!vy.is_null(1)); // b,y exists
    }

    #[test]
    fn test_melt_basic() {
        let df = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("id", &["a", "b"])),
            Box::new(Series::new("x", vec![1.0_f64, 2.0])),
            Box::new(Series::new("y", vec![3.0_f64, 4.0])),
        ])
        .unwrap();
        let result = df.melt(&["id"], &["x", "y"], None, None).unwrap();
        assert_eq!(result.nrows(), 4); // 2 rows × 2 value_vars
        assert_eq!(result.ncols(), 3); // id, variable, value
    }

    #[test]
    fn test_melt_custom_names() {
        let df = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("id", &["a"])),
            Box::new(Series::new("x", vec![1.0_f64])),
        ])
        .unwrap();
        let result = df
            .melt(&["id"], &["x"], Some("metric"), Some("measurement"))
            .unwrap();
        assert!(result.column("metric").is_ok());
        assert!(result.column("measurement").is_ok());
    }

    #[test]
    fn test_melt_subset() {
        let df = DataFrame::new(vec![
            Box::new(StringSeries::from_strs("id", &["a", "b"])),
            Box::new(Series::new("x", vec![1.0_f64, 2.0])),
            Box::new(Series::new("y", vec![3.0_f64, 4.0])),
            Box::new(Series::new("z", vec![5.0_f64, 6.0])),
        ])
        .unwrap();
        // Only melt x and y, not z
        let result = df.melt(&["id"], &["x", "y"], None, None).unwrap();
        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 3);
    }

    #[test]
    fn test_crosstab() {
        let df = DataFrame::new(vec![
            Box::new(StringSeries::from_strs(
                "animal",
                &["cat", "cat", "dog", "dog", "cat"],
            )),
            Box::new(StringSeries::from_strs(
                "color",
                &["black", "white", "black", "black", "black"],
            )),
        ])
        .unwrap();
        let result = df.crosstab("animal", "color").unwrap();
        assert_eq!(result.nrows(), 2); // cat, dog
    }
}
