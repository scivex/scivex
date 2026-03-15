//! DataFrame schema validation.
//!
//! A `Schema` describes the expected structure of a [`DataFrame`]: column
//! names, data types, nullability, and value constraints. Use
//! [`DataFrame::validate`] to check conformance.
//!
//! # Example
//!
//! ```
//! use scivex_frame::prelude::*;
//! use scivex_frame::schema::{Schema, ColumnSchema, Constraint};
//!
//! let schema = Schema::new(vec![
//!     ColumnSchema::new("age", DType::I64).not_null(),
//!     ColumnSchema::new("name", DType::Str),
//! ]);
//!
//! let df = DataFrame::builder()
//!     .add_column("age", vec![25_i64, 30])
//!     .add_boxed(Box::new(StringSeries::from_strs("name", &["Alice", "Bob"])))
//!     .build()
//!     .unwrap();
//!
//! let errors = df.validate(&schema);
//! assert!(errors.is_empty());
//! ```

use crate::dataframe::DataFrame;
use crate::dtype::DType;

use std::fmt;

/// A schema describing the expected structure of a [`DataFrame`].
#[derive(Debug, Clone)]
pub struct Schema {
    columns: Vec<ColumnSchema>,
}

/// Schema definition for a single column.
#[derive(Debug, Clone)]
pub struct ColumnSchema {
    /// Column name.
    pub name: String,
    /// Expected data type.
    pub dtype: DType,
    /// Whether nulls are allowed (default: true).
    pub nullable: bool,
    /// Value constraints.
    pub constraints: Vec<Constraint>,
}

/// A value constraint for a column.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// All values must be unique.
    Unique,
    /// Numeric values must fall within `[min, max]`.
    Range { min: f64, max: f64 },
    /// String values must be one of the given options.
    OneOf(Vec<String>),
    /// Minimum string length.
    MinLength(usize),
    /// Maximum string length.
    MaxLength(usize),
}

/// A single validation error.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Which column failed validation.
    pub column: String,
    /// Description of the validation failure.
    pub reason: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "column {:?}: {}", self.column, self.reason)
    }
}

// ---------------------------------------------------------------------------
// Schema construction
// ---------------------------------------------------------------------------

impl Schema {
    /// Create a schema from column definitions.
    pub fn new(columns: Vec<ColumnSchema>) -> Self {
        Self { columns }
    }

    /// Column schemas.
    pub fn columns(&self) -> &[ColumnSchema] {
        &self.columns
    }
}

impl ColumnSchema {
    /// Create a column schema with the given name and type.
    pub fn new(name: &str, dtype: DType) -> Self {
        Self {
            name: name.to_string(),
            dtype,
            nullable: true,
            constraints: Vec::new(),
        }
    }

    /// Mark this column as non-nullable.
    pub fn not_null(mut self) -> Self {
        self.nullable = false;
        self
    }

    /// Add a constraint.
    pub fn constraint(mut self, c: Constraint) -> Self {
        self.constraints.push(c);
        self
    }

    /// Add a uniqueness constraint.
    pub fn unique(self) -> Self {
        self.constraint(Constraint::Unique)
    }

    /// Add a numeric range constraint.
    pub fn range(self, min: f64, max: f64) -> Self {
        self.constraint(Constraint::Range { min, max })
    }

    /// Add a set-membership constraint.
    pub fn one_of(self, values: &[&str]) -> Self {
        self.constraint(Constraint::OneOf(
            values.iter().map(|s| (*s).to_string()).collect(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

impl DataFrame {
    /// Validate this `DataFrame` against a [`Schema`].
    ///
    /// Returns a list of [`ValidationError`]s. An empty list means the
    /// `DataFrame` is valid.
    pub fn validate(&self, schema: &Schema) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for col_schema in schema.columns() {
            validate_column(self, col_schema, &mut errors);
        }

        errors
    }
}

fn validate_column(df: &DataFrame, cs: &ColumnSchema, errors: &mut Vec<ValidationError>) {
    // Check column exists
    let Ok(col) = df.column(&cs.name) else {
        errors.push(ValidationError {
            column: cs.name.clone(),
            reason: "column not found".to_string(),
        });
        return;
    };

    // Check dtype
    if col.dtype() != cs.dtype {
        errors.push(ValidationError {
            column: cs.name.clone(),
            reason: format!("expected type {:?}, got {:?}", cs.dtype, col.dtype()),
        });
    }

    // Check nullability
    if !cs.nullable && col.null_count() > 0 {
        errors.push(ValidationError {
            column: cs.name.clone(),
            reason: format!("column has {} nulls but is not nullable", col.null_count()),
        });
    }

    // Check constraints
    for constraint in &cs.constraints {
        validate_constraint(col, &cs.name, constraint, errors);
    }
}

fn validate_constraint(
    col: &dyn crate::series::AnySeries,
    col_name: &str,
    constraint: &Constraint,
    errors: &mut Vec<ValidationError>,
) {
    match constraint {
        Constraint::Unique => check_unique(col, col_name, errors),
        Constraint::Range { min, max } => check_range(col, col_name, *min, *max, errors),
        Constraint::OneOf(values) => check_one_of(col, col_name, values, errors),
        Constraint::MinLength(min) => check_min_length(col, col_name, *min, errors),
        Constraint::MaxLength(max) => check_max_length(col, col_name, *max, errors),
    }
}

fn check_unique(
    col: &dyn crate::series::AnySeries,
    col_name: &str,
    errors: &mut Vec<ValidationError>,
) {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    for i in 0..col.len() {
        if col.is_null(i) {
            continue;
        }
        let val = col.display_value(i);
        if !seen.insert(val) {
            errors.push(ValidationError {
                column: col_name.to_string(),
                reason: "duplicate values found".to_string(),
            });
            return;
        }
    }
}

fn check_range(
    col: &dyn crate::series::AnySeries,
    col_name: &str,
    min: f64,
    max: f64,
    errors: &mut Vec<ValidationError>,
) {
    for i in 0..col.len() {
        if col.is_null(i) {
            continue;
        }
        #[allow(clippy::collapsible_if)]
        if let Ok(v) = col.display_value(i).parse::<f64>() {
            if v < min || v > max {
                errors.push(ValidationError {
                    column: col_name.to_string(),
                    reason: format!("value {v} out of range [{min}, {max}]"),
                });
                return;
            }
        }
    }
}

fn check_one_of(
    col: &dyn crate::series::AnySeries,
    col_name: &str,
    values: &[String],
    errors: &mut Vec<ValidationError>,
) {
    for i in 0..col.len() {
        if col.is_null(i) {
            continue;
        }
        let val = col.display_value(i);
        if !values.iter().any(|v| v == &val) {
            errors.push(ValidationError {
                column: col_name.to_string(),
                reason: format!("value {val:?} not in allowed set"),
            });
            return;
        }
    }
}

fn check_min_length(
    col: &dyn crate::series::AnySeries,
    col_name: &str,
    min: usize,
    errors: &mut Vec<ValidationError>,
) {
    for i in 0..col.len() {
        if col.is_null(i) {
            continue;
        }
        let val = col.display_value(i);
        if val.len() < min {
            errors.push(ValidationError {
                column: col_name.to_string(),
                reason: format!("value {val:?} shorter than minimum length {min}"),
            });
            return;
        }
    }
}

fn check_max_length(
    col: &dyn crate::series::AnySeries,
    col_name: &str,
    max: usize,
    errors: &mut Vec<ValidationError>,
) {
    for i in 0..col.len() {
        if col.is_null(i) {
            continue;
        }
        let val = col.display_value(i);
        if val.len() > max {
            errors.push(ValidationError {
                column: col_name.to_string(),
                reason: format!("value {val:?} exceeds maximum length {max}"),
            });
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;
    use crate::series::string::StringSeries;

    #[test]
    fn test_valid_schema() {
        let schema = Schema::new(vec![
            ColumnSchema::new("a", DType::I32),
            ColumnSchema::new("b", DType::F64),
        ]);

        let df = DataFrame::new(vec![
            Box::new(Series::new("a", vec![1_i32, 2, 3])),
            Box::new(Series::new("b", vec![1.0_f64, 2.0, 3.0])),
        ])
        .unwrap();

        assert!(df.validate(&schema).is_empty());
    }

    #[test]
    fn test_missing_column() {
        let schema = Schema::new(vec![ColumnSchema::new("missing", DType::I32)]);
        let df = DataFrame::new(vec![Box::new(Series::new("a", vec![1_i32]))]).unwrap();

        let errors = df.validate(&schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].reason.contains("not found"));
    }

    #[test]
    fn test_wrong_dtype() {
        let schema = Schema::new(vec![ColumnSchema::new("a", DType::F64)]);
        let df = DataFrame::new(vec![Box::new(Series::new("a", vec![1_i32]))]).unwrap();

        let errors = df.validate(&schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].reason.contains("expected type"));
    }

    #[test]
    fn test_not_null_constraint() {
        let schema = Schema::new(vec![ColumnSchema::new("a", DType::I32).not_null()]);
        let df = DataFrame::new(vec![Box::new(
            Series::with_nulls("a", vec![1_i32, 0], vec![false, true]).unwrap(),
        )])
        .unwrap();

        let errors = df.validate(&schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].reason.contains("nulls"));
    }

    #[test]
    fn test_unique_constraint() {
        let schema = Schema::new(vec![ColumnSchema::new("a", DType::I32).unique()]);
        let df = DataFrame::new(vec![Box::new(Series::new("a", vec![1_i32, 2, 1]))]).unwrap();

        let errors = df.validate(&schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].reason.contains("duplicate"));
    }

    #[test]
    fn test_range_constraint() {
        let schema = Schema::new(vec![ColumnSchema::new("a", DType::F64).range(0.0, 10.0)]);
        let df = DataFrame::new(vec![Box::new(Series::new("a", vec![5.0_f64, 15.0]))]).unwrap();

        let errors = df.validate(&schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].reason.contains("out of range"));
    }

    #[test]
    fn test_one_of_constraint() {
        let schema = Schema::new(vec![
            ColumnSchema::new("status", DType::Str).one_of(&["active", "inactive"]),
        ]);
        let df = DataFrame::new(vec![Box::new(StringSeries::from_strs(
            "status",
            &["active", "deleted"],
        ))])
        .unwrap();

        let errors = df.validate(&schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].reason.contains("not in allowed set"));
    }

    #[test]
    fn test_valid_all_constraints() {
        let schema = Schema::new(vec![
            ColumnSchema::new("id", DType::I32).not_null().unique(),
            ColumnSchema::new("score", DType::F64).range(0.0, 100.0),
        ]);

        let df = DataFrame::new(vec![
            Box::new(Series::new("id", vec![1_i32, 2, 3])),
            Box::new(Series::new("score", vec![95.0_f64, 80.0, 100.0])),
        ])
        .unwrap();

        assert!(df.validate(&schema).is_empty());
    }

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError {
            column: "age".to_string(),
            reason: "value out of range".to_string(),
        };
        let s = format!("{err}");
        assert!(s.contains("age"));
        assert!(s.contains("value out of range"));
    }

    #[test]
    fn test_multiple_errors() {
        let schema = Schema::new(vec![
            ColumnSchema::new("a", DType::F64),       // wrong type
            ColumnSchema::new("missing", DType::I32), // not found
        ]);

        let df = DataFrame::new(vec![Box::new(Series::new("a", vec![1_i32]))]).unwrap();

        let errors = df.validate(&schema);
        assert_eq!(errors.len(), 2);
    }
}
