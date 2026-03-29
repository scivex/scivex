#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::module_name_repetitions
)]
//! # scivex-frame
//!
//! `DataFrame`s, `Series`, and `GroupBy` for the Scivex ecosystem.
//!
//! This crate provides:
//! - [`Series<T>`] — generic typed columns with compile-time safety
//! - [`StringSeries`] — string-specific column
//! - [`DataFrame`] — columnar data frame with type-erased columns
//! - [`GroupBy`] — group-by aggregation
//!
//! # Quick Example
//!
//! ```
//! use scivex_frame::prelude::*;
//!
//! let df = DataFrame::builder()
//!     .add_column("x", vec![1_i32, 2, 3])
//!     .add_column("y", vec![4.0_f64, 5.0, 6.0])
//!     .build()
//!     .unwrap();
//!
//! assert_eq!(df.shape(), (3, 2));
//! ```

/// DataFrame type and columnar operations.
pub mod dataframe;
/// Column data type enum and type introspection.
pub mod dtype;
/// Frame-specific error types.
pub mod error;
/// Group-by split-apply-combine aggregation.
pub mod groupby;
/// Lazy evaluation with expression trees and logical plans.
pub mod lazy;
/// Hierarchical (multi-level) row/column indexing.
pub mod multiindex;
/// Parallel DataFrame operations (requires `parallel` feature).
pub mod parallel;
/// DataFrame schema validation.
pub mod schema;
/// Typed series (columns) and type-erased column trait.
pub mod series;
/// SQL query engine on DataFrames.
pub mod sql;

// Re-export primary types at the crate root.
pub use dataframe::join::JoinType;
pub use dataframe::{DataFrame, DataFrameBuilder};
pub use dtype::DType;
pub use error::{FrameError, Result};
pub use groupby::{AggFunc, GroupBy};
pub use lazy::LazyFrame;
pub use multiindex::MultiIndex;
pub use series::categorical::CategoricalSeries;
pub use series::datetime::{DateTime, DateTimeSeries, Duration};
pub use series::string::StringSeries;
pub use series::window::RollingWindow;
pub use series::{AnySeries, Series};
pub use sql::{SqlContext, sql};

#[cfg(feature = "parallel")]
pub use parallel::{par_apply, par_filter, par_groupby_agg, par_sort};

/// Glob-import convenience: `use scivex_frame::prelude::*;`
pub mod prelude {
    pub use crate::dataframe::join::JoinType;
    pub use crate::dataframe::{DataFrame, DataFrameBuilder};
    pub use crate::dtype::DType;
    pub use crate::error::{FrameError, Result};
    pub use crate::groupby::{AggFunc, GroupBy};
    pub use crate::lazy::LazyFrame;
    pub use crate::multiindex::MultiIndex;
    pub use crate::series::categorical::CategoricalSeries;
    pub use crate::series::datetime::{DateTime, DateTimeSeries, Duration};
    pub use crate::series::string::StringSeries;
    pub use crate::series::window::RollingWindow;
    pub use crate::series::{AnySeries, Series};

    pub use crate::sql::{SqlContext, sql};

    #[cfg(feature = "parallel")]
    pub use crate::parallel::{par_apply, par_filter, par_groupby_agg, par_sort};
}
