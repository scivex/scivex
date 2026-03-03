//! # scivex-frame
//!
//! `DataFrame`s, `Series`, and `GroupBy` for the Scivex ecosystem.
//!
//! This crate provides:
//! - [`Series<T>`] — generic typed columns with compile-time safety
//! - [`StringSeries`](series::string::StringSeries) — string-specific column
//! - [`DataFrame`] — columnar data frame with type-erased columns
//! - [`GroupBy`](groupby::GroupBy) — group-by aggregation
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

pub mod dataframe;
pub mod dtype;
pub mod error;
pub mod groupby;
pub mod series;

// Re-export primary types at the crate root.
pub use dataframe::join::JoinType;
pub use dataframe::{DataFrame, DataFrameBuilder};
pub use dtype::DType;
pub use error::{FrameError, Result};
pub use groupby::{AggFunc, GroupBy};
pub use series::categorical::CategoricalSeries;
pub use series::string::StringSeries;
pub use series::window::RollingWindow;
pub use series::{AnySeries, Series};

/// Glob-import convenience: `use scivex_frame::prelude::*;`
pub mod prelude {
    pub use crate::dataframe::join::JoinType;
    pub use crate::dataframe::{DataFrame, DataFrameBuilder};
    pub use crate::dtype::DType;
    pub use crate::error::{FrameError, Result};
    pub use crate::groupby::{AggFunc, GroupBy};
    pub use crate::series::categorical::CategoricalSeries;
    pub use crate::series::string::StringSeries;
    pub use crate::series::window::RollingWindow;
    pub use crate::series::{AnySeries, Series};
}
