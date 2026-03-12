//! Support Vector Machine classifiers and regressors.

mod kernel;
mod svc;
mod svr;

pub use kernel::Kernel;
pub use svc::SVC;
pub use svr::SVR;
