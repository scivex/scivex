//! Matrix decompositions.
//!
//! | Decomposition | Module       | Factorization           |
//! |---------------|-------------|-------------------------|
//! | LU            | [`lu`]      | `PA = LU`               |
//! | QR            | [`qr`]      | `A = QR`                |
//! | Cholesky      | [`cholesky`]| `A = L L^T`             |
//! | SVD           | [`svd`]     | `A = U diag(s) V^T`     |
//! | Eigen         | [`eig`]     | `A = V diag(d) V^T`     |

pub mod cholesky;
pub mod eig;
pub mod lu;
pub mod qr;
pub mod svd;

pub use cholesky::CholeskyDecomposition;
pub use eig::EigDecomposition;
pub use lu::LuDecomposition;
pub use qr::{QrDecomposition, lstsq};
pub use svd::SvdDecomposition;
