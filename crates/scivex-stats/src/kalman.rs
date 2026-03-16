//! Kalman filter for linear state estimation.

use scivex_core::Float;
use scivex_core::Tensor;

use crate::error::{Result, StatsError};

/// A linear Kalman filter.
///
/// Maintains a state estimate `x` and covariance `P` that are updated via
/// `predict` (time update) and `update` (measurement update) steps.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct KalmanFilter<T: Float> {
    /// State dimension.
    dim: usize,
    /// Current state vector.
    state: Vec<T>,
    /// Current covariance matrix (dim x dim, row-major).
    cov: Tensor<T>,
}

impl<T: Float> KalmanFilter<T> {
    /// Create a new Kalman filter.
    ///
    /// - `state_dim`: dimension of the state vector.
    /// - `initial_state`: initial state estimate of length `state_dim`.
    /// - `initial_cov`: initial covariance matrix `[state_dim x state_dim]`.
    pub fn new(state_dim: usize, initial_state: &[T], initial_cov: &Tensor<T>) -> Result<Self> {
        if initial_state.len() != state_dim {
            return Err(StatsError::LengthMismatch {
                expected: state_dim,
                got: initial_state.len(),
            });
        }
        if initial_cov.ndim() != 2
            || initial_cov.shape()[0] != state_dim
            || initial_cov.shape()[1] != state_dim
        {
            return Err(StatsError::InvalidParameter {
                name: "initial_cov",
                reason: "must be a square matrix matching state_dim",
            });
        }
        Ok(Self {
            dim: state_dim,
            state: initial_state.to_vec(),
            cov: initial_cov.clone(),
        })
    }

    /// Predict step (time update).
    ///
    /// - `transition`: state transition matrix `F` `[dim x dim]`.
    /// - `process_noise`: process noise covariance `Q` `[dim x dim]`.
    ///
    /// Updates: `x = F * x`, `P = F * P * F^T + Q`.
    pub fn predict(&mut self, transition: &Tensor<T>, process_noise: &Tensor<T>) -> Result<()> {
        self.check_square(transition, "transition")?;
        self.check_square(process_noise, "process_noise")?;

        let x_t = Tensor::from_slice(&self.state, vec![self.dim])?;
        let new_x = transition.matvec(&x_t).map_err(StatsError::CoreError)?;
        self.state = new_x.as_slice().to_vec();

        let fp = transition
            .matmul(&self.cov)
            .map_err(StatsError::CoreError)?;
        let ft = transition.transpose().map_err(StatsError::CoreError)?;
        let fpft = fp.matmul(&ft).map_err(StatsError::CoreError)?;

        let fpft_s = fpft.as_slice();
        let q_s = process_noise.as_slice();
        let mut new_p = Vec::with_capacity(self.dim * self.dim);
        for i in 0..self.dim * self.dim {
            new_p.push(fpft_s[i] + q_s[i]);
        }
        self.cov = Tensor::from_vec(new_p, vec![self.dim, self.dim])?;
        Ok(())
    }

    /// Update step (measurement update).
    ///
    /// - `observation`: observed measurement vector of length `m`.
    /// - `obs_matrix`: observation matrix `H` `[m x dim]`.
    /// - `obs_noise`: observation noise covariance `R` `[m x m]`.
    #[allow(clippy::similar_names)]
    pub fn update(
        &mut self,
        observation: &[T],
        obs_matrix: &Tensor<T>,
        obs_noise: &Tensor<T>,
    ) -> Result<()> {
        let m = observation.len();
        if obs_matrix.ndim() != 2 || obs_matrix.shape()[0] != m || obs_matrix.shape()[1] != self.dim
        {
            return Err(StatsError::InvalidParameter {
                name: "obs_matrix",
                reason: "must be [m x dim]",
            });
        }
        if obs_noise.ndim() != 2 || obs_noise.shape()[0] != m || obs_noise.shape()[1] != m {
            return Err(StatsError::InvalidParameter {
                name: "obs_noise",
                reason: "must be [m x m]",
            });
        }

        let x_t = Tensor::from_slice(&self.state, vec![self.dim])?;

        // Innovation: y = z - H * x
        let hx = obs_matrix.matvec(&x_t).map_err(StatsError::CoreError)?;
        let hx_s = hx.as_slice();
        let mut innov = Vec::with_capacity(m);
        for i in 0..m {
            innov.push(observation[i] - hx_s[i]);
        }

        // S = H * P * H^T + R
        let hp = obs_matrix
            .matmul(&self.cov)
            .map_err(StatsError::CoreError)?;
        let ht = obs_matrix.transpose().map_err(StatsError::CoreError)?;
        let hpht = hp.matmul(&ht).map_err(StatsError::CoreError)?;

        let hpht_s = hpht.as_slice();
        let r_s = obs_noise.as_slice();
        let mut s_data = Vec::with_capacity(m * m);
        for i in 0..m * m {
            s_data.push(hpht_s[i] + r_s[i]);
        }
        let s_mat = Tensor::from_vec(s_data, vec![m, m])?;
        let s_inv = s_mat.inv().map_err(|_| StatsError::SingularMatrix)?;

        // K = P * H^T * S^{-1}
        let pht = self.cov.matmul(&ht).map_err(StatsError::CoreError)?;
        let k = pht.matmul(&s_inv).map_err(StatsError::CoreError)?;

        // x = x + K * y
        let innov_t = Tensor::from_slice(&innov, vec![m])?;
        let k_innov = k.matvec(&innov_t).map_err(StatsError::CoreError)?;
        let k_innov_s = k_innov.as_slice();
        for (st, &ki) in self.state.iter_mut().zip(k_innov_s.iter()) {
            *st += ki;
        }

        // P = (I - K * H) * P
        let kh = k.matmul(obs_matrix).map_err(StatsError::CoreError)?;
        let kh_s = kh.as_slice();
        let one = T::from_f64(1.0);
        let zero = T::from_f64(0.0);
        let mut ikh_data = Vec::with_capacity(self.dim * self.dim);
        for i in 0..self.dim {
            for j in 0..self.dim {
                let ident = if i == j { one } else { zero };
                ikh_data.push(ident - kh_s[i * self.dim + j]);
            }
        }
        let ikh = Tensor::from_vec(ikh_data, vec![self.dim, self.dim])?;
        self.cov = ikh.matmul(&self.cov).map_err(StatsError::CoreError)?;

        Ok(())
    }

    /// Return the current state estimate.
    pub fn state(&self) -> &[T] {
        &self.state
    }

    /// Return the current covariance matrix.
    pub fn covariance(&self) -> &Tensor<T> {
        &self.cov
    }

    fn check_square(&self, mat: &Tensor<T>, name: &'static str) -> Result<()> {
        if mat.ndim() != 2 || mat.shape()[0] != self.dim || mat.shape()[1] != self.dim {
            return Err(StatsError::InvalidParameter {
                name,
                reason: "must be a square matrix matching state_dim",
            });
        }
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn eye(n: usize) -> Tensor<f64> {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::from_vec(data, vec![n, n]).unwrap()
    }

    fn diag(vals: &[f64]) -> Tensor<f64> {
        let n = vals.len();
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = vals[i];
        }
        Tensor::from_vec(data, vec![n, n]).unwrap()
    }

    #[test]
    fn test_constant_state_converges() {
        let mut kf = KalmanFilter::new(1, &[0.0], &diag(&[100.0])).unwrap();
        let f = eye(1);
        let q = diag(&[0.0]);
        let h = Tensor::from_vec(vec![1.0], vec![1, 1]).unwrap();
        let r = diag(&[1.0]);

        for _ in 0..50 {
            kf.predict(&f, &q).unwrap();
            kf.update(&[5.0], &h, &r).unwrap();
        }

        assert!(
            (kf.state()[0] - 5.0).abs() < 0.2,
            "state = {}",
            kf.state()[0]
        );
    }

    #[test]
    fn test_linear_motion_tracking() {
        let dt = 1.0;
        let f_data = vec![1.0, dt, 0.0, 1.0];
        let f = Tensor::from_vec(f_data, vec![2, 2]).unwrap();
        let q = diag(&[0.01, 0.01]);
        let h = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let r = diag(&[0.5]);

        let mut kf = KalmanFilter::new(2, &[0.0, 0.0], &diag(&[100.0, 100.0])).unwrap();

        for t in 1..=30 {
            kf.predict(&f, &q).unwrap();
            let obs = 2.0 * f64::from(t);
            kf.update(&[obs], &h, &r).unwrap();
        }

        assert!(
            (kf.state()[0] - 60.0).abs() < 2.0,
            "pos = {}",
            kf.state()[0]
        );
        assert!((kf.state()[1] - 2.0).abs() < 0.5, "vel = {}", kf.state()[1]);
    }

    #[test]
    fn test_covariance_shrinks() {
        let mut kf = KalmanFilter::new(1, &[0.0], &diag(&[100.0])).unwrap();
        let f = eye(1);
        let q = diag(&[0.01]);
        let h = Tensor::from_vec(vec![1.0], vec![1, 1]).unwrap();
        let r = diag(&[1.0]);

        let initial_cov = kf.covariance().as_slice()[0];

        for _ in 0..20 {
            kf.predict(&f, &q).unwrap();
            kf.update(&[5.0], &h, &r).unwrap();
        }

        let final_cov = kf.covariance().as_slice()[0];
        assert!(final_cov < initial_cov, "cov should shrink");
    }

    #[test]
    fn test_dimension_mismatch() {
        let kf_result = KalmanFilter::new(2, &[0.0], &diag(&[1.0]));
        assert!(kf_result.is_err());
    }
}
