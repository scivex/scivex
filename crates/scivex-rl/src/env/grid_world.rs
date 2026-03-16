//! Grid world environment.

use super::{Environment, StepResult};

/// A simple grid world environment.
///
/// The agent starts at (0, 0) and must reach (N-1, N-1) in an NxN grid.
/// There are 4 discrete actions: 0 = up, 1 = down, 2 = left, 3 = right.
/// Observations are flat indices `row * N + col`.
pub struct GridWorld {
    size: usize,
    row: usize,
    col: usize,
    step_count: usize,
    max_steps: usize,
    done: bool,
    obs_shape: Vec<usize>,
}

impl GridWorld {
    /// Create a new `GridWorld` environment with the given grid size.
    ///
    /// # Errors
    ///
    /// Returns an error if `size` is zero.
    pub fn new(size: usize) -> crate::error::Result<Self> {
        if size == 0 {
            return Err(crate::error::RlError::InvalidParameter(
                "grid size must be positive".to_string(),
            ));
        }
        Ok(Self {
            size,
            row: 0,
            col: 0,
            step_count: 0,
            max_steps: size * size * 4,
            done: false,
            obs_shape: vec![1],
        })
    }

    fn flat_index(&self) -> usize {
        self.row * self.size + self.col
    }

    fn is_goal(&self) -> bool {
        self.row == self.size - 1 && self.col == self.size - 1
    }
}

impl Environment for GridWorld {
    type Observation = Vec<f64>;
    type Action = usize;

    fn reset(&mut self) -> Self::Observation {
        self.row = 0;
        self.col = 0;
        self.step_count = 0;
        self.done = false;
        vec![self.flat_index() as f64]
    }

    #[allow(clippy::collapsible_if)]
    fn step(&mut self, action: &Self::Action) -> StepResult<Self::Observation> {
        match *action {
            0 => {
                // up
                if self.row > 0 {
                    self.row -= 1;
                }
            }
            1 => {
                // down
                if self.row < self.size - 1 {
                    self.row += 1;
                }
            }
            2 => {
                // left
                if self.col > 0 {
                    self.col -= 1;
                }
            }
            3 => {
                // right
                if self.col < self.size - 1 {
                    self.col += 1;
                }
            }
            _ => {} // invalid action, no-op
        }

        self.step_count += 1;

        let goal = self.is_goal();
        let truncated = self.step_count >= self.max_steps;
        self.done = goal || truncated;

        let reward = if goal { 10.0 } else { -1.0 };

        StepResult {
            observation: vec![self.flat_index() as f64],
            reward,
            done: goal,
            truncated,
        }
    }

    fn observation_shape(&self) -> &[usize] {
        &self.obs_shape
    }

    fn action_count(&self) -> usize {
        4
    }

    fn is_done(&self) -> bool {
        self.done
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_world_reset() {
        let mut env = GridWorld::new(5).unwrap();
        let obs = env.reset();
        assert_eq!(obs, vec![0.0]);
        assert!(!env.is_done());
    }

    #[test]
    fn test_grid_world_navigation() {
        let mut env = GridWorld::new(3).unwrap();
        env.reset();
        // Move right twice, down twice to reach (2,2)
        env.step(&3); // right -> (0,1)
        env.step(&3); // right -> (0,2)
        env.step(&1); // down -> (1,2)
        let result = env.step(&1); // down -> (2,2) = goal
        assert_eq!(result.reward, 10.0);
        assert!(result.done);
    }

    #[test]
    fn test_grid_world_boundary() {
        let mut env = GridWorld::new(3).unwrap();
        env.reset();
        // Try moving up from (0,0) — should stay at (0,0)
        let result = env.step(&0);
        assert_eq!(result.observation, vec![0.0]);
        assert_eq!(result.reward, -1.0);
    }

    #[test]
    fn test_grid_world_invalid_size() {
        assert!(GridWorld::new(0).is_err());
    }
}
