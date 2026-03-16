//! Classic cart-pole balancing environment.

use super::{Environment, StepResult};

const GRAVITY: f64 = 9.8;
const MASS_CART: f64 = 1.0;
const MASS_POLE: f64 = 0.1;
const TOTAL_MASS: f64 = MASS_CART + MASS_POLE;
const LENGTH: f64 = 0.5; // half-pole length
const POLE_MASS_LENGTH: f64 = MASS_POLE * LENGTH;
const FORCE_MAG: f64 = 10.0;
const TAU: f64 = 0.02; // time step
const THETA_THRESHOLD: f64 = 12.0 * std::f64::consts::PI / 180.0;
const X_THRESHOLD: f64 = 2.4;
const MAX_STEPS: usize = 500;

const OBS_SHAPE: [usize; 1] = [4];

/// Cart-pole balancing environment.
///
/// The agent must balance a pole on a cart by pushing left or right.
/// Observations are `[x, x_dot, theta, theta_dot]` and there are 2
/// discrete actions (0 = push left, 1 = push right).
pub struct CartPole {
    x: f64,
    x_dot: f64,
    theta: f64,
    theta_dot: f64,
    step_count: usize,
    done: bool,
}

impl CartPole {
    /// Create a new `CartPole` environment (initially in the reset state).
    #[must_use]
    pub fn new() -> Self {
        Self {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.0,
            theta_dot: 0.0,
            step_count: 0,
            done: false,
        }
    }

    fn make_observation(&self) -> Vec<f64> {
        vec![self.x, self.x_dot, self.theta, self.theta_dot]
    }
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for CartPole {
    type Observation = Vec<f64>;
    type Action = usize;

    fn reset(&mut self) -> Self::Observation {
        self.x = 0.0;
        self.x_dot = 0.0;
        self.theta = 0.05;
        self.theta_dot = 0.0;
        self.step_count = 0;
        self.done = false;
        self.make_observation()
    }

    fn step(&mut self, action: &Self::Action) -> StepResult<Self::Observation> {
        let force = if *action == 1 { FORCE_MAG } else { -FORCE_MAG };

        let cos_theta = self.theta.cos();
        let sin_theta = self.theta.sin();

        let temp =
            (force + POLE_MASS_LENGTH * self.theta_dot * self.theta_dot * sin_theta) / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
            / (LENGTH * (4.0 / 3.0 - MASS_POLE * cos_theta * cos_theta / TOTAL_MASS));
        let x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

        // Euler integration
        self.x += TAU * self.x_dot;
        self.x_dot += TAU * x_acc;
        self.theta += TAU * self.theta_dot;
        self.theta_dot += TAU * theta_acc;

        self.step_count += 1;

        let terminated = self.x < -X_THRESHOLD
            || self.x > X_THRESHOLD
            || self.theta < -THETA_THRESHOLD
            || self.theta > THETA_THRESHOLD;
        let truncated = self.step_count >= MAX_STEPS;
        self.done = terminated || truncated;

        StepResult {
            observation: self.make_observation(),
            reward: 1.0,
            done: terminated,
            truncated,
        }
    }

    fn observation_shape(&self) -> &[usize] {
        &OBS_SHAPE
    }

    fn action_count(&self) -> usize {
        2
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
    fn test_cartpole_reset() {
        let mut env = CartPole::new();
        let obs = env.reset();
        assert_eq!(obs.len(), 4);
        assert!(!env.is_done());
    }

    #[test]
    fn test_cartpole_step() {
        let mut env = CartPole::new();
        env.reset();
        let result = env.step(&0);
        assert_eq!(result.observation.len(), 4);
        assert_eq!(result.reward, 1.0);
    }

    #[test]
    fn test_cartpole_terminates() {
        let mut env = CartPole::new();
        env.reset();
        let mut done = false;
        for _ in 0..1000 {
            let result = env.step(&0); // always push left
            if result.done || result.truncated {
                done = true;
                break;
            }
        }
        assert!(done, "CartPole should terminate eventually");
    }

    #[test]
    fn test_cartpole_observation_shape() {
        let env = CartPole::new();
        assert_eq!(env.observation_shape(), &[4]);
    }

    #[test]
    fn test_cartpole_action_count() {
        let env = CartPole::new();
        assert_eq!(env.action_count(), 2);
    }
}
