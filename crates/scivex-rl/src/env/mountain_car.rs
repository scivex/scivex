//! Mountain car environment.

use super::{Environment, StepResult};

const MIN_POSITION: f64 = -1.2;
const MAX_POSITION: f64 = 0.6;
const MIN_VELOCITY: f64 = -0.07;
const MAX_VELOCITY: f64 = 0.07;
const GOAL_POSITION: f64 = 0.5;
const FORCE: f64 = 0.001;
const GRAVITY: f64 = 0.0025;
const MAX_STEPS: usize = 200;

const OBS_SHAPE: [usize; 1] = [2];

/// Mountain car environment.
///
/// The agent must drive an under-powered car up a hill. Observations are
/// `[position, velocity]` and there are 3 discrete actions (0 = push left,
/// 1 = no push, 2 = push right).
pub struct MountainCar {
    position: f64,
    velocity: f64,
    step_count: usize,
    done: bool,
}

impl MountainCar {
    /// Create a new `MountainCar` environment.
    #[must_use]
    pub fn new() -> Self {
        Self {
            position: -0.5,
            velocity: 0.0,
            step_count: 0,
            done: false,
        }
    }

    fn make_observation(&self) -> Vec<f64> {
        vec![self.position, self.velocity]
    }
}

impl Default for MountainCar {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for MountainCar {
    type Observation = Vec<f64>;
    type Action = usize;

    fn reset(&mut self) -> Self::Observation {
        self.position = -0.5;
        self.velocity = 0.0;
        self.step_count = 0;
        self.done = false;
        self.make_observation()
    }

    fn step(&mut self, action: &Self::Action) -> StepResult<Self::Observation> {
        let force = (*action as f64) - 1.0; // -1, 0, or 1

        self.velocity += force * FORCE - (3.0 * self.position).cos() * GRAVITY;
        self.velocity = self.velocity.clamp(MIN_VELOCITY, MAX_VELOCITY);

        self.position += self.velocity;
        self.position = self.position.clamp(MIN_POSITION, MAX_POSITION);

        // If at left boundary, reset velocity
        if self.position <= MIN_POSITION {
            self.velocity = 0.0;
        }

        self.step_count += 1;

        let goal_reached = self.position >= GOAL_POSITION;
        let truncated = self.step_count >= MAX_STEPS;
        self.done = goal_reached || truncated;

        StepResult {
            observation: self.make_observation(),
            reward: -1.0,
            done: goal_reached,
            truncated,
        }
    }

    fn observation_shape(&self) -> &[usize] {
        &OBS_SHAPE
    }

    fn action_count(&self) -> usize {
        3
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
    fn test_mountain_car_reset() {
        let mut env = MountainCar::new();
        let obs = env.reset();
        assert_eq!(obs.len(), 2);
        assert!(!env.is_done());
    }

    #[test]
    fn test_mountain_car_step() {
        let mut env = MountainCar::new();
        env.reset();
        let result = env.step(&2); // push right
        assert_eq!(result.observation.len(), 2);
        assert_eq!(result.reward, -1.0);
    }

    #[test]
    fn test_mountain_car_physics() {
        let mut env = MountainCar::new();
        env.reset();
        // Pushing right should increase velocity positively
        let before = env.velocity;
        env.step(&2);
        // Velocity should have changed
        assert_ne!(env.velocity, before);
    }

    #[test]
    fn test_mountain_car_truncates() {
        let mut env = MountainCar::new();
        env.reset();
        let mut truncated = false;
        for _ in 0..250 {
            let result = env.step(&1); // no push
            if result.truncated {
                truncated = true;
                break;
            }
        }
        assert!(truncated, "MountainCar should truncate at 200 steps");
    }

    #[test]
    fn test_mountain_car_observation_shape() {
        let env = MountainCar::new();
        assert_eq!(env.observation_shape(), &[2]);
    }
}
