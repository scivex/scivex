//! Reinforcement learning environments.

pub mod cartpole;
pub mod grid_world;
pub mod mountain_car;

pub use cartpole::CartPole;
pub use grid_world::GridWorld;
pub use mountain_car::MountainCar;

/// The result of taking a step in an environment.
pub struct StepResult<O> {
    /// The observation after the step.
    pub observation: O,
    /// The reward received for this step.
    pub reward: f64,
    /// Whether the episode has ended (terminal state).
    pub done: bool,
    /// Whether the episode was truncated (e.g. time limit).
    pub truncated: bool,
}

/// A reinforcement learning environment.
pub trait Environment {
    /// The type of observations produced by this environment.
    type Observation;
    /// The type of actions accepted by this environment.
    type Action;

    /// Reset the environment and return the initial observation.
    fn reset(&mut self) -> Self::Observation;

    /// Take a step with the given action and return the result.
    fn step(&mut self, action: &Self::Action) -> StepResult<Self::Observation>;

    /// Return the shape of an observation (as a slice of dimension sizes).
    fn observation_shape(&self) -> &[usize];

    /// Return the number of discrete actions available.
    fn action_count(&self) -> usize;

    /// Return whether the current episode is done.
    fn is_done(&self) -> bool;
}
