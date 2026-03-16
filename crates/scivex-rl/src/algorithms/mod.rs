//! Reinforcement learning algorithms.

pub mod a2c;
pub mod dqn;
pub mod ppo;

pub use a2c::{A2cAgent, A2cConfig};
pub use dqn::{DqnAgent, DqnConfig};
pub use ppo::{PpoAgent, PpoConfig};
