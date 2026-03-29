//! # scivex-rl
//!
//! Reinforcement learning: environments, algorithms (DQN, PPO, A2C),
//! experience replay, and episode logging for the Scivex ecosystem.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`mod@env`] | `Environment` trait, `CartPole`, `MountainCar`, `GridWorld` |
//! | [`algorithms`] | `DqnAgent`, `PpoAgent`, `A2cAgent` |
//! | [`replay`] | `ReplayBuffer` for off-policy methods |
//! | [`logger`] | `EpisodeLogger` for tracking training progress |
//! | [`error`] | `RlError` and `Result` type alias |

/// Reinforcement learning algorithms: DQN, PPO, A2C.
pub mod algorithms;
/// Reinforcement learning environments.
pub mod env;
/// Error types for the RL crate.
pub mod error;
/// Hindsight Experience Replay for goal-conditioned RL.
pub mod her;
/// Episode logging utilities.
pub mod logger;
/// Experience replay buffer.
pub mod replay;
/// Experience replay buffer for continuous-action algorithms.
pub mod replay_continuous;

pub use algorithms::{
    A2cAgent, A2cConfig, DqnAgent, DqnConfig, PpoAgent, PpoConfig, SacAgent, SacConfig, Td3Agent,
    Td3Config,
};
pub use env::{
    CartPole, CooperativeNavigation, Environment, GridWorld, MountainCar, MultiAgentEnv, StepResult,
};
pub use error::{Result, RlError};
pub use her::{GoalTransition, HerReplayBuffer, HerStrategy};
pub use logger::EpisodeLogger;
pub use replay::ReplayBuffer;

/// Convenience re-exports.
pub mod prelude {
    pub use crate::algorithms::{
        A2cAgent, A2cConfig, DqnAgent, DqnConfig, PpoAgent, PpoConfig, SacAgent, SacConfig,
        Td3Agent, Td3Config,
    };
    pub use crate::env::{
        CartPole, CooperativeNavigation, Environment, GridWorld, MountainCar, MultiAgentEnv,
        StepResult,
    };
    pub use crate::error::{Result, RlError};
    pub use crate::her::{GoalTransition, HerReplayBuffer, HerStrategy};
    pub use crate::logger::EpisodeLogger;
    pub use crate::replay::ReplayBuffer;
}
