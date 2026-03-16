//! Deep Q-Network (DQN) with experience replay and target network.

use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::functional::relu;
use scivex_nn::layer::{Layer, Linear};
use scivex_nn::loss::mse_loss;
use scivex_nn::optim::{Adam, Optimizer};

use crate::env::Environment;
use crate::error::{Result, RlError};
use crate::logger::EpisodeLogger;
use crate::replay::ReplayBuffer;

/// Configuration for the DQN agent.
pub struct DqnConfig {
    /// Learning rate for the optimizer.
    pub learning_rate: f64,
    /// Discount factor for future rewards.
    pub gamma: f64,
    /// Initial exploration rate.
    pub epsilon: f64,
    /// Multiplicative decay applied to epsilon after each episode.
    pub epsilon_decay: f64,
    /// Minimum exploration rate.
    pub min_epsilon: f64,
    /// Batch size for experience replay.
    pub batch_size: usize,
    /// How often (in steps) to copy weights to the target network.
    pub target_update_freq: usize,
    /// Capacity of the replay buffer.
    pub buffer_capacity: usize,
    /// Random seed.
    pub seed: u64,
}

impl DqnConfig {
    /// Create a new DQN configuration with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            learning_rate: 0.001,
            gamma: 0.99,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            min_epsilon: 0.01,
            batch_size: 32,
            target_update_freq: 100,
            buffer_capacity: 10_000,
            seed: 42,
        }
    }

    /// Set the learning rate.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the discount factor.
    #[must_use]
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the initial epsilon.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the epsilon decay.
    #[must_use]
    pub fn with_epsilon_decay(mut self, decay: f64) -> Self {
        self.epsilon_decay = decay;
        self
    }

    /// Set the minimum epsilon.
    #[must_use]
    pub fn with_min_epsilon(mut self, min_eps: f64) -> Self {
        self.min_epsilon = min_eps;
        self
    }

    /// Set the batch size.
    #[must_use]
    pub fn with_batch_size(mut self, bs: usize) -> Self {
        self.batch_size = bs;
        self
    }

    /// Set the target network update frequency (in steps).
    #[must_use]
    pub fn with_target_update_freq(mut self, freq: usize) -> Self {
        self.target_update_freq = freq;
        self
    }

    /// Set the replay buffer capacity.
    #[must_use]
    pub fn with_buffer_capacity(mut self, cap: usize) -> Self {
        self.buffer_capacity = cap;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple two-hidden-layer MLP Q-network.
struct QNetwork {
    fc1: Linear<f64>,
    fc2: Linear<f64>,
    fc3: Linear<f64>,
}

impl QNetwork {
    fn new(input_dim: usize, action_count: usize, rng: &mut Rng) -> Self {
        Self {
            fc1: Linear::new(input_dim, 64, true, rng),
            fc2: Linear::new(64, 64, true, rng),
            fc3: Linear::new(64, action_count, true, rng),
        }
    }

    fn forward(&self, x: &Variable<f64>) -> Result<Variable<f64>> {
        let h1 = self
            .fc1
            .forward(x)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        let h1 = relu(&h1);
        let h2 = self
            .fc2
            .forward(&h1)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        let h2 = relu(&h2);
        self.fc3
            .forward(&h2)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))
    }

    fn parameters(&self) -> Vec<Variable<f64>> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }

    /// Copy weights from another Q-network (for target network updates).
    fn copy_weights_from(&self, source: &Self) {
        let src_params = source.parameters();
        let dst_params = self.parameters();
        for (dst, src) in dst_params.iter().zip(src_params.iter()) {
            dst.set_data(src.data());
        }
    }
}

/// A DQN agent.
pub struct DqnAgent {
    q_network: QNetwork,
    target_network: QNetwork,
    optimizer: Adam<f64>,
    config: DqnConfig,
    rng: Rng,
    buffer: ReplayBuffer,
    step_count: usize,
    input_dim: usize,
    action_count: usize,
}

impl DqnAgent {
    /// Create a new DQN agent for the given observation and action dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(input_dim: usize, action_count: usize, config: DqnConfig) -> Result<Self> {
        if input_dim == 0 {
            return Err(RlError::InvalidParameter("input_dim must be > 0".into()));
        }
        if action_count == 0 {
            return Err(RlError::InvalidParameter("action_count must be > 0".into()));
        }

        let mut rng = Rng::new(config.seed);
        let q_network = QNetwork::new(input_dim, action_count, &mut rng);
        let target_network = QNetwork::new(input_dim, action_count, &mut rng);
        target_network.copy_weights_from(&q_network);

        let optimizer = Adam::new(q_network.parameters(), config.learning_rate);
        let buffer = ReplayBuffer::new(config.buffer_capacity);

        Ok(Self {
            q_network,
            target_network,
            optimizer,
            config,
            rng,
            buffer,
            step_count: 0,
            input_dim,
            action_count,
        })
    }

    /// Select an action using epsilon-greedy exploration.
    pub fn act(&mut self, observation: &[f64]) -> Result<usize> {
        if self.rng.next_f64() < self.config.epsilon {
            // Random action
            let action = (self.rng.next_f64() * self.action_count as f64) as usize;
            return Ok(action.min(self.action_count - 1));
        }

        // Greedy action from Q-network
        let input = Variable::new(
            Tensor::from_vec(observation.to_vec(), vec![1, self.input_dim])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            false,
        );
        let q_values = self.q_network.forward(&input)?;
        let q_slice = q_values.data();
        let q_data = q_slice.as_slice();

        let mut best_action = 0;
        let mut best_value = q_data[0];
        for (i, &v) in q_data.iter().enumerate().skip(1) {
            if v > best_value {
                best_value = v;
                best_action = i;
            }
        }
        Ok(best_action)
    }

    /// Perform a single training step on a batch of experience.
    ///
    /// Returns the loss value.
    #[allow(clippy::too_many_lines)]
    pub fn train_step(&mut self) -> Result<f64> {
        if self.buffer.len() < self.config.batch_size {
            return Ok(0.0);
        }

        let batch = self.buffer.sample(self.config.batch_size, &mut self.rng)?;
        let bs = self.config.batch_size;

        // Build state tensor [batch, input_dim]
        let mut state_data = Vec::with_capacity(bs * self.input_dim);
        for s in &batch.states {
            state_data.extend_from_slice(s);
        }
        let state_tensor = Variable::new(
            Tensor::from_vec(state_data, vec![bs, self.input_dim])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            false,
        );

        // Build next-state tensor [batch, input_dim]
        let mut next_state_data = Vec::with_capacity(bs * self.input_dim);
        for s in &batch.next_states {
            next_state_data.extend_from_slice(s);
        }
        let next_state_tensor = Variable::new(
            Tensor::from_vec(next_state_data, vec![bs, self.input_dim])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            false,
        );

        // Get current Q-values: q_network(state)[action]
        let q_all = self.q_network.forward(&state_tensor)?;
        let q_all_data = q_all.data();
        let q_all_slice = q_all_data.as_slice();

        // Extract Q-values for the actions taken
        let mut q_values_data = Vec::with_capacity(bs);
        for i in 0..bs {
            q_values_data.push(q_all_slice[i * self.action_count + batch.actions[i]]);
        }

        // Get target Q-values from target network
        let target_q_all = self.target_network.forward(&next_state_tensor)?;
        let target_q_data = target_q_all.data();
        let target_q_slice = target_q_data.as_slice();

        // target = reward + gamma * max(target_q) * (1 - done)
        let mut target_data = Vec::with_capacity(bs);
        for i in 0..bs {
            let row_start = i * self.action_count;
            let mut max_q = target_q_slice[row_start];
            for j in 1..self.action_count {
                let v = target_q_slice[row_start + j];
                if v > max_q {
                    max_q = v;
                }
            }
            let done_mask = if batch.dones[i] { 0.0 } else { 1.0 };
            target_data.push(batch.rewards[i] + self.config.gamma * max_q * done_mask);
        }

        // Compute MSE loss between predicted and target Q-values
        let predicted = Variable::new(
            Tensor::from_vec(q_values_data, vec![bs])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            true,
        );
        let target = Variable::new(
            Tensor::from_vec(target_data, vec![bs])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            false,
        );

        let loss =
            mse_loss(&predicted, &target).map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        let loss_val = loss.data().as_slice()[0];

        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        // Update target network periodically
        self.step_count += 1;
        #[allow(clippy::manual_is_multiple_of)]
        if self.step_count % self.config.target_update_freq == 0 {
            self.target_network.copy_weights_from(&self.q_network);
        }

        Ok(loss_val)
    }

    /// Train the agent on the given environment for the specified number of
    /// episodes.
    pub fn train<E>(&mut self, env: &mut E, episodes: usize) -> Result<EpisodeLogger>
    where
        E: Environment<Observation = Vec<f64>, Action = usize>,
    {
        let mut logger = EpisodeLogger::new();

        for _ in 0..episodes {
            let mut obs = env.reset();
            loop {
                let action = self.act(&obs)?;
                let result = env.step(&action);
                let done = result.done || result.truncated;

                self.buffer.push(
                    obs.clone(),
                    action,
                    result.reward,
                    result.observation.clone(),
                    done,
                );

                logger.log_step(result.reward);
                let _loss = self.train_step()?;

                obs = result.observation;
                if done {
                    break;
                }
            }
            logger.end_episode();

            // Decay epsilon
            self.config.epsilon =
                (self.config.epsilon * self.config.epsilon_decay).max(self.config.min_epsilon);
        }

        Ok(logger)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::CartPole;

    #[test]
    fn test_dqn_construction() {
        let config = DqnConfig::new();
        let agent = DqnAgent::new(4, 2, config);
        assert!(agent.is_ok());
    }

    #[test]
    fn test_dqn_act() {
        let config = DqnConfig::new().with_epsilon(0.0); // greedy
        let mut agent = DqnAgent::new(4, 2, config).unwrap();
        let action = agent.act(&[0.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(action < 2);
    }

    #[test]
    fn test_dqn_train_cartpole() {
        let config = DqnConfig::new()
            .with_batch_size(8)
            .with_buffer_capacity(200)
            .with_epsilon(0.5)
            .with_target_update_freq(50);
        let mut agent = DqnAgent::new(4, 2, config).unwrap();
        let mut env = CartPole::new();
        let logger = agent.train(&mut env, 3).unwrap();
        assert_eq!(logger.total_episodes(), 3);
    }

    #[test]
    fn test_dqn_invalid_params() {
        assert!(DqnAgent::new(0, 2, DqnConfig::new()).is_err());
        assert!(DqnAgent::new(4, 0, DqnConfig::new()).is_err());
    }
}
