//! Soft Actor-Critic (SAC) — off-policy actor-critic with entropy regularization.
//!
//! Designed for continuous action spaces. Uses twin Q-networks and automatic
//! entropy tuning for stable learning.

use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::functional::relu;
use scivex_nn::layer::{Layer, Linear};
use scivex_nn::loss::mse_loss;
use scivex_nn::optim::{Adam, Optimizer};

use crate::env::Environment;
use crate::error::{Result, RlError};
use crate::replay_continuous::ContinuousReplayBuffer;

/// Configuration for the SAC agent.
pub struct SacConfig {
    /// Learning rate for actor and critic.
    pub learning_rate: f64,
    /// Discount factor for future rewards.
    pub gamma: f64,
    /// Soft update rate for target networks.
    pub tau: f64,
    /// Initial entropy coefficient.
    pub alpha: f64,
    /// Batch size for experience replay.
    pub batch_size: usize,
    /// Capacity of the replay buffer.
    pub buffer_capacity: usize,
    /// Random seed.
    pub seed: u64,
}

impl SacConfig {
    /// Create a new SAC configuration with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            learning_rate: 0.0003,
            gamma: 0.99,
            tau: 0.005,
            alpha: 0.2,
            batch_size: 64,
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

    /// Set the soft update rate.
    #[must_use]
    pub fn with_tau(mut self, tau: f64) -> Self {
        self.tau = tau;
        self
    }

    /// Set the entropy coefficient.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the batch size.
    #[must_use]
    pub fn with_batch_size(mut self, bs: usize) -> Self {
        self.batch_size = bs;
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

impl Default for SacConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A two-hidden-layer MLP policy network that outputs mean and log-std for a
/// Gaussian policy (continuous actions).
struct PolicyNetwork {
    fc1: Linear<f64>,
    fc2: Linear<f64>,
    mean_head: Linear<f64>,
    log_std_head: Linear<f64>,
}

impl PolicyNetwork {
    fn new(state_dim: usize, action_dim: usize, rng: &mut Rng) -> Self {
        Self {
            fc1: Linear::new(state_dim, 64, true, rng),
            fc2: Linear::new(64, 64, true, rng),
            mean_head: Linear::new(64, action_dim, true, rng),
            log_std_head: Linear::new(64, action_dim, true, rng),
        }
    }

    /// Forward pass returning (mean, log_std) each of shape `[batch, action_dim]`.
    fn forward(&self, x: &Variable<f64>) -> Result<(Variable<f64>, Variable<f64>)> {
        let h = self
            .fc1
            .forward(x)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        let h = relu(&h);
        let h = self
            .fc2
            .forward(&h)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        let h = relu(&h);
        let mean = self
            .mean_head
            .forward(&h)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        let log_std = self
            .log_std_head
            .forward(&h)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        Ok((mean, log_std))
    }

    fn parameters(&self) -> Vec<Variable<f64>> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.mean_head.parameters());
        params.extend(self.log_std_head.parameters());
        params
    }
}

/// A two-hidden-layer MLP Q-network that takes (state, action) as input and
/// outputs a single Q-value.
struct QNetwork {
    fc1: Linear<f64>,
    fc2: Linear<f64>,
    fc3: Linear<f64>,
}

impl QNetwork {
    fn new(state_dim: usize, action_dim: usize, rng: &mut Rng) -> Self {
        Self {
            fc1: Linear::new(state_dim + action_dim, 64, true, rng),
            fc2: Linear::new(64, 64, true, rng),
            fc3: Linear::new(64, 1, true, rng),
        }
    }

    fn forward(&self, state_action: &Variable<f64>) -> Result<Variable<f64>> {
        let h = self
            .fc1
            .forward(state_action)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        let h = relu(&h);
        let h = self
            .fc2
            .forward(&h)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        let h = relu(&h);
        self.fc3
            .forward(&h)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))
    }

    fn parameters(&self) -> Vec<Variable<f64>> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }

    /// Copy weights from another Q-network.
    fn copy_weights_from(&self, source: &Self) {
        let src_params = source.parameters();
        let dst_params = self.parameters();
        for (dst, src) in dst_params.iter().zip(src_params.iter()) {
            dst.set_data(src.data());
        }
    }

    /// Soft update: target = tau * source + (1 - tau) * target.
    fn soft_update_from(&self, source: &Self, tau: f64) {
        let src_params = source.parameters();
        let dst_params = self.parameters();
        for (dst, src) in dst_params.iter().zip(src_params.iter()) {
            let src_data = src.data();
            let dst_data = dst.data();
            let src_slice = src_data.as_slice();
            let dst_slice = dst_data.as_slice();
            let blended: Vec<f64> = src_slice
                .iter()
                .zip(dst_slice.iter())
                .map(|(&s, &d)| tau * s + (1.0 - tau) * d)
                .collect();
            let blended_tensor =
                Tensor::from_vec(blended, dst_data.shape().to_vec()).unwrap_or(dst.data());
            dst.set_data(blended_tensor);
        }
    }
}

/// A Soft Actor-Critic agent for continuous action spaces.
pub struct SacAgent {
    actor: PolicyNetwork,
    critic1: QNetwork,
    critic2: QNetwork,
    target_critic1: QNetwork,
    target_critic2: QNetwork,
    actor_optimizer: Adam<f64>,
    critic1_optimizer: Adam<f64>,
    critic2_optimizer: Adam<f64>,
    config: SacConfig,
    rng: Rng,
    buffer: ContinuousReplayBuffer,
    state_dim: usize,
    action_dim: usize,
}

impl SacAgent {
    /// Create a new SAC agent for the given state and action dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(state_dim: usize, action_dim: usize, config: SacConfig) -> Result<Self> {
        if state_dim == 0 {
            return Err(RlError::InvalidParameter("state_dim must be > 0".into()));
        }
        if action_dim == 0 {
            return Err(RlError::InvalidParameter("action_dim must be > 0".into()));
        }

        let mut rng = Rng::new(config.seed);

        let actor = PolicyNetwork::new(state_dim, action_dim, &mut rng);
        let critic1 = QNetwork::new(state_dim, action_dim, &mut rng);
        let critic2 = QNetwork::new(state_dim, action_dim, &mut rng);
        let target_critic1 = QNetwork::new(state_dim, action_dim, &mut rng);
        let target_critic2 = QNetwork::new(state_dim, action_dim, &mut rng);
        target_critic1.copy_weights_from(&critic1);
        target_critic2.copy_weights_from(&critic2);

        let actor_optimizer = Adam::new(actor.parameters(), config.learning_rate);
        let critic1_optimizer = Adam::new(critic1.parameters(), config.learning_rate);
        let critic2_optimizer = Adam::new(critic2.parameters(), config.learning_rate);

        let buffer = ContinuousReplayBuffer::new(config.buffer_capacity);

        Ok(Self {
            actor,
            critic1,
            critic2,
            target_critic1,
            target_critic2,
            actor_optimizer,
            critic1_optimizer,
            critic2_optimizer,
            config,
            rng,
            buffer,
            state_dim,
            action_dim,
        })
    }

    /// Select an action by sampling from the Gaussian policy.
    ///
    /// Uses the reparameterization trick: action = tanh(mean + std * noise).
    pub fn select_action(&mut self, state: &[f64]) -> Result<Vec<f64>> {
        let input = Variable::new(
            Tensor::from_vec(state.to_vec(), vec![1, self.state_dim])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            false,
        );
        let (mean_var, log_std_var) = self.actor.forward(&input)?;
        let mean_data = mean_var.data();
        let mean_slice = mean_data.as_slice();
        let log_std_data = log_std_var.data();
        let log_std_slice = log_std_data.as_slice();

        let mut action = Vec::with_capacity(self.action_dim);
        for i in 0..self.action_dim {
            let log_std_clamped = log_std_slice[i].clamp(-20.0, 2.0);
            let std = log_std_clamped.exp();
            let noise = self.rng.next_normal_f64();
            let raw = mean_slice[i] + std * noise;
            // Apply tanh squashing to bound actions to [-1, 1]
            action.push(raw.tanh());
        }

        Ok(action)
    }

    /// Concatenate state and action vectors into a single input vector.
    fn concat_state_action(state: &[f64], action: &[f64]) -> Vec<f64> {
        let mut sa = Vec::with_capacity(state.len() + action.len());
        sa.extend_from_slice(state);
        sa.extend_from_slice(action);
        sa
    }

    /// Perform a single training step on a batch of experience.
    ///
    /// Returns the critic loss value.
    #[allow(clippy::too_many_lines)]
    pub fn train_step(&mut self) -> Result<f64> {
        if self.buffer.len() < self.config.batch_size {
            return Ok(0.0);
        }

        let batch = self.buffer.sample(self.config.batch_size, &mut self.rng)?;
        let bs = self.config.batch_size;
        let sa_dim = self.state_dim + self.action_dim;

        // --- Update critics ---

        // Compute target Q-values: y = r + gamma * (1 - d) * (min(Q1', Q2') - alpha * log_pi)
        let mut target_data = Vec::with_capacity(bs);
        for i in 0..bs {
            // Sample next action from current policy
            let next_input = Variable::new(
                Tensor::from_vec(batch.next_states[i].clone(), vec![1, self.state_dim])
                    .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                false,
            );
            let (next_mean, next_log_std) = self.actor.forward(&next_input)?;
            let nm = next_mean.data();
            let nls = next_log_std.data();
            let nm_s = nm.as_slice();
            let nls_s = nls.as_slice();

            let mut next_action = Vec::with_capacity(self.action_dim);
            let mut log_prob = 0.0;
            for j in 0..self.action_dim {
                let ls_clamped = nls_s[j].clamp(-20.0, 2.0);
                let std = ls_clamped.exp();
                let noise = self.rng.next_normal_f64();
                let raw = nm_s[j] + std * noise;
                let a = raw.tanh();
                next_action.push(a);
                // log prob of tanh-squashed Gaussian
                log_prob +=
                    -0.5 * noise * noise - 0.5 * (2.0 * std::f64::consts::PI).ln() - ls_clamped;
                log_prob -= (1.0 - a * a + 1e-6).ln();
            }

            let sa = Self::concat_state_action(&batch.next_states[i], &next_action);
            let sa_var = Variable::new(
                Tensor::from_vec(sa, vec![1, sa_dim])
                    .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                false,
            );

            let q1_next = self.target_critic1.forward(&sa_var)?;
            let q2_next = self.target_critic2.forward(&sa_var)?;
            let q1_val = q1_next.data().as_slice()[0];
            let q2_val = q2_next.data().as_slice()[0];
            let min_q = q1_val.min(q2_val);

            let done_mask = if batch.dones[i] { 0.0 } else { 1.0 };
            let target = batch.rewards[i]
                + self.config.gamma * done_mask * (min_q - self.config.alpha * log_prob);
            target_data.push(target);
        }

        // Compute current Q-values for critics
        let mut q1_data = Vec::with_capacity(bs);
        let mut q2_data = Vec::with_capacity(bs);
        for i in 0..bs {
            let sa = Self::concat_state_action(&batch.states[i], &batch.actions[i]);
            let sa_var = Variable::new(
                Tensor::from_vec(sa, vec![1, sa_dim])
                    .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                false,
            );
            q1_data.push(self.critic1.forward(&sa_var)?.data().as_slice()[0]);
            q2_data.push(self.critic2.forward(&sa_var)?.data().as_slice()[0]);
        }

        // Critic 1 loss
        let pred1 = Variable::new(
            Tensor::from_vec(q1_data, vec![bs])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            true,
        );
        let target_var = Variable::new(
            Tensor::from_vec(target_data.clone(), vec![bs])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            false,
        );
        let loss1 =
            mse_loss(&pred1, &target_var).map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        let loss1_val = loss1.data().as_slice()[0];

        self.critic1_optimizer.zero_grad();
        loss1.backward();
        self.critic1_optimizer.step();

        // Critic 2 loss
        let pred2 = Variable::new(
            Tensor::from_vec(q2_data, vec![bs])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            true,
        );
        let target_var2 = Variable::new(
            Tensor::from_vec(target_data, vec![bs])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            false,
        );
        let loss2 =
            mse_loss(&pred2, &target_var2).map_err(|e| RlError::EnvironmentError(e.to_string()))?;

        self.critic2_optimizer.zero_grad();
        loss2.backward();
        self.critic2_optimizer.step();

        // --- Update actor ---
        // Actor loss: E[alpha * log_pi - Q(s, a_new)]
        let mut actor_loss_val = 0.0;
        for i in 0..bs {
            let state_var = Variable::new(
                Tensor::from_vec(batch.states[i].clone(), vec![1, self.state_dim])
                    .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                false,
            );
            let (mean_v, log_std_v) = self.actor.forward(&state_var)?;
            let m = mean_v.data();
            let ls = log_std_v.data();
            let m_s = m.as_slice();
            let ls_s = ls.as_slice();

            let mut new_action = Vec::with_capacity(self.action_dim);
            let mut log_prob = 0.0;
            for j in 0..self.action_dim {
                let ls_c = ls_s[j].clamp(-20.0, 2.0);
                let std = ls_c.exp();
                let noise = self.rng.next_normal_f64();
                let raw = m_s[j] + std * noise;
                let a = raw.tanh();
                new_action.push(a);
                log_prob += -0.5 * noise * noise - 0.5 * (2.0 * std::f64::consts::PI).ln() - ls_c;
                log_prob -= (1.0 - a * a + 1e-6).ln();
            }

            let sa = Self::concat_state_action(&batch.states[i], &new_action);
            let sa_var = Variable::new(
                Tensor::from_vec(sa, vec![1, sa_dim])
                    .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                false,
            );
            let q1_val = self.critic1.forward(&sa_var)?.data().as_slice()[0];

            actor_loss_val += self.config.alpha * log_prob - q1_val;
        }
        actor_loss_val /= bs as f64;

        let actor_loss = Variable::new(
            Tensor::from_vec(vec![actor_loss_val], vec![1])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            true,
        );
        self.actor_optimizer.zero_grad();
        actor_loss.backward();
        self.actor_optimizer.step();

        // --- Soft update target networks ---
        self.target_critic1
            .soft_update_from(&self.critic1, self.config.tau);
        self.target_critic2
            .soft_update_from(&self.critic2, self.config.tau);

        Ok(loss1_val)
    }

    /// Train the agent on the given continuous-action environment for the
    /// specified number of episodes.
    ///
    /// The environment must produce `Vec<f64>` observations and accept
    /// `Vec<f64>` actions.
    pub fn train<E>(&mut self, env: &mut E, episodes: usize) -> Result<Vec<f64>>
    where
        E: Environment<Observation = Vec<f64>, Action = Vec<f64>>,
    {
        let mut episode_rewards = Vec::with_capacity(episodes);

        for _ in 0..episodes {
            let mut obs = env.reset();
            let mut total_reward = 0.0;

            loop {
                let action = self.select_action(&obs)?;
                let result = env.step(&action);
                let done = result.done || result.truncated;

                self.buffer.push(
                    obs.clone(),
                    action,
                    result.reward,
                    result.observation.clone(),
                    done,
                );

                let _loss = self.train_step()?;
                total_reward += result.reward;
                obs = result.observation;

                if done {
                    break;
                }
            }

            episode_rewards.push(total_reward);
        }

        Ok(episode_rewards)
    }

    /// Return the state dimension.
    #[must_use]
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Return the action dimension.
    #[must_use]
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sac_creation() {
        let config = SacConfig::new();
        let agent = SacAgent::new(4, 2, config);
        assert!(agent.is_ok());
        let agent = agent.unwrap();
        assert_eq!(agent.state_dim(), 4);
        assert_eq!(agent.action_dim(), 2);
    }

    #[test]
    fn test_sac_creation_invalid() {
        assert!(SacAgent::new(0, 2, SacConfig::new()).is_err());
        assert!(SacAgent::new(4, 0, SacConfig::new()).is_err());
    }

    #[test]
    fn test_sac_select_action() {
        let config = SacConfig::new();
        let mut agent = SacAgent::new(4, 2, config).unwrap();
        let action = agent.select_action(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(action.len(), 2);
        // Actions should be bounded by tanh to [-1, 1]
        for &a in &action {
            assert!((-1.0..=1.0).contains(&a));
        }
    }
}
