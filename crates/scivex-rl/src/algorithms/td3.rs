//! Twin Delayed Deep Deterministic Policy Gradient (TD3).
//!
//! Deterministic policy gradient with twin critics, delayed policy updates,
//! and target policy smoothing for continuous action spaces.

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

/// Configuration for the TD3 agent.
pub struct Td3Config {
    /// Learning rate for actor and critic.
    pub learning_rate: f64,
    /// Discount factor for future rewards.
    pub gamma: f64,
    /// Soft update rate for target networks.
    pub tau: f64,
    /// Standard deviation of noise added to target policy actions.
    pub policy_noise: f64,
    /// Clipping range for target policy noise.
    pub noise_clip: f64,
    /// How often (in train steps) to update the actor and target networks.
    pub policy_delay: usize,
    /// Batch size for experience replay.
    pub batch_size: usize,
    /// Capacity of the replay buffer.
    pub buffer_capacity: usize,
    /// Random seed.
    pub seed: u64,
}

impl Td3Config {
    /// Create a new TD3 configuration with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            learning_rate: 0.0003,
            gamma: 0.99,
            tau: 0.005,
            policy_noise: 0.2,
            noise_clip: 0.5,
            policy_delay: 2,
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

    /// Set the target policy noise standard deviation.
    #[must_use]
    pub fn with_policy_noise(mut self, noise: f64) -> Self {
        self.policy_noise = noise;
        self
    }

    /// Set the target policy noise clip range.
    #[must_use]
    pub fn with_noise_clip(mut self, clip: f64) -> Self {
        self.noise_clip = clip;
        self
    }

    /// Set the policy update delay (in train steps).
    #[must_use]
    pub fn with_policy_delay(mut self, delay: usize) -> Self {
        self.policy_delay = delay;
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

impl Default for Td3Config {
    fn default() -> Self {
        Self::new()
    }
}

/// A two-hidden-layer deterministic actor network.
struct ActorNetwork {
    fc1: Linear<f64>,
    fc2: Linear<f64>,
    fc3: Linear<f64>,
}

impl ActorNetwork {
    fn new(state_dim: usize, action_dim: usize, rng: &mut Rng) -> Self {
        Self {
            fc1: Linear::new(state_dim, 64, true, rng),
            fc2: Linear::new(64, 64, true, rng),
            fc3: Linear::new(64, action_dim, true, rng),
        }
    }

    /// Forward pass returning raw action values of shape `[batch, action_dim]`.
    fn forward(&self, x: &Variable<f64>) -> Result<Variable<f64>> {
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

    /// Copy weights from another actor network.
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

/// A two-hidden-layer Q-network that takes (state, action) as input.
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

    fn copy_weights_from(&self, source: &Self) {
        let src_params = source.parameters();
        let dst_params = self.parameters();
        for (dst, src) in dst_params.iter().zip(src_params.iter()) {
            dst.set_data(src.data());
        }
    }

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

/// A TD3 agent for continuous action spaces.
pub struct Td3Agent {
    actor: ActorNetwork,
    target_actor: ActorNetwork,
    critic1: QNetwork,
    critic2: QNetwork,
    target_critic1: QNetwork,
    target_critic2: QNetwork,
    actor_optimizer: Adam<f64>,
    critic1_optimizer: Adam<f64>,
    critic2_optimizer: Adam<f64>,
    config: Td3Config,
    rng: Rng,
    buffer: ContinuousReplayBuffer,
    update_count: usize,
    state_dim: usize,
    action_dim: usize,
}

impl Td3Agent {
    /// Create a new TD3 agent for the given state and action dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(state_dim: usize, action_dim: usize, config: Td3Config) -> Result<Self> {
        if state_dim == 0 {
            return Err(RlError::InvalidParameter("state_dim must be > 0".into()));
        }
        if action_dim == 0 {
            return Err(RlError::InvalidParameter("action_dim must be > 0".into()));
        }

        let mut rng = Rng::new(config.seed);

        let actor = ActorNetwork::new(state_dim, action_dim, &mut rng);
        let target_actor = ActorNetwork::new(state_dim, action_dim, &mut rng);
        target_actor.copy_weights_from(&actor);

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
            target_actor,
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
            update_count: 0,
            state_dim,
            action_dim,
        })
    }

    /// Select an action using the deterministic policy plus optional
    /// exploration noise.
    ///
    /// The `exploration_noise` parameter controls the standard deviation of
    /// Gaussian noise added to the deterministic action. Set to `0.0` for
    /// evaluation.
    pub fn select_action(&mut self, state: &[f64], exploration_noise: f64) -> Result<Vec<f64>> {
        let input = Variable::new(
            Tensor::from_vec(state.to_vec(), vec![1, self.state_dim])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            false,
        );
        let output = self.actor.forward(&input)?;
        let out_data = output.data();
        let out_slice = out_data.as_slice();

        let mut action = Vec::with_capacity(self.action_dim);
        for val in &out_slice[..self.action_dim] {
            let noise = if exploration_noise > 0.0 {
                self.rng.next_normal_f64() * exploration_noise
            } else {
                0.0
            };
            // Clamp to [-1, 1] after adding noise
            action.push((val.tanh() + noise).clamp(-1.0, 1.0));
        }

        Ok(action)
    }

    /// Concatenate state and action into a single vector.
    fn concat_state_action(state: &[f64], action: &[f64]) -> Vec<f64> {
        let mut sa = Vec::with_capacity(state.len() + action.len());
        sa.extend_from_slice(state);
        sa.extend_from_slice(action);
        sa
    }

    /// Perform a single training step on a batch of experience.
    ///
    /// Updates critics every step, but only updates the actor and target
    /// networks every `policy_delay` steps.
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

        // --- Compute target Q-values ---
        let mut target_data = Vec::with_capacity(bs);
        for i in 0..bs {
            // Target action with clipped noise (target policy smoothing)
            let next_input = Variable::new(
                Tensor::from_vec(batch.next_states[i].clone(), vec![1, self.state_dim])
                    .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                false,
            );
            let target_out = self.target_actor.forward(&next_input)?;
            let t_data = target_out.data();
            let t_slice = t_data.as_slice();

            let mut next_action = Vec::with_capacity(self.action_dim);
            for val in &t_slice[..self.action_dim] {
                let noise = (self.rng.next_normal_f64() * self.config.policy_noise)
                    .clamp(-self.config.noise_clip, self.config.noise_clip);
                let a = (val.tanh() + noise).clamp(-1.0, 1.0);
                next_action.push(a);
            }

            let sa = Self::concat_state_action(&batch.next_states[i], &next_action);
            let sa_var = Variable::new(
                Tensor::from_vec(sa, vec![1, sa_dim])
                    .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                false,
            );

            let q1_val = self.target_critic1.forward(&sa_var)?.data().as_slice()[0];
            let q2_val = self.target_critic2.forward(&sa_var)?.data().as_slice()[0];
            let min_q = q1_val.min(q2_val);

            let done_mask = if batch.dones[i] { 0.0 } else { 1.0 };
            target_data.push(batch.rewards[i] + self.config.gamma * done_mask * min_q);
        }

        // --- Update critics ---
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

        // --- Delayed actor update ---
        self.update_count += 1;
        #[allow(clippy::manual_is_multiple_of)]
        if self.update_count % self.config.policy_delay == 0 {
            // Actor loss: -mean(Q1(s, actor(s)))
            let mut actor_loss_val = 0.0;
            for i in 0..bs {
                let state_var = Variable::new(
                    Tensor::from_vec(batch.states[i].clone(), vec![1, self.state_dim])
                        .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                    false,
                );
                let actor_out = self.actor.forward(&state_var)?;
                let a_data = actor_out.data();
                let a_slice = a_data.as_slice();

                let mut new_action = Vec::with_capacity(self.action_dim);
                for val in &a_slice[..self.action_dim] {
                    new_action.push(val.tanh());
                }

                let sa = Self::concat_state_action(&batch.states[i], &new_action);
                let sa_var = Variable::new(
                    Tensor::from_vec(sa, vec![1, sa_dim])
                        .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                    false,
                );
                let q1_val = self.critic1.forward(&sa_var)?.data().as_slice()[0];
                actor_loss_val -= q1_val;
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

            // Soft update targets
            self.target_actor
                .soft_update_from(&self.actor, self.config.tau);
            self.target_critic1
                .soft_update_from(&self.critic1, self.config.tau);
            self.target_critic2
                .soft_update_from(&self.critic2, self.config.tau);
        }

        Ok(loss1_val)
    }

    /// Train the agent on the given continuous-action environment for the
    /// specified number of episodes.
    pub fn train<E>(&mut self, env: &mut E, episodes: usize) -> Result<Vec<f64>>
    where
        E: Environment<Observation = Vec<f64>, Action = Vec<f64>>,
    {
        let mut episode_rewards = Vec::with_capacity(episodes);

        for _ in 0..episodes {
            let mut obs = env.reset();
            let mut total_reward = 0.0;

            loop {
                let action = self.select_action(&obs, 0.1)?;
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
    fn test_td3_creation() {
        let config = Td3Config::new();
        let agent = Td3Agent::new(4, 2, config);
        assert!(agent.is_ok());
        let agent = agent.unwrap();
        assert_eq!(agent.state_dim(), 4);
        assert_eq!(agent.action_dim(), 2);
    }

    #[test]
    fn test_td3_creation_invalid() {
        assert!(Td3Agent::new(0, 2, Td3Config::new()).is_err());
        assert!(Td3Agent::new(4, 0, Td3Config::new()).is_err());
    }

    #[test]
    fn test_td3_select_action() {
        let config = Td3Config::new();
        let mut agent = Td3Agent::new(4, 2, config).unwrap();

        // Deterministic (no noise)
        let action = agent.select_action(&[0.1, 0.2, 0.3, 0.4], 0.0).unwrap();
        assert_eq!(action.len(), 2);
        for &a in &action {
            assert!((-1.0..=1.0).contains(&a));
        }

        // With exploration noise
        let action_noisy = agent.select_action(&[0.1, 0.2, 0.3, 0.4], 0.5).unwrap();
        assert_eq!(action_noisy.len(), 2);
        for &a in &action_noisy {
            assert!((-1.0..=1.0).contains(&a));
        }
    }
}
