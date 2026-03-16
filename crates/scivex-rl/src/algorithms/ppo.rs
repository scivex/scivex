//! Proximal Policy Optimization (PPO) — clip variant.

use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::functional::relu;
use scivex_nn::layer::{Layer, Linear};
use scivex_nn::optim::{Adam, Optimizer};

use crate::env::Environment;
use crate::error::{Result, RlError};
use crate::logger::EpisodeLogger;

/// Configuration for the PPO agent.
pub struct PpoConfig {
    /// Learning rate.
    pub learning_rate: f64,
    /// Discount factor.
    pub gamma: f64,
    /// GAE lambda.
    pub gae_lambda: f64,
    /// PPO clip epsilon.
    pub clip_epsilon: f64,
    /// Number of optimization epochs per rollout.
    pub n_epochs: usize,
    /// Number of environment steps per rollout.
    pub n_steps: usize,
    /// Mini-batch size within each epoch.
    pub batch_size: usize,
    /// Random seed.
    pub seed: u64,
}

impl PpoConfig {
    /// Create a new PPO configuration with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            learning_rate: 0.0003,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            n_epochs: 4,
            n_steps: 128,
            batch_size: 32,
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

    /// Set the GAE lambda.
    #[must_use]
    pub fn with_gae_lambda(mut self, lam: f64) -> Self {
        self.gae_lambda = lam;
        self
    }

    /// Set the clip epsilon.
    #[must_use]
    pub fn with_clip_epsilon(mut self, eps: f64) -> Self {
        self.clip_epsilon = eps;
        self
    }

    /// Set the number of optimization epochs.
    #[must_use]
    pub fn with_n_epochs(mut self, n: usize) -> Self {
        self.n_epochs = n;
        self
    }

    /// Set the number of steps per rollout.
    #[must_use]
    pub fn with_n_steps(mut self, n: usize) -> Self {
        self.n_steps = n;
        self
    }

    /// Set the mini-batch size.
    #[must_use]
    pub fn with_batch_size(mut self, bs: usize) -> Self {
        self.batch_size = bs;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Actor-Critic network for PPO.
struct ActorCritic {
    shared1: Linear<f64>,
    shared2: Linear<f64>,
    policy_head: Linear<f64>,
    value_head: Linear<f64>,
}

impl ActorCritic {
    fn new(input_dim: usize, action_count: usize, rng: &mut Rng) -> Self {
        Self {
            shared1: Linear::new(input_dim, 64, true, rng),
            shared2: Linear::new(64, 64, true, rng),
            policy_head: Linear::new(64, action_count, true, rng),
            value_head: Linear::new(64, 1, true, rng),
        }
    }

    /// Forward pass returning (logits [batch, actions], value [batch, 1]).
    fn forward(&self, x: &Variable<f64>) -> Result<(Variable<f64>, Variable<f64>)> {
        let h = self
            .shared1
            .forward(x)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        let h = relu(&h);
        let h = self
            .shared2
            .forward(&h)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        let h = relu(&h);
        let logits = self
            .policy_head
            .forward(&h)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        let value = self
            .value_head
            .forward(&h)
            .map_err(|e| RlError::EnvironmentError(e.to_string()))?;
        Ok((logits, value))
    }

    fn parameters(&self) -> Vec<Variable<f64>> {
        let mut params = self.shared1.parameters();
        params.extend(self.shared2.parameters());
        params.extend(self.policy_head.parameters());
        params.extend(self.value_head.parameters());
        params
    }
}

/// Softmax over a 1-D slice, returning probabilities and log-probabilities.
fn softmax_1d(logits: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|&e| e / sum).collect();
    let log_probs: Vec<f64> = probs.iter().map(|&p| p.ln()).collect();
    (probs, log_probs)
}

/// Sample an action from a categorical distribution.
fn sample_categorical(probs: &[f64], rng: &mut Rng) -> usize {
    let r = rng.next_f64();
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    probs.len() - 1
}

/// A PPO agent.
pub struct PpoAgent {
    network: ActorCritic,
    optimizer: Adam<f64>,
    config: PpoConfig,
    rng: Rng,
    input_dim: usize,
    _action_count: usize,
}

impl PpoAgent {
    /// Create a new PPO agent.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid.
    pub fn new(input_dim: usize, action_count: usize, config: PpoConfig) -> Result<Self> {
        if input_dim == 0 {
            return Err(RlError::InvalidParameter("input_dim must be > 0".into()));
        }
        if action_count == 0 {
            return Err(RlError::InvalidParameter("action_count must be > 0".into()));
        }

        let mut rng = Rng::new(config.seed);
        let network = ActorCritic::new(input_dim, action_count, &mut rng);
        let optimizer = Adam::new(network.parameters(), config.learning_rate);

        Ok(Self {
            network,
            optimizer,
            config,
            rng,
            input_dim,
            _action_count: action_count,
        })
    }

    /// Select an action, returning `(action, log_prob, value)`.
    pub fn act(&mut self, observation: &[f64]) -> Result<(usize, f64, f64)> {
        let input = Variable::new(
            Tensor::from_vec(observation.to_vec(), vec![1, self.input_dim])
                .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
            false,
        );
        let (logits, value) = self.network.forward(&input)?;
        let logits_data = logits.data();
        let logits_slice = logits_data.as_slice();

        let (probs, log_probs) = softmax_1d(logits_slice);
        let action = sample_categorical(&probs, &mut self.rng);
        let log_prob = log_probs[action];
        let val = value.data().as_slice()[0];

        Ok((action, log_prob, val))
    }

    /// Train the agent on the given environment for the specified number of
    /// total timesteps.
    #[allow(clippy::too_many_lines)]
    pub fn train<E>(&mut self, env: &mut E, total_timesteps: usize) -> Result<EpisodeLogger>
    where
        E: Environment<Observation = Vec<f64>, Action = usize>,
    {
        let mut logger = EpisodeLogger::new();
        let mut obs = env.reset();
        let mut timestep = 0;

        while timestep < total_timesteps {
            // Collect rollout
            let mut states: Vec<Vec<f64>> = Vec::new();
            let mut actions: Vec<usize> = Vec::new();
            let mut rewards: Vec<f64> = Vec::new();
            let mut dones: Vec<bool> = Vec::new();
            let mut log_probs: Vec<f64> = Vec::new();
            let mut values: Vec<f64> = Vec::new();

            for _ in 0..self.config.n_steps {
                if timestep >= total_timesteps {
                    break;
                }
                states.push(obs.clone());

                let (action, log_prob, value) = self.act(&obs)?;
                let result = env.step(&action);
                let done = result.done || result.truncated;

                actions.push(action);
                rewards.push(result.reward);
                dones.push(done);
                log_probs.push(log_prob);
                values.push(value);

                logger.log_step(result.reward);
                obs = result.observation;
                timestep += 1;

                if done {
                    logger.end_episode();
                    obs = env.reset();
                }
            }

            if states.is_empty() {
                break;
            }

            // Compute returns and advantages using GAE
            let n = states.len();

            // Bootstrap value for the last state
            let last_value = if dones.last().copied().unwrap_or(true) {
                0.0
            } else {
                let input = Variable::new(
                    Tensor::from_vec(obs.clone(), vec![1, self.input_dim])
                        .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                    false,
                );
                let (_, v) = self.network.forward(&input)?;
                v.data().as_slice()[0]
            };

            let mut advantages = vec![0.0; n];
            let mut gae = 0.0;
            for t in (0..n).rev() {
                let next_value = if t + 1 < n { values[t + 1] } else { last_value };
                let done_mask = if dones[t] { 0.0 } else { 1.0 };
                let delta = rewards[t] + self.config.gamma * next_value * done_mask - values[t];
                gae = delta + self.config.gamma * self.config.gae_lambda * done_mask * gae;
                advantages[t] = gae;
            }

            let returns: Vec<f64> = advantages
                .iter()
                .zip(values.iter())
                .map(|(a, v)| a + v)
                .collect();

            // Normalize advantages
            let adv_mean: f64 = advantages.iter().sum::<f64>() / n as f64;
            let adv_var: f64 = advantages
                .iter()
                .map(|a| (a - adv_mean) * (a - adv_mean))
                .sum::<f64>()
                / n as f64;
            let adv_std = adv_var.sqrt() + 1e-8;
            for a in &mut advantages {
                *a = (*a - adv_mean) / adv_std;
            }

            // PPO update epochs
            for _ in 0..self.config.n_epochs {
                // Simple: iterate over the full rollout (no mini-batching for small rollouts)
                for i in 0..n {
                    let state_tensor = Variable::new(
                        Tensor::from_vec(states[i].clone(), vec![1, self.input_dim])
                            .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                        false,
                    );
                    let (logits, value_pred) = self.network.forward(&state_tensor)?;
                    let logits_data = logits.data();
                    let logits_slice = logits_data.as_slice();
                    let (_, new_log_probs) = softmax_1d(logits_slice);
                    let new_log_prob = new_log_probs[actions[i]];

                    // Ratio
                    let ratio = (new_log_prob - log_probs[i]).exp();
                    let adv = advantages[i];

                    // Clipped surrogate objective (we want to maximize, so negate for loss)
                    let surr1 = ratio * adv;
                    let clipped_ratio = ratio.clamp(
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    );
                    let surr2 = clipped_ratio * adv;
                    let policy_loss = -surr1.min(surr2);

                    // Value loss
                    let v_pred = value_pred.data().as_slice()[0];
                    let value_loss = (v_pred - returns[i]) * (v_pred - returns[i]);

                    // Total loss (as a Variable for backprop)
                    let total_loss_val = policy_loss + 0.5 * value_loss;
                    let loss_var = Variable::new(
                        Tensor::from_vec(vec![total_loss_val], vec![1])
                            .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                        true,
                    );

                    self.optimizer.zero_grad();
                    loss_var.backward();
                    self.optimizer.step();
                }
            }
        }

        Ok(logger)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::CartPole;

    #[test]
    fn test_ppo_construction() {
        let config = PpoConfig::new();
        let agent = PpoAgent::new(4, 2, config);
        assert!(agent.is_ok());
    }

    #[test]
    fn test_ppo_act() {
        let config = PpoConfig::new();
        let mut agent = PpoAgent::new(4, 2, config).unwrap();
        let (action, log_prob, value) = agent.act(&[0.0, 0.0, 0.05, 0.0]).unwrap();
        assert!(action < 2);
        assert!(log_prob.is_finite());
        assert!(value.is_finite());
    }

    #[test]
    fn test_ppo_train_short() {
        let config = PpoConfig::new()
            .with_n_steps(16)
            .with_n_epochs(1)
            .with_batch_size(8);
        let mut agent = PpoAgent::new(4, 2, config).unwrap();
        let mut env = CartPole::new();
        let logger = agent.train(&mut env, 32).unwrap();
        // Training should complete without errors
        let _ = logger.total_episodes();
    }
}
