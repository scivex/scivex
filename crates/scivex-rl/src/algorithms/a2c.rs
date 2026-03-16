//! Advantage Actor-Critic (A2C) — synchronous variant.

use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::functional::relu;
use scivex_nn::layer::{Layer, Linear};
use scivex_nn::optim::{Adam, Optimizer};

use crate::env::Environment;
use crate::error::{Result, RlError};
use crate::logger::EpisodeLogger;

/// Configuration for the A2C agent.
pub struct A2cConfig {
    /// Learning rate.
    pub learning_rate: f64,
    /// Discount factor.
    pub gamma: f64,
    /// Weight of the value loss term.
    pub value_loss_coef: f64,
    /// Weight of the entropy bonus.
    pub entropy_coef: f64,
    /// Number of environment steps per update.
    pub n_steps: usize,
    /// Random seed.
    pub seed: u64,
}

impl A2cConfig {
    /// Create a new A2C configuration with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            learning_rate: 0.0007,
            gamma: 0.99,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
            n_steps: 5,
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

    /// Set the value loss coefficient.
    #[must_use]
    pub fn with_value_loss_coef(mut self, coef: f64) -> Self {
        self.value_loss_coef = coef;
        self
    }

    /// Set the entropy coefficient.
    #[must_use]
    pub fn with_entropy_coef(mut self, coef: f64) -> Self {
        self.entropy_coef = coef;
        self
    }

    /// Set the number of steps per update.
    #[must_use]
    pub fn with_n_steps(mut self, n: usize) -> Self {
        self.n_steps = n;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for A2cConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Actor-Critic network for A2C.
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

/// An A2C agent.
pub struct A2cAgent {
    network: ActorCritic,
    optimizer: Adam<f64>,
    config: A2cConfig,
    rng: Rng,
    input_dim: usize,
    _action_count: usize,
}

impl A2cAgent {
    /// Create a new A2C agent.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid.
    pub fn new(input_dim: usize, action_count: usize, config: A2cConfig) -> Result<Self> {
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
            // Collect n_steps of experience
            let mut states: Vec<Vec<f64>> = Vec::new();
            let mut actions_taken: Vec<usize> = Vec::new();
            let mut rewards: Vec<f64> = Vec::new();
            let mut dones: Vec<bool> = Vec::new();
            let mut log_probs_old: Vec<f64> = Vec::new();
            let mut values: Vec<f64> = Vec::new();

            for _ in 0..self.config.n_steps {
                if timestep >= total_timesteps {
                    break;
                }
                states.push(obs.clone());

                let (action, log_prob, value) = self.act(&obs)?;
                let result = env.step(&action);
                let done = result.done || result.truncated;

                actions_taken.push(action);
                rewards.push(result.reward);
                dones.push(done);
                log_probs_old.push(log_prob);
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

            let n = states.len();

            // Bootstrap value for last state
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

            // Compute returns (n-step returns)
            let mut returns = vec![0.0; n];
            let mut r = last_value;
            for t in (0..n).rev() {
                let done_mask = if dones[t] { 0.0 } else { 1.0 };
                r = rewards[t] + self.config.gamma * r * done_mask;
                returns[t] = r;
            }

            // Compute advantages
            let advantages: Vec<f64> = returns
                .iter()
                .zip(values.iter())
                .map(|(r_val, v)| r_val - v)
                .collect();

            // A2C update: single pass through collected data
            let mut total_policy_loss = 0.0;
            let mut total_value_loss = 0.0;
            let mut total_entropy = 0.0;

            for i in 0..n {
                let state_tensor = Variable::new(
                    Tensor::from_vec(states[i].clone(), vec![1, self.input_dim])
                        .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                    false,
                );
                let (logits, value_pred) = self.network.forward(&state_tensor)?;
                let logits_data = logits.data();
                let logits_slice = logits_data.as_slice();
                let (probs, new_log_probs) = softmax_1d(logits_slice);

                let log_prob = new_log_probs[actions_taken[i]];

                // Policy loss: -log_prob * advantage
                let policy_loss = -log_prob * advantages[i];

                // Value loss
                let v_pred = value_pred.data().as_slice()[0];
                let value_loss = (v_pred - returns[i]) * (v_pred - returns[i]);

                // Entropy bonus: -sum(p * log(p))
                let entropy: f64 = -probs
                    .iter()
                    .zip(new_log_probs.iter())
                    .map(|(p, lp)| p * lp)
                    .sum::<f64>();

                total_policy_loss += policy_loss;
                total_value_loss += value_loss;
                total_entropy += entropy;
            }

            // Compute total loss and update
            let n_f = n as f64;
            let loss_val = total_policy_loss / n_f
                + self.config.value_loss_coef * total_value_loss / n_f
                - self.config.entropy_coef * total_entropy / n_f;

            let loss_var = Variable::new(
                Tensor::from_vec(vec![loss_val], vec![1])
                    .map_err(|e| RlError::EnvironmentError(e.to_string()))?,
                true,
            );

            self.optimizer.zero_grad();
            loss_var.backward();
            self.optimizer.step();
        }

        Ok(logger)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::CartPole;

    #[test]
    fn test_a2c_construction() {
        let config = A2cConfig::new();
        let agent = A2cAgent::new(4, 2, config);
        assert!(agent.is_ok());
    }

    #[test]
    fn test_a2c_act() {
        let config = A2cConfig::new();
        let mut agent = A2cAgent::new(4, 2, config).unwrap();
        let (action, log_prob, value) = agent.act(&[0.0, 0.0, 0.05, 0.0]).unwrap();
        assert!(action < 2);
        assert!(log_prob.is_finite());
        assert!(value.is_finite());
    }

    #[test]
    fn test_a2c_train_short() {
        let config = A2cConfig::new().with_n_steps(8);
        let mut agent = A2cAgent::new(4, 2, config).unwrap();
        let mut env = CartPole::new();
        let logger = agent.train(&mut env, 20).unwrap();
        // Training should complete without errors
        let _ = logger.total_episodes();
    }
}
