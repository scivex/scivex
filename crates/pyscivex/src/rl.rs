//! Python bindings for scivex-rl — reinforcement learning.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scivex_rl::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn py_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

fn logger_to_dict(py: Python<'_>, logger: &EpisodeLogger) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("episode_rewards", &logger.episode_rewards)?;
    dict.set_item("episode_lengths", &logger.episode_lengths)?;
    dict.set_item("total_episodes", logger.total_episodes())?;
    Ok(dict.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Environments
// ---------------------------------------------------------------------------

/// Classic cart-pole balancing environment.
#[pyclass(name = "CartPole", unsendable)]
pub struct PyCartPole {
    inner: CartPole,
}

#[pymethods]
impl PyCartPole {
    /// Create a new CartPole environment with default parameters.
    #[new]
    fn new() -> Self {
        Self {
            inner: CartPole::new(),
        }
    }

    /// Reset environment and return initial observation.
    fn reset(&mut self) -> Vec<f64> {
        self.inner.reset()
    }

    /// Take an action, return {observation, reward, done, truncated}.
    fn step(&mut self, action: usize) -> PyResult<PyObject> {
        let r = self.inner.step(&action);
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("observation", &r.observation)?;
            dict.set_item("reward", r.reward)?;
            dict.set_item("done", r.done)?;
            dict.set_item("truncated", r.truncated)?;
            Ok(dict.into_any().unbind())
        })
    }

    /// Return the observation space dimensions.
    fn observation_shape(&self) -> Vec<usize> {
        self.inner.observation_shape().to_vec()
    }

    /// Return the number of discrete actions.
    fn action_count(&self) -> usize {
        self.inner.action_count()
    }

    /// Check if the episode is finished.
    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    /// Return a string representation of this CartPole environment.
    fn __repr__(&self) -> &'static str {
        "CartPole()"
    }
}

/// Mountain car environment — reach the goal by building momentum.
#[pyclass(name = "MountainCar", unsendable)]
pub struct PyMountainCar {
    inner: MountainCar,
}

#[pymethods]
impl PyMountainCar {
    /// Create a new MountainCar environment with default parameters.
    #[new]
    fn new() -> Self {
        Self {
            inner: MountainCar::new(),
        }
    }

    /// Reset environment and return initial observation.
    fn reset(&mut self) -> Vec<f64> {
        self.inner.reset()
    }

    /// Take an action, return {observation, reward, done, truncated}.
    fn step(&mut self, action: usize) -> PyResult<PyObject> {
        let r = self.inner.step(&action);
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("observation", &r.observation)?;
            dict.set_item("reward", r.reward)?;
            dict.set_item("done", r.done)?;
            dict.set_item("truncated", r.truncated)?;
            Ok(dict.into_any().unbind())
        })
    }

    /// Return the observation space dimensions.
    fn observation_shape(&self) -> Vec<usize> {
        self.inner.observation_shape().to_vec()
    }

    /// Return the number of discrete actions.
    fn action_count(&self) -> usize {
        self.inner.action_count()
    }

    /// Check if the episode is finished.
    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    /// Return a string representation of this MountainCar environment.
    fn __repr__(&self) -> &'static str {
        "MountainCar()"
    }
}

/// Grid world environment for tabular RL.
#[pyclass(name = "GridWorld", unsendable)]
pub struct PyGridWorld {
    inner: GridWorld,
}

#[pymethods]
impl PyGridWorld {
    /// Create a new GridWorld environment with the given grid size (default 5).
    #[new]
    #[pyo3(signature = (size = 5))]
    fn new(size: usize) -> PyResult<Self> {
        let inner = GridWorld::new(size).map_err(py_err)?;
        Ok(Self { inner })
    }

    /// Reset environment and return initial observation.
    fn reset(&mut self) -> Vec<f64> {
        self.inner.reset()
    }

    /// Take an action, return dict with observation, reward, done, and truncated.
    fn step(&mut self, action: usize) -> PyResult<PyObject> {
        let r = self.inner.step(&action);
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("observation", &r.observation)?;
            dict.set_item("reward", r.reward)?;
            dict.set_item("done", r.done)?;
            dict.set_item("truncated", r.truncated)?;
            Ok(dict.into_any().unbind())
        })
    }

    /// Return the observation space dimensions.
    fn observation_shape(&self) -> Vec<usize> {
        self.inner.observation_shape().to_vec()
    }

    /// Return the number of discrete actions.
    fn action_count(&self) -> usize {
        self.inner.action_count()
    }

    /// Check if the episode is finished.
    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    /// Return a string representation of this GridWorld environment.
    fn __repr__(&self) -> String {
        "GridWorld()".to_string()
    }
}

// ---------------------------------------------------------------------------
// DQN
// ---------------------------------------------------------------------------

/// Deep Q-Network (DQN) agent for discrete action spaces.
#[pyclass(name = "DQN", unsendable)]
pub struct PyDqn {
    inner: DqnAgent,
}

#[pymethods]
impl PyDqn {
    /// Create a new DQN agent with the given input dimensions, action count, and hyperparameters.
    #[new]
    #[pyo3(signature = (input_dim, action_count, learning_rate = 0.001, gamma = 0.99, epsilon = 1.0, epsilon_decay = 0.995, min_epsilon = 0.01, batch_size = 32, target_update_freq = 100, buffer_capacity = 10000, seed = 42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        action_count: usize,
        learning_rate: f64,
        gamma: f64,
        epsilon: f64,
        epsilon_decay: f64,
        min_epsilon: f64,
        batch_size: usize,
        target_update_freq: usize,
        buffer_capacity: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let config = DqnConfig::new()
            .with_learning_rate(learning_rate)
            .with_gamma(gamma)
            .with_epsilon(epsilon)
            .with_epsilon_decay(epsilon_decay)
            .with_min_epsilon(min_epsilon)
            .with_batch_size(batch_size)
            .with_target_update_freq(target_update_freq)
            .with_buffer_capacity(buffer_capacity)
            .with_seed(seed);
        let agent = DqnAgent::new(input_dim, action_count, config).map_err(py_err)?;
        Ok(Self { inner: agent })
    }

    /// Select an action for the given observation using epsilon-greedy policy.
    /// Returns the chosen action index.
    fn act(&mut self, observation: Vec<f64>) -> PyResult<usize> {
        self.inner.act(&observation).map_err(py_err)
    }

    /// Perform one training step by sampling from the replay buffer. Returns the loss.
    fn train_step(&mut self) -> PyResult<f64> {
        self.inner.train_step().map_err(py_err)
    }

    /// Train the agent on the CartPole environment for the given number of episodes.
    /// Returns a dict with episode_rewards, episode_lengths, and total_episodes.
    fn train_cartpole(&mut self, episodes: usize) -> PyResult<PyObject> {
        let mut env = CartPole::new();
        let logger = self.inner.train(&mut env, episodes).map_err(py_err)?;
        Python::with_gil(|py| logger_to_dict(py, &logger))
    }

    /// Train the agent on a GridWorld environment of the given size for the given number of episodes.
    /// Returns a dict with episode_rewards, episode_lengths, and total_episodes.
    fn train_gridworld(&mut self, size: usize, episodes: usize) -> PyResult<PyObject> {
        let mut env = GridWorld::new(size).map_err(py_err)?;
        let logger = self.inner.train(&mut env, episodes).map_err(py_err)?;
        Python::with_gil(|py| logger_to_dict(py, &logger))
    }

    /// Return a string representation of this DQN agent.
    fn __repr__(&self) -> &'static str {
        "DQN()"
    }
}

// ---------------------------------------------------------------------------
// PPO
// ---------------------------------------------------------------------------

/// Proximal Policy Optimization (PPO) agent for discrete action spaces.
#[pyclass(name = "PPO", unsendable)]
pub struct PyPpo {
    inner: PpoAgent,
}

#[pymethods]
impl PyPpo {
    /// Create a new PPO agent with the given input dimensions, action count, and hyperparameters.
    #[new]
    #[pyo3(signature = (input_dim, action_count, learning_rate = 0.0003, gamma = 0.99, gae_lambda = 0.95, clip_epsilon = 0.2, n_epochs = 4, n_steps = 128, batch_size = 32, seed = 42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        action_count: usize,
        learning_rate: f64,
        gamma: f64,
        gae_lambda: f64,
        clip_epsilon: f64,
        n_epochs: usize,
        n_steps: usize,
        batch_size: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let config = PpoConfig::new()
            .with_learning_rate(learning_rate)
            .with_gamma(gamma)
            .with_gae_lambda(gae_lambda)
            .with_clip_epsilon(clip_epsilon)
            .with_n_epochs(n_epochs)
            .with_n_steps(n_steps)
            .with_batch_size(batch_size)
            .with_seed(seed);
        let agent = PpoAgent::new(input_dim, action_count, config).map_err(py_err)?;
        Ok(Self { inner: agent })
    }

    /// Select an action for the given observation.
    /// Returns a tuple of (action_index, log_probability, value_estimate).
    fn act(&mut self, observation: Vec<f64>) -> PyResult<(usize, f64, f64)> {
        self.inner.act(&observation).map_err(py_err)
    }

    /// Train the agent on CartPole for the given number of total timesteps.
    /// Returns a dict with episode_rewards, episode_lengths, and total_episodes.
    fn train_cartpole(&mut self, total_timesteps: usize) -> PyResult<PyObject> {
        let mut env = CartPole::new();
        let logger = self
            .inner
            .train(&mut env, total_timesteps)
            .map_err(py_err)?;
        Python::with_gil(|py| logger_to_dict(py, &logger))
    }

    /// Return a string representation of this PPO agent.
    fn __repr__(&self) -> &'static str {
        "PPO()"
    }
}

// ---------------------------------------------------------------------------
// A2C
// ---------------------------------------------------------------------------

/// Advantage Actor-Critic (A2C) agent for discrete action spaces.
#[pyclass(name = "A2C", unsendable)]
pub struct PyA2c {
    inner: A2cAgent,
}

#[pymethods]
impl PyA2c {
    /// Create a new A2C agent with the given input dimensions, action count, and hyperparameters.
    #[new]
    #[pyo3(signature = (input_dim, action_count, learning_rate = 0.0007, gamma = 0.99, value_loss_coef = 0.5, entropy_coef = 0.01, n_steps = 5, seed = 42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        action_count: usize,
        learning_rate: f64,
        gamma: f64,
        value_loss_coef: f64,
        entropy_coef: f64,
        n_steps: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let config = A2cConfig::new()
            .with_learning_rate(learning_rate)
            .with_gamma(gamma)
            .with_value_loss_coef(value_loss_coef)
            .with_entropy_coef(entropy_coef)
            .with_n_steps(n_steps)
            .with_seed(seed);
        let agent = A2cAgent::new(input_dim, action_count, config).map_err(py_err)?;
        Ok(Self { inner: agent })
    }

    /// Select an action for the given observation.
    /// Returns a tuple of (action_index, log_probability, value_estimate).
    fn act(&mut self, observation: Vec<f64>) -> PyResult<(usize, f64, f64)> {
        self.inner.act(&observation).map_err(py_err)
    }

    /// Train the agent on CartPole for the given number of total timesteps.
    /// Returns a dict with episode_rewards, episode_lengths, and total_episodes.
    fn train_cartpole(&mut self, total_timesteps: usize) -> PyResult<PyObject> {
        let mut env = CartPole::new();
        let logger = self
            .inner
            .train(&mut env, total_timesteps)
            .map_err(py_err)?;
        Python::with_gil(|py| logger_to_dict(py, &logger))
    }

    /// Return a string representation of this A2C agent.
    fn __repr__(&self) -> &'static str {
        "A2C()"
    }
}

// ---------------------------------------------------------------------------
// SAC
// ---------------------------------------------------------------------------

/// Soft Actor-Critic (SAC) agent for continuous action spaces.
#[pyclass(name = "SAC", unsendable)]
pub struct PySac {
    inner: SacAgent,
}

#[pymethods]
impl PySac {
    /// Create a new SAC agent with the given state/action dimensions and hyperparameters.
    #[new]
    #[pyo3(signature = (state_dim, action_dim, learning_rate = 0.0003, gamma = 0.99, tau = 0.005, alpha = 0.2, batch_size = 64, buffer_capacity = 10000, seed = 42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        state_dim: usize,
        action_dim: usize,
        learning_rate: f64,
        gamma: f64,
        tau: f64,
        alpha: f64,
        batch_size: usize,
        buffer_capacity: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let config = SacConfig::new()
            .with_learning_rate(learning_rate)
            .with_gamma(gamma)
            .with_tau(tau)
            .with_alpha(alpha)
            .with_batch_size(batch_size)
            .with_buffer_capacity(buffer_capacity)
            .with_seed(seed);
        let agent = SacAgent::new(state_dim, action_dim, config).map_err(py_err)?;
        Ok(Self { inner: agent })
    }

    /// Select a continuous action for the given state. Returns a list of action values.
    fn select_action(&mut self, state: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.select_action(&state).map_err(py_err)
    }

    /// Perform one training step by sampling from the replay buffer. Returns the loss.
    fn train_step(&mut self) -> PyResult<f64> {
        self.inner.train_step().map_err(py_err)
    }

    /// Return the dimensionality of the state space.
    fn state_dim(&self) -> usize {
        self.inner.state_dim()
    }

    /// Return the dimensionality of the action space.
    fn action_dim(&self) -> usize {
        self.inner.action_dim()
    }

    /// Return a string representation of this SAC agent.
    fn __repr__(&self) -> &'static str {
        "SAC()"
    }
}

// ---------------------------------------------------------------------------
// TD3
// ---------------------------------------------------------------------------

/// Twin Delayed DDPG (TD3) agent for continuous action spaces.
#[pyclass(name = "TD3", unsendable)]
pub struct PyTd3 {
    inner: Td3Agent,
}

#[pymethods]
impl PyTd3 {
    /// Create a new TD3 agent with the given state/action dimensions and hyperparameters.
    #[new]
    #[pyo3(signature = (state_dim, action_dim, learning_rate = 0.0003, gamma = 0.99, tau = 0.005, policy_noise = 0.2, noise_clip = 0.5, policy_delay = 2, batch_size = 64, buffer_capacity = 10000, seed = 42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        state_dim: usize,
        action_dim: usize,
        learning_rate: f64,
        gamma: f64,
        tau: f64,
        policy_noise: f64,
        noise_clip: f64,
        policy_delay: usize,
        batch_size: usize,
        buffer_capacity: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let config = Td3Config::new()
            .with_learning_rate(learning_rate)
            .with_gamma(gamma)
            .with_tau(tau)
            .with_policy_noise(policy_noise)
            .with_noise_clip(noise_clip)
            .with_policy_delay(policy_delay)
            .with_batch_size(batch_size)
            .with_buffer_capacity(buffer_capacity)
            .with_seed(seed);
        let agent = Td3Agent::new(state_dim, action_dim, config).map_err(py_err)?;
        Ok(Self { inner: agent })
    }

    /// Select a continuous action for the given state with added exploration noise.
    /// Returns a list of action values.
    #[pyo3(signature = (state, exploration_noise = 0.1))]
    fn select_action(&mut self, state: Vec<f64>, exploration_noise: f64) -> PyResult<Vec<f64>> {
        self.inner
            .select_action(&state, exploration_noise)
            .map_err(py_err)
    }

    /// Perform one training step by sampling from the replay buffer. Returns the loss.
    fn train_step(&mut self) -> PyResult<f64> {
        self.inner.train_step().map_err(py_err)
    }

    /// Return the dimensionality of the state space.
    fn state_dim(&self) -> usize {
        self.inner.state_dim()
    }

    /// Return the dimensionality of the action space.
    fn action_dim(&self) -> usize {
        self.inner.action_dim()
    }

    /// Return a string representation of this TD3 agent.
    fn __repr__(&self) -> &'static str {
        "TD3()"
    }
}

// ---------------------------------------------------------------------------
// EpisodeLogger
// ---------------------------------------------------------------------------

/// Logger for tracking episode rewards and lengths during RL training.
#[pyclass(name = "EpisodeLogger", unsendable)]
pub struct PyEpisodeLogger {
    inner: EpisodeLogger,
}

#[pymethods]
impl PyEpisodeLogger {
    /// Create a new empty EpisodeLogger.
    #[new]
    fn new() -> Self {
        Self {
            inner: EpisodeLogger::new(),
        }
    }

    /// Log a single step reward within the current episode.
    fn log_step(&mut self, reward: f64) {
        self.inner.log_step(reward);
    }

    /// Mark the current episode as finished and record its cumulative stats.
    fn end_episode(&mut self) {
        self.inner.end_episode();
    }

    /// Return the mean reward over the last `last_n` episodes.
    fn mean_reward(&self, last_n: usize) -> f64 {
        self.inner.mean_reward(last_n)
    }

    /// Return the total number of completed episodes.
    fn total_episodes(&self) -> usize {
        self.inner.total_episodes()
    }

    /// List of cumulative rewards for each completed episode.
    #[getter]
    fn episode_rewards(&self) -> Vec<f64> {
        self.inner.episode_rewards.clone()
    }

    /// List of step counts for each completed episode.
    #[getter]
    fn episode_lengths(&self) -> Vec<usize> {
        self.inner.episode_lengths.clone()
    }

    /// Return a string representation of this EpisodeLogger.
    fn __repr__(&self) -> String {
        format!("EpisodeLogger(episodes={})", self.inner.total_episodes())
    }
}

// ---------------------------------------------------------------------------
// HER Replay Buffer
// ---------------------------------------------------------------------------

/// Hindsight Experience Replay (HER) buffer for goal-conditioned RL.
///
/// Stores transitions with goal-conditioned relabeling to improve sample
/// efficiency in sparse-reward environments. Supports "future", "final",
/// and "episode" relabeling strategies.
#[pyclass(name = "HerReplayBuffer", unsendable)]
pub struct PyHerReplayBuffer {
    inner: scivex_rl::HerReplayBuffer,
    rng: scivex_core::random::Rng,
}

#[pymethods]
impl PyHerReplayBuffer {
    /// Create a new HER replay buffer.
    ///
    /// # Arguments
    ///
    /// * `capacity` — Maximum number of transitions the buffer can hold.
    /// * `strategy` — Relabeling strategy: `"future"`, `"final"`, or `"episode"`.
    /// * `k` — For the `"future"` strategy, how many future goals to sample
    ///   per transition (default 4, ignored for other strategies).
    /// * `seed` — Random seed for reproducibility.
    #[new]
    #[pyo3(signature = (capacity, strategy="final", k=4, seed=42))]
    fn new(capacity: usize, strategy: &str, k: usize, seed: u64) -> PyResult<Self> {
        let strat = match strategy.to_lowercase().as_str() {
            "future" => scivex_rl::HerStrategy::Future(k),
            "final" => scivex_rl::HerStrategy::Final,
            "episode" => scivex_rl::HerStrategy::Episode,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "strategy must be 'future', 'final', or 'episode'",
                ));
            }
        };

        // Default sparse reward: 0 if achieved == desired, -1 otherwise.
        let reward_fn = |achieved: &[f64], desired: &[f64]| -> f64 {
            if achieved == desired { 0.0 } else { -1.0 }
        };

        Ok(Self {
            inner: scivex_rl::HerReplayBuffer::new(capacity, strat, reward_fn),
            rng: scivex_core::random::Rng::new(seed),
        })
    }

    /// Push a goal-conditioned transition into the episode buffer.
    ///
    /// Transitions are not available for sampling until `end_episode()` is called.
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
        desired_goal: Vec<f64>,
        achieved_goal: Vec<f64>,
    ) {
        self.inner.push(scivex_rl::GoalTransition {
            state,
            action,
            reward,
            next_state,
            done,
            desired_goal,
            achieved_goal,
        });
    }

    /// End the current episode, apply HER relabeling, and flush transitions
    /// into the main ring buffer.
    fn end_episode(&mut self) {
        self.inner.end_episode(&mut self.rng);
    }

    /// Sample a random batch of transitions from the main buffer.
    ///
    /// Returns a list of dicts, each with keys: `state`, `action`, `reward`,
    /// `next_state`, `done`, `desired_goal`, `achieved_goal`.
    fn sample(&mut self, batch_size: usize) -> PyResult<Vec<PyObject>> {
        let batch = self
            .inner
            .sample(batch_size, &mut self.rng)
            .map_err(py_err)?;
        Python::with_gil(|py| {
            let mut result = Vec::with_capacity(batch.len());
            for t in &batch {
                let dict = PyDict::new(py);
                dict.set_item("state", &t.state)?;
                dict.set_item("action", t.action)?;
                dict.set_item("reward", t.reward)?;
                dict.set_item("next_state", &t.next_state)?;
                dict.set_item("done", t.done)?;
                dict.set_item("desired_goal", &t.desired_goal)?;
                dict.set_item("achieved_goal", &t.achieved_goal)?;
                result.push(dict.into_any().unbind());
            }
            Ok(result)
        })
    }

    /// Return the number of transitions stored in the main buffer.
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return whether the main buffer is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return the number of transitions in the current (incomplete) episode buffer.
    fn episode_len(&self) -> usize {
        self.inner.episode_len()
    }

    /// Return a string representation of this HER replay buffer.
    fn __repr__(&self) -> String {
        format!(
            "HerReplayBuffer(len={}, episode_len={})",
            self.inner.len(),
            self.inner.episode_len()
        )
    }
}

// ---------------------------------------------------------------------------
// ReplayBuffer (off-policy experience replay)
// ---------------------------------------------------------------------------

/// Experience replay buffer for off-policy RL algorithms (DQN, etc.).
#[pyclass(name = "ReplayBuffer")]
pub struct PyReplayBuffer {
    inner: scivex_rl::ReplayBuffer,
    rng: scivex_core::random::Rng,
}

#[pymethods]
impl PyReplayBuffer {
    /// Create a new replay buffer with the given capacity.
    #[new]
    #[pyo3(signature = (capacity, seed=42))]
    fn new(capacity: usize, seed: u64) -> Self {
        Self {
            inner: scivex_rl::ReplayBuffer::new(capacity),
            rng: scivex_core::random::Rng::new(seed),
        }
    }

    /// Push a transition (state, action, reward, next_state, done).
    fn push(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
    ) {
        self.inner.push(state, action, reward, next_state, done);
    }

    /// Sample a random batch of transitions.
    fn sample(&mut self, batch_size: usize) -> PyResult<PyObject> {
        let batch = self
            .inner
            .sample(batch_size, &mut self.rng)
            .map_err(py_err)?;
        Python::with_gil(|py| {
            let d = PyDict::new(py);
            d.set_item("states", &batch.states)?;
            d.set_item("actions", &batch.actions)?;
            d.set_item("rewards", &batch.rewards)?;
            d.set_item("next_states", &batch.next_states)?;
            d.set_item("dones", &batch.dones)?;
            Ok(d.into_any().unbind())
        })
    }

    /// Return the number of transitions stored.
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return whether the buffer is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("ReplayBuffer(len={})", self.inner.len())
    }
}

// ---------------------------------------------------------------------------
// CooperativeNavigation (multi-agent environment)
// ---------------------------------------------------------------------------

/// Cooperative navigation multi-agent environment.
///
/// Multiple agents must navigate to landmark positions while avoiding collisions.
#[pyclass(name = "CooperativeNavigation")]
pub struct PyCooperativeNavigation {
    inner: scivex_rl::env::CooperativeNavigation,
}

#[pymethods]
impl PyCooperativeNavigation {
    /// Create a new cooperative navigation environment.
    #[new]
    #[pyo3(signature = (num_agents=3, arena_size=1.0, seed=42))]
    fn new(num_agents: usize, arena_size: f64, seed: u64) -> Self {
        Self {
            inner: scivex_rl::env::CooperativeNavigation::new(num_agents, arena_size, seed),
        }
    }

    /// Reset the environment and return initial observations for all agents.
    fn reset(&mut self) -> Vec<Vec<f64>> {
        use scivex_rl::env::MultiAgentEnv;
        self.inner.reset()
    }

    /// Step with actions for all agents. Returns (observations, rewards, dones, all_done).
    fn step(&mut self, actions: Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<f64>, Vec<bool>, bool) {
        use scivex_rl::env::MultiAgentEnv;
        self.inner.step(&actions)
    }

    /// Return the number of agents.
    fn num_agents(&self) -> usize {
        use scivex_rl::env::MultiAgentEnv;
        self.inner.num_agents()
    }

    /// Return the observation dimension per agent.
    fn observation_dim(&self) -> usize {
        use scivex_rl::env::MultiAgentEnv;
        self.inner.observation_dim()
    }

    /// Return the action dimension per agent.
    fn action_dim(&self) -> usize {
        use scivex_rl::env::MultiAgentEnv;
        self.inner.action_dim()
    }

    fn __repr__(&self) -> String {
        use scivex_rl::env::MultiAgentEnv;
        format!("CooperativeNavigation(agents={})", self.inner.num_agents())
    }
}

// ---------------------------------------------------------------------------
// Register submodule
// ---------------------------------------------------------------------------

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "rl")?;

    // Environments
    m.add_class::<PyCartPole>()?;
    m.add_class::<PyMountainCar>()?;
    m.add_class::<PyGridWorld>()?;

    // Algorithms
    m.add_class::<PyDqn>()?;
    m.add_class::<PyPpo>()?;
    m.add_class::<PyA2c>()?;
    m.add_class::<PySac>()?;
    m.add_class::<PyTd3>()?;

    // Logger
    m.add_class::<PyEpisodeLogger>()?;

    // Replay buffers
    m.add_class::<PyReplayBuffer>()?;
    m.add_class::<PyHerReplayBuffer>()?;

    // Multi-agent
    m.add_class::<PyCooperativeNavigation>()?;

    parent.add_submodule(&m)?;
    Ok(())
}
