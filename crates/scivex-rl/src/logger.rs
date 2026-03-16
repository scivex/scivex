//! Episode logging for RL training.

/// Tracks episode rewards and lengths over the course of training.
pub struct EpisodeLogger {
    /// Total reward for each completed episode.
    pub episode_rewards: Vec<f64>,
    /// Step count for each completed episode.
    pub episode_lengths: Vec<usize>,
    current_reward: f64,
    current_length: usize,
}

impl EpisodeLogger {
    /// Create a new, empty logger.
    #[must_use]
    pub fn new() -> Self {
        Self {
            episode_rewards: Vec::new(),
            episode_lengths: Vec::new(),
            current_reward: 0.0,
            current_length: 0,
        }
    }

    /// Record a single step's reward.
    pub fn log_step(&mut self, reward: f64) {
        self.current_reward += reward;
        self.current_length += 1;
    }

    /// Mark the current episode as finished and reset accumulators.
    pub fn end_episode(&mut self) {
        self.episode_rewards.push(self.current_reward);
        self.episode_lengths.push(self.current_length);
        self.current_reward = 0.0;
        self.current_length = 0;
    }

    /// Mean reward over the last `n` episodes.
    ///
    /// If fewer than `n` episodes have been completed, averages over all
    /// available episodes. Returns 0.0 if no episodes have been completed.
    #[must_use]
    pub fn mean_reward(&self, last_n: usize) -> f64 {
        if self.episode_rewards.is_empty() {
            return 0.0;
        }
        let n = last_n.min(self.episode_rewards.len());
        let start = self.episode_rewards.len() - n;
        let sum: f64 = self.episode_rewards[start..].iter().sum();
        sum / n as f64
    }

    /// Total number of completed episodes.
    #[must_use]
    pub fn total_episodes(&self) -> usize {
        self.episode_rewards.len()
    }
}

impl Default for EpisodeLogger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logger_log_and_end() {
        let mut logger = EpisodeLogger::new();
        logger.log_step(1.0);
        logger.log_step(2.0);
        logger.end_episode();
        assert_eq!(logger.total_episodes(), 1);
        assert!((logger.episode_rewards[0] - 3.0).abs() < 1e-10);
        assert_eq!(logger.episode_lengths[0], 2);
    }

    #[test]
    fn test_logger_mean_reward() {
        let mut logger = EpisodeLogger::new();
        // Episode 1: reward = 10
        for _ in 0..10 {
            logger.log_step(1.0);
        }
        logger.end_episode();
        // Episode 2: reward = 20
        for _ in 0..10 {
            logger.log_step(2.0);
        }
        logger.end_episode();

        assert!((logger.mean_reward(1) - 20.0).abs() < 1e-10);
        assert!((logger.mean_reward(2) - 15.0).abs() < 1e-10);
        assert!((logger.mean_reward(100) - 15.0).abs() < 1e-10); // more than available
    }

    #[test]
    fn test_logger_empty_mean() {
        let logger = EpisodeLogger::new();
        assert!((logger.mean_reward(10)).abs() < 1e-10);
    }
}
