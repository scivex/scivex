//! Hindsight Experience Replay (HER) for goal-conditioned reinforcement learning.
//!
//! HER improves sample efficiency in sparse-reward, goal-conditioned settings
//! by relabeling past transitions with alternative goals that were actually
//! achieved during the episode. This turns failures into useful learning signal.
//!
//! # References
//!
//! Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017.

use scivex_core::random::Rng;

use crate::error::{Result, RlError};

/// Relabeling strategy for HER.
#[derive(Debug, Clone)]
pub enum HerStrategy {
    /// Replace goal with a future achieved goal from the same episode.
    /// The parameter specifies how many future goals to sample per transition.
    Future(usize),
    /// Replace goal with the final achieved goal of the episode.
    Final,
    /// Replace goal with a random achieved goal from the episode.
    Episode,
}

/// A single goal-conditioned transition.
#[derive(Debug, Clone)]
pub struct GoalTransition {
    /// Observation / state vector.
    pub state: Vec<f64>,
    /// Discrete action taken.
    pub action: usize,
    /// Scalar reward received.
    pub reward: f64,
    /// Next observation / state vector.
    pub next_state: Vec<f64>,
    /// Whether the episode terminated after this transition.
    pub done: bool,
    /// The goal the agent was trying to reach.
    pub desired_goal: Vec<f64>,
    /// The goal the agent actually achieved at this step.
    pub achieved_goal: Vec<f64>,
}

/// Goal-conditioned replay buffer with Hindsight Experience Replay.
///
/// Transitions are collected into an episode buffer via [`push`](Self::push).
/// When [`end_episode`](Self::end_episode) is called, the original transitions
/// plus HER-relabeled copies are flushed into the main ring buffer.
#[allow(clippy::type_complexity)]
pub struct HerReplayBuffer {
    buffer: Vec<GoalTransition>,
    capacity: usize,
    position: usize,
    len: usize,
    episode_buffer: Vec<GoalTransition>,
    strategy: HerStrategy,
    reward_fn: Box<dyn Fn(&[f64], &[f64]) -> f64>,
}

impl HerReplayBuffer {
    /// Create a new HER replay buffer.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of transitions the buffer can hold.
    /// * `strategy` - The HER relabeling strategy to use.
    /// * `reward_fn` - A function `(achieved_goal, desired_goal) -> reward`
    ///   used to recompute rewards for relabeled transitions.
    #[must_use]
    pub fn new(
        capacity: usize,
        strategy: HerStrategy,
        reward_fn: impl Fn(&[f64], &[f64]) -> f64 + 'static,
    ) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
            len: 0,
            episode_buffer: Vec::new(),
            strategy,
            reward_fn: Box::new(reward_fn),
        }
    }

    /// Add a transition to the current episode buffer.
    ///
    /// Transitions are not available for sampling until
    /// [`end_episode`](Self::end_episode) is called.
    pub fn push(&mut self, transition: GoalTransition) {
        self.episode_buffer.push(transition);
    }

    /// End the current episode, apply HER relabeling, and flush all
    /// transitions (original + relabeled) into the main ring buffer.
    pub fn end_episode(&mut self, rng: &mut Rng) {
        let episode = std::mem::take(&mut self.episode_buffer);
        if episode.is_empty() {
            return;
        }

        // Store original transitions.
        for t in &episode {
            self.store(t.clone());
        }

        // Generate relabeled transitions according to the chosen strategy.
        let ep_len = episode.len();
        match &self.strategy {
            HerStrategy::Future(k) => {
                let k = *k;
                for (i, t) in episode.iter().enumerate() {
                    // Only transitions with at least one future step can be relabeled.
                    if i + 1 >= ep_len {
                        continue;
                    }
                    for _ in 0..k {
                        // Sample a random index strictly after i.
                        let future_idx =
                            i + 1 + (rng.next_f64() * (ep_len - i - 1) as f64) as usize;
                        let future_idx = future_idx.min(ep_len - 1);
                        let new_goal = episode[future_idx].achieved_goal.clone();
                        let new_reward = (self.reward_fn)(&t.achieved_goal, &new_goal);
                        self.store(GoalTransition {
                            state: t.state.clone(),
                            action: t.action,
                            reward: new_reward,
                            next_state: t.next_state.clone(),
                            done: t.done,
                            desired_goal: new_goal,
                            achieved_goal: t.achieved_goal.clone(),
                        });
                    }
                }
            }
            HerStrategy::Final => {
                let final_goal = episode[ep_len - 1].achieved_goal.clone();
                for t in &episode {
                    let new_reward = (self.reward_fn)(&t.achieved_goal, &final_goal);
                    self.store(GoalTransition {
                        state: t.state.clone(),
                        action: t.action,
                        reward: new_reward,
                        next_state: t.next_state.clone(),
                        done: t.done,
                        desired_goal: final_goal.clone(),
                        achieved_goal: t.achieved_goal.clone(),
                    });
                }
            }
            HerStrategy::Episode => {
                for t in &episode {
                    let rand_idx = (rng.next_f64() * ep_len as f64) as usize;
                    let rand_idx = rand_idx.min(ep_len - 1);
                    let new_goal = episode[rand_idx].achieved_goal.clone();
                    let new_reward = (self.reward_fn)(&t.achieved_goal, &new_goal);
                    self.store(GoalTransition {
                        state: t.state.clone(),
                        action: t.action,
                        reward: new_reward,
                        next_state: t.next_state.clone(),
                        done: t.done,
                        desired_goal: new_goal,
                        achieved_goal: t.achieved_goal.clone(),
                    });
                }
            }
        }
    }

    /// Sample a random batch of transitions from the main buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch_size` exceeds the number of stored transitions.
    pub fn sample(&self, batch_size: usize, rng: &mut Rng) -> Result<Vec<GoalTransition>> {
        if batch_size > self.len {
            return Err(RlError::InvalidParameter(format!(
                "batch_size ({batch_size}) exceeds buffer length ({})",
                self.len
            )));
        }

        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let idx = (rng.next_f64() * self.len as f64) as usize;
            let idx = idx.min(self.len - 1);
            batch.push(self.buffer[idx].clone());
        }
        Ok(batch)
    }

    /// Return the number of transitions currently stored in the main buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether the main buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the number of transitions in the current (incomplete) episode buffer.
    #[must_use]
    pub fn episode_len(&self) -> usize {
        self.episode_buffer.len()
    }

    /// Store a transition into the ring buffer.
    fn store(&mut self, transition: GoalTransition) {
        if self.len < self.capacity {
            self.buffer.push(transition);
            self.len += 1;
        } else {
            self.buffer[self.position] = transition;
        }
        self.position = (self.position + 1) % self.capacity;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a simple transition.
    fn make_transition(step: usize) -> GoalTransition {
        GoalTransition {
            state: vec![step as f64],
            action: step % 3,
            reward: -1.0,
            next_state: vec![step as f64 + 1.0],
            done: false,
            desired_goal: vec![100.0],
            achieved_goal: vec![step as f64 * 10.0],
        }
    }

    /// Sparse reward: 0 if achieved == desired, -1 otherwise.
    fn sparse_reward(achieved: &[f64], desired: &[f64]) -> f64 {
        if achieved == desired { 0.0 } else { -1.0 }
    }

    // ── Push and episode buffer ─────────────────────────────────────────

    #[test]
    fn test_push_populates_episode_buffer() {
        let mut buf = HerReplayBuffer::new(100, HerStrategy::Final, sparse_reward);
        assert!(buf.is_empty());
        assert_eq!(buf.episode_len(), 0);

        buf.push(make_transition(0));
        buf.push(make_transition(1));

        // Main buffer is still empty; transitions live in episode buffer.
        assert!(buf.is_empty());
        assert_eq!(buf.episode_len(), 2);
    }

    // ── End episode with Final strategy ─────────────────────────────────

    #[test]
    fn test_end_episode_final_strategy() {
        let mut buf = HerReplayBuffer::new(100, HerStrategy::Final, sparse_reward);
        let mut rng = Rng::new(42);

        for i in 0..5 {
            buf.push(make_transition(i));
        }
        buf.end_episode(&mut rng);

        // 5 original + 5 relabeled = 10
        assert_eq!(buf.len(), 10);
        assert_eq!(buf.episode_len(), 0);

        // All relabeled transitions should have desired_goal == final achieved_goal.
        // Final achieved_goal = step 4 -> 40.0
        let relabeled_count = buf
            .buffer
            .iter()
            .filter(|t| t.desired_goal == vec![40.0])
            .count();
        assert_eq!(relabeled_count, 5);
    }

    // ── End episode with Future(k) strategy ─────────────────────────────

    #[test]
    fn test_end_episode_future_strategy() {
        let k = 4;
        let mut buf = HerReplayBuffer::new(1000, HerStrategy::Future(k), sparse_reward);
        let mut rng = Rng::new(7);

        let ep_len = 10;
        for i in 0..ep_len {
            buf.push(make_transition(i));
        }
        buf.end_episode(&mut rng);

        // Original transitions: 10
        // Relabeled: each of the first 9 transitions gets k relabeled copies
        // (the last transition has no future, so 0 relabeled)
        let expected = ep_len + (ep_len - 1) * k;
        assert_eq!(buf.len(), expected);
    }

    // ── End episode with Episode strategy ───────────────────────────────

    #[test]
    fn test_end_episode_episode_strategy() {
        let mut buf = HerReplayBuffer::new(1000, HerStrategy::Episode, sparse_reward);
        let mut rng = Rng::new(99);

        let ep_len = 8;
        for i in 0..ep_len {
            buf.push(make_transition(i));
        }
        buf.end_episode(&mut rng);

        // 8 original + 8 relabeled = 16
        assert_eq!(buf.len(), ep_len * 2);

        // Every relabeled transition's desired_goal must be an achieved_goal
        // from the episode (i.e., a multiple of 10 in 0..80).
        let valid_goals: Vec<f64> = (0..ep_len).map(|i| i as f64 * 10.0).collect();
        for t in &buf.buffer {
            if t.desired_goal != vec![100.0] {
                // This is a relabeled transition.
                assert!(
                    valid_goals.contains(&t.desired_goal[0]),
                    "unexpected desired_goal: {:?}",
                    t.desired_goal
                );
            }
        }
    }

    // ── Empty episode is a no-op ────────────────────────────────────────

    #[test]
    fn test_end_episode_empty() {
        let mut buf = HerReplayBuffer::new(100, HerStrategy::Final, sparse_reward);
        let mut rng = Rng::new(0);

        buf.end_episode(&mut rng);
        assert!(buf.is_empty());
    }

    // ── Sample ──────────────────────────────────────────────────────────

    #[test]
    fn test_sample_returns_correct_batch_size() {
        let mut buf = HerReplayBuffer::new(100, HerStrategy::Final, sparse_reward);
        let mut rng = Rng::new(42);

        for i in 0..5 {
            buf.push(make_transition(i));
        }
        buf.end_episode(&mut rng);

        let batch = buf.sample(4, &mut rng).unwrap();
        assert_eq!(batch.len(), 4);
    }

    #[test]
    fn test_sample_too_large() {
        let buf = HerReplayBuffer::new(100, HerStrategy::Final, sparse_reward);
        let mut rng = Rng::new(0);
        assert!(buf.sample(1, &mut rng).is_err());
    }

    // ── Ring buffer wrapping ────────────────────────────────────────────

    #[test]
    fn test_buffer_wrapping() {
        // Capacity 10, but we insert many more transitions over multiple episodes.
        let mut buf = HerReplayBuffer::new(10, HerStrategy::Final, sparse_reward);
        let mut rng = Rng::new(1);

        // Episode 1: 5 transitions -> 5 original + 5 relabeled = 10 (fills buffer)
        for i in 0..5 {
            buf.push(make_transition(i));
        }
        buf.end_episode(&mut rng);
        assert_eq!(buf.len(), 10);

        // Episode 2: 3 transitions -> 3 original + 3 relabeled = 6 (overwrites oldest)
        for i in 10..13 {
            buf.push(make_transition(i));
        }
        buf.end_episode(&mut rng);

        // Buffer should still be at capacity.
        assert_eq!(buf.len(), 10);

        // We should be able to sample without issues.
        let batch = buf.sample(5, &mut rng).unwrap();
        assert_eq!(batch.len(), 5);
    }

    // ── Reward relabeling correctness ───────────────────────────────────

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_reward_relabeling() {
        // Use a reward function where matching goal gives +1.0
        let reward_fn = |achieved: &[f64], desired: &[f64]| -> f64 {
            if achieved == desired { 1.0 } else { -1.0 }
        };

        let mut buf = HerReplayBuffer::new(100, HerStrategy::Final, reward_fn);
        let mut rng = Rng::new(0);

        // Build an episode where the last step achieves goal [50.0].
        buf.push(GoalTransition {
            state: vec![0.0],
            action: 0,
            reward: -1.0,
            next_state: vec![1.0],
            done: false,
            desired_goal: vec![999.0],
            achieved_goal: vec![10.0],
        });
        buf.push(GoalTransition {
            state: vec![1.0],
            action: 1,
            reward: -1.0,
            next_state: vec![2.0],
            done: false,
            desired_goal: vec![999.0],
            achieved_goal: vec![50.0],
        });
        buf.push(GoalTransition {
            state: vec![2.0],
            action: 0,
            reward: -1.0,
            next_state: vec![3.0],
            done: true,
            desired_goal: vec![999.0],
            achieved_goal: vec![50.0],
        });

        buf.end_episode(&mut rng);

        // Final strategy: relabeled goal = [50.0] (last achieved_goal).
        // Transition at step 1 achieved [50.0], so its relabeled reward should be 1.0.
        let relabeled: Vec<_> = buf
            .buffer
            .iter()
            .filter(|t| t.desired_goal == vec![50.0])
            .collect();
        assert_eq!(relabeled.len(), 3);

        // The transition that achieved [50.0] should get reward 1.0.
        let matching = relabeled.iter().filter(|t| t.reward == 1.0).count();
        // Steps 1 and 2 both achieved [50.0], so 2 should have reward 1.0.
        assert_eq!(matching, 2);
    }

    // ── Multiple episodes accumulate correctly ──────────────────────────

    #[test]
    fn test_multiple_episodes() {
        let mut buf = HerReplayBuffer::new(1000, HerStrategy::Final, sparse_reward);
        let mut rng = Rng::new(123);

        // Episode 1
        for i in 0..4 {
            buf.push(make_transition(i));
        }
        buf.end_episode(&mut rng);
        let after_first = buf.len();
        assert_eq!(after_first, 8); // 4 original + 4 relabeled

        // Episode 2
        for i in 10..15 {
            buf.push(make_transition(i));
        }
        buf.end_episode(&mut rng);
        // 8 from first + 5 original + 5 relabeled = 18
        assert_eq!(buf.len(), 18);
    }

    // ── Single-transition episode ───────────────────────────────────────

    #[test]
    fn test_single_transition_episode() {
        let mut buf = HerReplayBuffer::new(100, HerStrategy::Future(4), sparse_reward);
        let mut rng = Rng::new(0);

        buf.push(make_transition(0));
        buf.end_episode(&mut rng);

        // 1 original, 0 relabeled (no future steps available).
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_single_transition_final() {
        let mut buf = HerReplayBuffer::new(100, HerStrategy::Final, sparse_reward);
        let mut rng = Rng::new(0);

        buf.push(make_transition(0));
        buf.end_episode(&mut rng);

        // 1 original + 1 relabeled = 2
        assert_eq!(buf.len(), 2);
    }
}
