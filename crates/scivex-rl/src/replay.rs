//! Experience replay buffer for off-policy RL algorithms.

use scivex_core::random::Rng;

/// A single batch of experience sampled from the replay buffer.
pub struct Batch {
    /// Batch of state vectors.
    pub states: Vec<Vec<f64>>,
    /// Batch of action indices.
    pub actions: Vec<usize>,
    /// Batch of rewards.
    pub rewards: Vec<f64>,
    /// Batch of next-state vectors.
    pub next_states: Vec<Vec<f64>>,
    /// Batch of done flags.
    pub dones: Vec<bool>,
}

/// A ring-buffer for storing and sampling transitions.
pub struct ReplayBuffer {
    /// Stored states.
    pub states: Vec<Vec<f64>>,
    /// Stored actions.
    pub actions: Vec<usize>,
    /// Stored rewards.
    pub rewards: Vec<f64>,
    /// Stored next states.
    pub next_states: Vec<Vec<f64>>,
    /// Stored done flags.
    pub dones: Vec<bool>,
    capacity: usize,
    position: usize,
    len: usize,
}

impl ReplayBuffer {
    /// Create a new replay buffer with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            next_states: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
            capacity,
            position: 0,
            len: 0,
        }
    }

    /// Add a transition to the buffer.
    pub fn push(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
    ) {
        if self.len < self.capacity {
            self.states.push(state);
            self.actions.push(action);
            self.rewards.push(reward);
            self.next_states.push(next_state);
            self.dones.push(done);
            self.len += 1;
        } else {
            self.states[self.position] = state;
            self.actions[self.position] = action;
            self.rewards[self.position] = reward;
            self.next_states[self.position] = next_state;
            self.dones[self.position] = done;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    /// Return the number of transitions currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Sample a random batch of transitions.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch_size` exceeds the number of stored
    /// transitions.
    pub fn sample(&self, batch_size: usize, rng: &mut Rng) -> crate::error::Result<Batch> {
        if batch_size > self.len {
            return Err(crate::error::RlError::InvalidParameter(format!(
                "batch_size ({batch_size}) exceeds buffer length ({})",
                self.len
            )));
        }

        let mut states = Vec::with_capacity(batch_size);
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut next_states = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let idx = (rng.next_f64() * self.len as f64) as usize;
            let idx = idx.min(self.len - 1);
            states.push(self.states[idx].clone());
            actions.push(self.actions[idx]);
            rewards.push(self.rewards[idx]);
            next_states.push(self.next_states[idx].clone());
            dones.push(self.dones[idx]);
        }

        Ok(Batch {
            states,
            actions,
            rewards,
            next_states,
            dones,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_push_and_len() {
        let mut buf = ReplayBuffer::new(10);
        assert!(buf.is_empty());
        buf.push(vec![1.0], 0, 1.0, vec![2.0], false);
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_replay_capacity_overflow() {
        let mut buf = ReplayBuffer::new(3);
        for i in 0..5 {
            buf.push(vec![f64::from(i)], 0, 0.0, vec![0.0], false);
        }
        assert_eq!(buf.len(), 3);
        // Oldest entries should be overwritten
        // Position 0 was overwritten by entry 3, position 1 by entry 4
        assert_eq!(buf.states[0], vec![3.0]);
        assert_eq!(buf.states[1], vec![4.0]);
        assert_eq!(buf.states[2], vec![2.0]);
    }

    #[test]
    fn test_replay_sample() {
        let mut buf = ReplayBuffer::new(100);
        for i in 0..50 {
            buf.push(vec![i as f64], i % 3, i as f64 * 0.1, vec![0.0], false);
        }
        let mut rng = Rng::new(42);
        let batch = buf.sample(10, &mut rng).unwrap();
        assert_eq!(batch.states.len(), 10);
        assert_eq!(batch.actions.len(), 10);
        assert_eq!(batch.rewards.len(), 10);
    }

    #[test]
    fn test_replay_sample_too_large() {
        let buf = ReplayBuffer::new(10);
        let mut rng = Rng::new(0);
        assert!(buf.sample(1, &mut rng).is_err());
    }
}
