//! Experience replay buffer for continuous-action off-policy RL algorithms (SAC, TD3).

use scivex_core::random::Rng;

use crate::error::{Result, RlError};

/// A single batch of experience with continuous actions.
pub struct ContinuousBatch {
    /// Batch of state vectors.
    pub states: Vec<Vec<f64>>,
    /// Batch of continuous action vectors.
    pub actions: Vec<Vec<f64>>,
    /// Batch of rewards.
    pub rewards: Vec<f64>,
    /// Batch of next-state vectors.
    pub next_states: Vec<Vec<f64>>,
    /// Batch of done flags.
    pub dones: Vec<bool>,
}

/// A ring-buffer for storing and sampling transitions with continuous actions.
pub struct ContinuousReplayBuffer {
    /// Stored states.
    pub states: Vec<Vec<f64>>,
    /// Stored continuous actions.
    pub actions: Vec<Vec<f64>>,
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

impl ContinuousReplayBuffer {
    /// Create a new continuous replay buffer with the given capacity.
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
        action: Vec<f64>,
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
    /// Returns an error if `batch_size` exceeds the number of stored transitions.
    pub fn sample(&self, batch_size: usize, rng: &mut Rng) -> Result<ContinuousBatch> {
        if batch_size > self.len {
            return Err(RlError::InvalidParameter(format!(
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
            actions.push(self.actions[idx].clone());
            rewards.push(self.rewards[idx]);
            next_states.push(self.next_states[idx].clone());
            dones.push(self.dones[idx]);
        }

        Ok(ContinuousBatch {
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
    fn test_continuous_replay_push_and_len() {
        let mut buf = ContinuousReplayBuffer::new(10);
        assert!(buf.is_empty());
        buf.push(vec![1.0], vec![0.5, -0.3], 1.0, vec![2.0], false);
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_continuous_replay_sample() {
        let mut buf = ContinuousReplayBuffer::new(100);
        for i in 0..50 {
            buf.push(
                vec![f64::from(i)],
                vec![f64::from(i) * 0.1],
                f64::from(i) * 0.01,
                vec![0.0],
                false,
            );
        }
        let mut rng = Rng::new(42);
        let batch = buf.sample(10, &mut rng).unwrap();
        assert_eq!(batch.states.len(), 10);
        assert_eq!(batch.actions.len(), 10);
    }
}
