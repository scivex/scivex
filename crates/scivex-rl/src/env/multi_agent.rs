//! Multi-agent reinforcement learning environments.

use scivex_core::random::Rng;

/// A multi-agent reinforcement learning environment.
pub trait MultiAgentEnv {
    /// Reset the environment and return initial observations for all agents.
    fn reset(&mut self) -> Vec<Vec<f64>>;

    /// Take actions for all agents and return (observations, rewards, dones, all_done).
    fn step(&mut self, actions: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<bool>, bool);

    /// Return the number of agents.
    fn num_agents(&self) -> usize;

    /// Return the observation dimension per agent.
    fn observation_dim(&self) -> usize;

    /// Return the action dimension per agent.
    fn action_dim(&self) -> usize;
}

/// Cooperative navigation: N agents must navigate to N landmarks.
///
/// Reward is the negative sum of minimum distances from each landmark to
/// the nearest agent, encouraging cooperation.
///
/// # Examples
///
/// ```
/// use scivex_rl::env::multi_agent::{CooperativeNavigation, MultiAgentEnv};
/// let mut env = CooperativeNavigation::new(2, 5.0, 42);
/// let obs = env.reset();
/// assert_eq!(obs.len(), 2);
/// let actions = vec![vec![0.1, 0.0], vec![0.0, -0.1]];
/// let (next_obs, rewards, dones, all_done) = env.step(&actions);
/// assert_eq!(next_obs.len(), 2);
/// assert_eq!(rewards.len(), 2);
/// ```
pub struct CooperativeNavigation {
    num_agents: usize,
    arena_size: f64,
    max_steps: usize,
    current_step: usize,
    /// Agent positions: [agent_idx] -> (x, y)
    positions: Vec<(f64, f64)>,
    /// Agent velocities: [agent_idx] -> (vx, vy)
    velocities: Vec<(f64, f64)>,
    /// Landmark positions: [landmark_idx] -> (x, y)
    landmarks: Vec<(f64, f64)>,
    rng: Rng,
}

impl CooperativeNavigation {
    /// Create a new cooperative navigation environment.
    ///
    /// - `num_agents`: number of agents and landmarks
    /// - `arena_size`: half-width of the arena (positions in [-arena, arena])
    /// - `seed`: random seed
    #[must_use]
    pub fn new(num_agents: usize, arena_size: f64, seed: u64) -> Self {
        let mut env = Self {
            num_agents,
            arena_size,
            max_steps: 100,
            current_step: 0,
            positions: vec![(0.0, 0.0); num_agents],
            velocities: vec![(0.0, 0.0); num_agents],
            landmarks: vec![(0.0, 0.0); num_agents],
            rng: Rng::new(seed),
        };
        let _ = env.reset();
        env
    }

    /// Set the maximum number of steps per episode.
    pub fn set_max_steps(&mut self, max_steps: usize) -> &mut Self {
        self.max_steps = max_steps;
        self
    }

    fn random_pos(&mut self) -> (f64, f64) {
        let x = (self.rng.next_f64() * 2.0 - 1.0) * self.arena_size;
        let y = (self.rng.next_f64() * 2.0 - 1.0) * self.arena_size;
        (x, y)
    }

    fn compute_reward(&self) -> f64 {
        // Negative sum of minimum distances from each landmark to nearest agent
        let mut total = 0.0;
        for lm in &self.landmarks {
            let min_dist = self
                .positions
                .iter()
                .map(|p| ((p.0 - lm.0).powi(2) + (p.1 - lm.1).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min);
            total += min_dist;
        }
        -total
    }

    fn get_observations(&self) -> Vec<Vec<f64>> {
        // Each agent observes: own position, own velocity, all landmark positions,
        // relative positions of other agents
        let mut obs = Vec::with_capacity(self.num_agents);
        for i in 0..self.num_agents {
            let mut o = vec![
                self.positions[i].0,
                self.positions[i].1,
                self.velocities[i].0,
                self.velocities[i].1,
            ];
            // Landmark positions relative to self
            for lm in &self.landmarks {
                o.push(lm.0 - self.positions[i].0);
                o.push(lm.1 - self.positions[i].1);
            }
            // Other agents relative to self
            for (j, pos) in self.positions.iter().enumerate() {
                if j != i {
                    o.push(pos.0 - self.positions[i].0);
                    o.push(pos.1 - self.positions[i].1);
                }
            }
            obs.push(o);
        }
        obs
    }
}

impl MultiAgentEnv for CooperativeNavigation {
    fn reset(&mut self) -> Vec<Vec<f64>> {
        self.current_step = 0;
        for i in 0..self.num_agents {
            self.positions[i] = self.random_pos();
            self.velocities[i] = (0.0, 0.0);
            self.landmarks[i] = self.random_pos();
        }
        self.get_observations()
    }

    fn step(&mut self, actions: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<bool>, bool) {
        let n = self.num_agents;
        // Apply actions as forces (clipped)
        for i in 0..n {
            if i < actions.len() && actions[i].len() >= 2 {
                let fx = actions[i][0].clamp(-1.0, 1.0);
                let fy = actions[i][1].clamp(-1.0, 1.0);
                self.velocities[i].0 = (self.velocities[i].0 + fx) * 0.9; // damping
                self.velocities[i].1 = (self.velocities[i].1 + fy) * 0.9;
            }
            // Update position
            self.positions[i].0 = (self.positions[i].0 + self.velocities[i].0 * 0.1)
                .clamp(-self.arena_size, self.arena_size);
            self.positions[i].1 = (self.positions[i].1 + self.velocities[i].1 * 0.1)
                .clamp(-self.arena_size, self.arena_size);
        }

        self.current_step += 1;
        let reward = self.compute_reward();
        let all_done = self.current_step >= self.max_steps;
        let rewards = vec![reward; n];
        let dones = vec![all_done; n];
        let obs = self.get_observations();

        (obs, rewards, dones, all_done)
    }

    fn num_agents(&self) -> usize {
        self.num_agents
    }

    fn observation_dim(&self) -> usize {
        // 4 (self) + 2*n (landmarks) + 2*(n-1) (other agents)
        4 + 2 * self.num_agents + 2 * (self.num_agents - 1)
    }

    fn action_dim(&self) -> usize {
        2 // force in x and y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooperative_navigation_basic() {
        let mut env = CooperativeNavigation::new(3, 5.0, 42);
        let obs = env.reset();
        assert_eq!(obs.len(), 3);
        assert_eq!(obs[0].len(), env.observation_dim());
    }

    #[test]
    fn test_cooperative_navigation_step() {
        let mut env = CooperativeNavigation::new(2, 5.0, 42);
        env.reset();
        let actions = vec![vec![0.5, 0.0], vec![0.0, -0.5]];
        let (obs, rewards, dones, all_done) = env.step(&actions);
        assert_eq!(obs.len(), 2);
        assert_eq!(rewards.len(), 2);
        assert_eq!(dones.len(), 2);
        assert!(!all_done);
        // Reward should be negative (distances are positive)
        assert!(rewards[0] < 0.0);
    }

    #[test]
    fn test_cooperative_navigation_terminates() {
        let mut env = CooperativeNavigation::new(2, 5.0, 42);
        env.set_max_steps(5);
        env.reset();
        let actions = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let mut all_done = false;
        for _ in 0..10 {
            let result = env.step(&actions);
            all_done = result.3;
            if all_done {
                break;
            }
        }
        assert!(all_done);
    }
}
