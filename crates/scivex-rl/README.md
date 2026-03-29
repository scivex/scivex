# scivex-rl

Reinforcement learning for Scivex. Environments, agents, and training loops
built on top of scivex-nn for neural network function approximation.

## Highlights

- **Environments** — CartPole, MountainCar, GridWorld, cooperative multi-agent
- **Agents** — DQN, DoubleDQN, PPO, A2C with configurable hyperparameters
- **Replay buffers** — Uniform, Prioritized (PER), HER for experience replay
- **Multi-agent** — Cooperative navigation environment with shared rewards
- **Trait-based** — `Environment` and `Agent` traits for custom implementations
- **Neural net integration** — Uses scivex-nn layers and optimizers

## Usage

```rust
use scivex_rl::prelude::*;

let env = CartPole::new();
let mut agent = DQN::new(env.observation_dim(), env.action_dim());

for episode in 0..1000 {
    let mut obs = env.reset();
    loop {
        let action = agent.act(&obs);
        let (next_obs, reward, done) = env.step(action);
        agent.remember(obs, action, reward, next_obs, done);
        agent.train();
        obs = next_obs;
        if done { break; }
    }
}
```

## License

MIT
