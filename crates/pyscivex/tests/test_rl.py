"""Tests for pyscivex reinforcement learning — rl submodule."""

import pyscivex as sv


# ===========================================================================
# ENVIRONMENTS
# ===========================================================================


class TestCartPole:
    def test_create(self):
        env = sv.rl.CartPole()
        assert repr(env) == "CartPole()"

    def test_observation_shape(self):
        env = sv.rl.CartPole()
        shape = env.observation_shape()
        assert len(shape) > 0

    def test_action_count(self):
        env = sv.rl.CartPole()
        assert env.action_count() == 2

    def test_reset(self):
        env = sv.rl.CartPole()
        obs = env.reset()
        assert isinstance(obs, list)
        assert len(obs) == env.observation_shape()[0]

    def test_step(self):
        env = sv.rl.CartPole()
        env.reset()
        result = env.step(0)
        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert "truncated" in result
        assert isinstance(result["observation"], list)
        assert isinstance(result["reward"], float)
        assert isinstance(result["done"], bool)

    def test_episode_loop(self):
        env = sv.rl.CartPole()
        obs = env.reset()
        total_reward = 0.0
        for _ in range(10):
            result = env.step(0)
            total_reward += result["reward"]
            if result["done"] or result["truncated"]:
                break
        assert total_reward > 0.0

    def test_is_done(self):
        env = sv.rl.CartPole()
        env.reset()
        assert not env.is_done()


class TestMountainCar:
    def test_create(self):
        env = sv.rl.MountainCar()
        assert repr(env) == "MountainCar()"

    def test_observation_shape(self):
        env = sv.rl.MountainCar()
        shape = env.observation_shape()
        assert len(shape) > 0

    def test_action_count(self):
        env = sv.rl.MountainCar()
        assert env.action_count() == 3

    def test_reset(self):
        env = sv.rl.MountainCar()
        obs = env.reset()
        assert isinstance(obs, list)
        assert len(obs) == env.observation_shape()[0]

    def test_step(self):
        env = sv.rl.MountainCar()
        env.reset()
        result = env.step(1)
        assert "observation" in result
        assert "reward" in result
        assert "done" in result


class TestGridWorld:
    def test_create(self):
        env = sv.rl.GridWorld(5)
        assert "GridWorld" in repr(env)

    def test_default_size(self):
        env = sv.rl.GridWorld()
        assert env.action_count() == 4

    def test_observation_shape(self):
        env = sv.rl.GridWorld(4)
        shape = env.observation_shape()
        assert len(shape) > 0

    def test_reset(self):
        env = sv.rl.GridWorld(5)
        obs = env.reset()
        assert isinstance(obs, list)

    def test_step(self):
        env = sv.rl.GridWorld(5)
        env.reset()
        result = env.step(0)  # up
        assert "observation" in result
        assert "reward" in result
        assert "done" in result

    def test_episode(self):
        env = sv.rl.GridWorld(3)
        env.reset()
        for _ in range(50):
            result = env.step(1)  # right
            if result["done"]:
                break


# ===========================================================================
# DQN
# ===========================================================================


class TestDQN:
    def test_create(self):
        agent = sv.rl.DQN(4, 2)
        assert repr(agent) == "DQN()"

    def test_create_custom(self):
        agent = sv.rl.DQN(
            4, 2,
            learning_rate=0.01,
            gamma=0.95,
            epsilon=0.5,
            batch_size=16,
        )
        assert repr(agent) == "DQN()"

    def test_act(self):
        agent = sv.rl.DQN(4, 2)
        action = agent.act([0.0, 0.0, 0.0, 0.0])
        assert action in (0, 1)

    def test_train_cartpole(self):
        agent = sv.rl.DQN(4, 2, seed=123)
        result = agent.train_cartpole(5)
        assert "episode_rewards" in result
        assert "episode_lengths" in result
        assert "total_episodes" in result
        assert result["total_episodes"] == 5

    def test_train_gridworld(self):
        agent = sv.rl.DQN(1, 4, seed=42)
        result = agent.train_gridworld(3, 5)
        assert result["total_episodes"] == 5


# ===========================================================================
# PPO
# ===========================================================================


class TestPPO:
    def test_create(self):
        agent = sv.rl.PPO(4, 2)
        assert repr(agent) == "PPO()"

    def test_create_custom(self):
        agent = sv.rl.PPO(
            4, 2,
            learning_rate=0.001,
            gamma=0.95,
            clip_epsilon=0.1,
        )
        assert repr(agent) == "PPO()"

    def test_act(self):
        agent = sv.rl.PPO(4, 2)
        action, log_prob, value = agent.act([0.0, 0.0, 0.0, 0.0])
        assert action in (0, 1)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_train_cartpole(self):
        agent = sv.rl.PPO(4, 2, n_steps=32, seed=42)
        result = agent.train_cartpole(100)
        assert "episode_rewards" in result
        assert result["total_episodes"] > 0


# ===========================================================================
# A2C
# ===========================================================================


class TestA2C:
    def test_create(self):
        agent = sv.rl.A2C(4, 2)
        assert repr(agent) == "A2C()"

    def test_act(self):
        agent = sv.rl.A2C(4, 2)
        action, log_prob, value = agent.act([0.0, 0.0, 0.0, 0.0])
        assert action in (0, 1)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_train_cartpole(self):
        agent = sv.rl.A2C(4, 2, n_steps=16, seed=42)
        result = agent.train_cartpole(100)
        assert "episode_rewards" in result
        assert result["total_episodes"] > 0


# ===========================================================================
# SAC
# ===========================================================================


class TestSAC:
    def test_create(self):
        agent = sv.rl.SAC(4, 2)
        assert repr(agent) == "SAC()"

    def test_create_custom(self):
        agent = sv.rl.SAC(
            4, 2,
            learning_rate=0.001,
            gamma=0.95,
            tau=0.01,
            alpha=0.1,
        )
        assert repr(agent) == "SAC()"

    def test_select_action(self):
        agent = sv.rl.SAC(4, 2)
        action = agent.select_action([0.0, 0.0, 0.0, 0.0])
        assert isinstance(action, list)
        assert len(action) == 2

    def test_dims(self):
        agent = sv.rl.SAC(4, 2)
        assert agent.state_dim() == 4
        assert agent.action_dim() == 2


# ===========================================================================
# TD3
# ===========================================================================


class TestTD3:
    def test_create(self):
        agent = sv.rl.TD3(4, 2)
        assert repr(agent) == "TD3()"

    def test_create_custom(self):
        agent = sv.rl.TD3(
            4, 2,
            learning_rate=0.001,
            policy_noise=0.1,
            noise_clip=0.3,
            policy_delay=3,
        )
        assert repr(agent) == "TD3()"

    def test_select_action(self):
        agent = sv.rl.TD3(4, 2)
        action = agent.select_action([0.0, 0.0, 0.0, 0.0])
        assert isinstance(action, list)
        assert len(action) == 2

    def test_select_action_custom_noise(self):
        agent = sv.rl.TD3(4, 2)
        action = agent.select_action([0.0, 0.0, 0.0, 0.0], exploration_noise=0.5)
        assert isinstance(action, list)
        assert len(action) == 2

    def test_dims(self):
        agent = sv.rl.TD3(4, 2)
        assert agent.state_dim() == 4
        assert agent.action_dim() == 2


# ===========================================================================
# EpisodeLogger
# ===========================================================================


class TestEpisodeLogger:
    def test_create(self):
        logger = sv.rl.EpisodeLogger()
        assert "EpisodeLogger" in repr(logger)

    def test_log_and_end(self):
        logger = sv.rl.EpisodeLogger()
        logger.log_step(1.0)
        logger.log_step(2.0)
        logger.log_step(3.0)
        logger.end_episode()
        assert logger.total_episodes() == 1
        assert len(logger.episode_rewards) == 1
        assert abs(logger.episode_rewards[0] - 6.0) < 1e-10
        assert logger.episode_lengths[0] == 3

    def test_mean_reward(self):
        logger = sv.rl.EpisodeLogger()
        for _ in range(5):
            logger.log_step(10.0)
            logger.end_episode()
        assert abs(logger.mean_reward(5) - 10.0) < 1e-10

    def test_multiple_episodes(self):
        logger = sv.rl.EpisodeLogger()
        # Episode 1: reward = 5
        for _ in range(5):
            logger.log_step(1.0)
        logger.end_episode()
        # Episode 2: reward = 20
        for _ in range(4):
            logger.log_step(5.0)
        logger.end_episode()
        assert logger.total_episodes() == 2
        assert abs(logger.episode_rewards[0] - 5.0) < 1e-10
        assert abs(logger.episode_rewards[1] - 20.0) < 1e-10


# ===========================================================================
# INTEGRATION (all accessible)
# ===========================================================================


class TestIntegrationAccessible:
    def test_all_accessible(self):
        items = [
            # Environments
            sv.rl.CartPole,
            sv.rl.MountainCar,
            sv.rl.GridWorld,
            # Algorithms
            sv.rl.DQN,
            sv.rl.PPO,
            sv.rl.A2C,
            sv.rl.SAC,
            sv.rl.TD3,
            # Utilities
            sv.rl.EpisodeLogger,
        ]
        for item in items:
            assert item is not None
