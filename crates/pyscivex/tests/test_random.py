"""Tests for pyscivex.random submodule."""

import pyscivex as sv


class TestRng:
    def test_create(self):
        rng = sv.random.Rng(42)
        assert isinstance(rng, sv.random.Rng)

    def test_next_f64(self):
        rng = sv.random.Rng(42)
        val = rng.next_f64()
        assert 0.0 <= val < 1.0

    def test_deterministic(self):
        rng1 = sv.random.Rng(123)
        rng2 = sv.random.Rng(123)
        assert rng1.next_f64() == rng2.next_f64()

    def test_reseed(self):
        rng = sv.random.Rng(42)
        rng.seed(99)
        rng2 = sv.random.Rng(99)
        assert rng.next_f64() == rng2.next_f64()


class TestRngTensors:
    def test_uniform(self):
        rng = sv.random.Rng(42)
        t = rng.uniform([3, 4])
        assert t.shape() == [3, 4]
        assert t.min() >= 0.0
        assert t.max() < 1.0

    def test_uniform_range(self):
        rng = sv.random.Rng(42)
        t = rng.uniform_range([100], 5.0, 10.0)
        assert t.shape() == [100]
        assert t.min() >= 5.0
        assert t.max() < 10.0

    def test_normal(self):
        rng = sv.random.Rng(42)
        t = rng.normal([1000], 0.0, 1.0)
        assert t.shape() == [1000]
        # Mean should be roughly 0
        assert abs(t.mean()) < 0.2

    def test_standard_normal(self):
        rng = sv.random.Rng(42)
        t = rng.standard_normal([500])
        assert t.shape() == [500]

    def test_randint(self):
        rng = sv.random.Rng(42)
        t = rng.randint([100], 0, 10)
        assert t.shape() == [100]
        assert t.min() >= 0.0
        assert t.max() < 10.0
