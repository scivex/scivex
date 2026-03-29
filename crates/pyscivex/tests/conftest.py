"""Shared fixtures for pyscivex tests."""

import pytest


@pytest.fixture
def sample_data():
    """Sample numeric data for testing."""
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture
def matrix_2x2():
    """2x2 matrix data."""
    return [1.0, 2.0, 3.0, 4.0]
