"""Shared fixtures.

The synthetic datasets here are small and module-scoped: generation is cheap but
not free, and every test that needs "a valid book" should get the same one so a
failure points at the code under test rather than at a different random draw.
"""

from __future__ import annotations

import pytest

from trading_research.data import synthetic


@pytest.fixture(scope="session")
def config() -> synthetic.SyntheticConfig:
    """A small, fast configuration exercising every part of the generator."""
    return synthetic.SyntheticConfig(n_steps=3_000, depth=5, seed=7)


@pytest.fixture(scope="session")
def dataset(config: synthetic.SyntheticConfig) -> synthetic.SyntheticDataset:
    return synthetic.generate(config)


@pytest.fixture(scope="session")
def book(dataset: synthetic.SyntheticDataset):
    return dataset.book


@pytest.fixture(scope="session")
def trades(dataset: synthetic.SyntheticDataset):
    return dataset.trades
