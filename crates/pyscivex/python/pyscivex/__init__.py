"""
pyscivex — The complete data science toolkit, powered by Rust.

One ``pip install pyscivex`` replaces numpy, pandas, scipy, scikit-learn,
matplotlib, and more — with Rust performance and zero external dependencies.

Quick start::

    import pyscivex as sv

    # Tensors
    a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = sv.Tensor.ones([2, 2])
    c = a + b

    # DataFrames
    df = sv.DataFrame()
    df.add_column("x", [1.0, 2.0, 3.0])

    # Statistics
    print(sv.mean([1.0, 2.0, 3.0, 4.0]))

    # ML
    model = sv.LinearRegression()

    # Linear algebra
    lu = sv.linalg.LU.decompose(a)

    # Random
    rng = sv.random.Rng(42)
    r = rng.uniform([3, 3])
"""

from pyscivex._native import (
    # Core tensor
    Tensor,
    # DataFrame
    DataFrame,
    LazyFrame,
    Series,
    # Stats functions
    mean,
    variance,
    std_dev,
    median,
    pearson,
    # ML models
    LinearRegression,
    KMeans,
    # Visualization
    Figure,
    # Submodules
    linalg,
    fft,
    random,
    io,
    stats,
    ml,
    optim,
    nn,
    signal,
    image,
    nlp,
    graph,
    sym,
    rl,
    gpu,
    viz,
)

__all__ = [
    # Core
    "Tensor",
    # DataFrame
    "DataFrame",
    "LazyFrame",
    "Series",
    # Stats
    "mean",
    "variance",
    "std_dev",
    "median",
    "pearson",
    # ML
    "LinearRegression",
    "KMeans",
    # Viz
    "Figure",
    # Submodules
    "linalg",
    "fft",
    "random",
    "io",
    "stats",
    "ml",
    "optim",
    "nn",
    "signal",
    "image",
    "nlp",
    "graph",
    "sym",
    "rl",
    "gpu",
    "viz",
    # Compat helpers
    "compat",
]

__version__ = "0.1.0"
