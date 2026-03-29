"""
pyscivex.compat — Optional compatibility / interop layers.

These helpers convert between pyscivex objects and popular Python
data-science libraries (numpy, pandas, scikit-learn, gymnasium).
Each function lazily imports the external library and raises a
clear ``ImportError`` if it is not installed.

Usage::

    from pyscivex.compat import from_numpy, tensor_to_numpy
    from pyscivex.compat import from_pandas, to_pandas
    from pyscivex.compat import SklearnWrapper
    from pyscivex.compat import GymnasiumEnvWrapper
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = [
    # numpy
    "from_numpy",
    "tensor_to_numpy",
    # pandas
    "from_pandas",
    "to_pandas",
    # scikit-learn
    "SklearnWrapper",
    # gymnasium
    "GymnasiumEnvWrapper",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(package: str):
    """Import *package* or raise a friendly ``ImportError``."""
    import importlib

    try:
        return importlib.import_module(package)
    except ModuleNotFoundError:
        raise ImportError(
            f"The '{package}' package is required for this function but is "
            f"not installed.  Install it with:  pip install {package}"
        ) from None


# ---------------------------------------------------------------------------
# P2.15  Numpy compatibility
# ---------------------------------------------------------------------------

def from_numpy(arr) -> "Tensor":
    """Convert a numpy ``ndarray`` to a pyscivex :class:`Tensor`.

    Parameters
    ----------
    arr : numpy.ndarray
        Must be a numeric dtype that can be cast to ``float64``.

    Returns
    -------
    Tensor
    """
    np = _require("numpy")

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr).__name__}")

    from pyscivex import Tensor

    arr = arr.astype(float, copy=False)
    data = arr.flatten().tolist()
    shape = list(arr.shape)
    return Tensor(data, shape)


def tensor_to_numpy(tensor):
    """Convert a pyscivex :class:`Tensor` to a numpy ``ndarray``.

    Parameters
    ----------
    tensor : pyscivex.Tensor

    Returns
    -------
    numpy.ndarray
    """
    np = _require("numpy")
    return np.array(tensor.to_list()).reshape(tensor.shape())


# ---------------------------------------------------------------------------
# P3.24  Pandas compatibility
# ---------------------------------------------------------------------------

def from_pandas(df) -> "DataFrame":
    """Convert a pandas ``DataFrame`` to a pyscivex :class:`DataFrame`.

    Each column is added individually.  Only columns whose dtype can be
    coerced to ``float64`` or ``str`` are supported.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pyscivex.DataFrame
    """
    pd = _require("pandas")

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df).__name__}")

    from pyscivex import DataFrame as SvDataFrame

    sv_df = SvDataFrame()
    for col in df.columns:
        series = df[col]
        if series.dtype.kind in ("f", "i", "u"):
            sv_df.add_column(str(col), series.astype(float).tolist())
        else:
            sv_df.add_column(str(col), series.astype(str).tolist())

    return sv_df


def to_pandas(sv_df):
    """Convert a pyscivex :class:`DataFrame` to a pandas ``DataFrame``.

    Parameters
    ----------
    sv_df : pyscivex.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    pd = _require("pandas")

    columns = sv_df.column_names()
    data: Dict[str, list] = {}
    for col in columns:
        data[col] = sv_df.column(col).to_list()

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# P5.20  scikit-learn compatibility
# ---------------------------------------------------------------------------

class SklearnWrapper:
    """Wrap a pyscivex ML model so it passes scikit-learn estimator checks.

    The wrapper exposes the standard ``fit`` / ``predict`` /
    ``get_params`` / ``set_params`` interface expected by scikit-learn
    utilities (e.g. ``cross_val_score``, ``GridSearchCV``).

    Parameters
    ----------
    model : object
        A pyscivex model instance that implements ``.fit()`` and
        ``.predict()`` (e.g. ``LinearRegression``, ``KMeans``).
    **params
        Initial hyper-parameters forwarded to the model.

    Example
    -------
    ::

        from pyscivex import LinearRegression
        from pyscivex.compat import SklearnWrapper
        from sklearn.model_selection import cross_val_score

        est = SklearnWrapper(LinearRegression())
        scores = cross_val_score(est, X, y, cv=5)
    """

    def __init__(self, model: Any, **params: Any) -> None:
        _require("sklearn")
        self.model = model
        self._params: Dict[str, Any] = dict(params)

    # -- scikit-learn estimator interface ------------------------------------

    def fit(self, X, y=None):
        """Fit the underlying pyscivex model.

        *X* and *y* are converted from numpy arrays to pyscivex Tensors
        automatically.
        """
        X_sv = from_numpy(X) if _is_numpy(X) else X
        if y is not None:
            y_sv = from_numpy(y) if _is_numpy(y) else y
            self.model.fit(X_sv, y_sv)
        else:
            self.model.fit(X_sv)
        return self

    def predict(self, X):
        """Predict using the underlying pyscivex model and return numpy."""
        X_sv = from_numpy(X) if _is_numpy(X) else X
        result = self.model.predict(X_sv)
        try:
            return tensor_to_numpy(result)
        except Exception:
            return result

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self._params)

    def set_params(self, **params: Any) -> "SklearnWrapper":
        self._params.update(params)
        return self


def _is_numpy(obj) -> bool:
    """Return ``True`` if *obj* is a numpy array (without importing numpy)."""
    return type(obj).__module__ == "numpy" and type(obj).__name__ == "ndarray"


# ---------------------------------------------------------------------------
# P14.9  Gymnasium compatibility
# ---------------------------------------------------------------------------

class GymnasiumEnvWrapper:
    """Wrap a ``gymnasium.Env`` for use with pyscivex RL agents.

    This thin adapter converts gymnasium observations and actions between
    numpy arrays and pyscivex Tensors so that pyscivex RL agents can
    interact with standard Gymnasium environments without manual
    conversion.

    Parameters
    ----------
    env : gymnasium.Env
        A Gymnasium environment instance.

    Example
    -------
    ::

        import gymnasium as gym
        from pyscivex.compat import GymnasiumEnvWrapper

        env = GymnasiumEnvWrapper(gym.make("CartPole-v1"))
        obs = env.reset()  # returns pyscivex Tensor
        obs, reward, terminated, truncated, info = env.step(0)
    """

    def __init__(self, env) -> None:
        _require("gymnasium")
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def _obs_to_tensor(self, obs):
        """Best-effort conversion of an observation to Tensor."""
        try:
            return from_numpy(obs)
        except Exception:
            return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment and return the initial observation as Tensor."""
        result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple):
            obs, info = result
            return self._obs_to_tensor(obs), info
        return self._obs_to_tensor(result)

    def step(self, action):
        """Take a step.  ``action`` may be an int, float, or Tensor."""
        # Convert Tensor action back to numpy if needed
        act = action
        try:
            act = tensor_to_numpy(action)
        except Exception:
            pass

        obs, reward, terminated, truncated, info = self.env.step(act)
        return self._obs_to_tensor(obs), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def __getattr__(self, name: str):
        """Proxy unknown attributes to the underlying environment."""
        return getattr(self.env, name)
