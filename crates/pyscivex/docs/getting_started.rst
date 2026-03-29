Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install pyscivex

Quick example
-------------

.. code-block:: python

   import pyscivex as sv

   # Tensors
   a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
   b = sv.Tensor.ones([2, 2])
   c = a + b

   # DataFrames
   df = sv.DataFrame()
   df.add_column("x", [1.0, 2.0, 3.0])
   df.add_column("y", [4.0, 5.0, 6.0])

   # Statistics
   print(sv.mean([1.0, 2.0, 3.0, 4.0]))

   # ML
   model = sv.LinearRegression()

   # Linear algebra
   lu = sv.linalg.LU.decompose(a)

Interop with numpy / pandas
----------------------------

pyscivex ships optional compatibility helpers for migrating existing code:

.. code-block:: python

   from pyscivex.compat import from_numpy, tensor_to_numpy
   from pyscivex.compat import from_pandas, to_pandas

   import numpy as np
   arr = np.array([1.0, 2.0, 3.0])
   t = from_numpy(arr)          # numpy -> pyscivex Tensor
   arr2 = tensor_to_numpy(t)    # pyscivex Tensor -> numpy

   import pandas as pd
   pdf = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
   sv_df = from_pandas(pdf)     # pandas -> pyscivex DataFrame
   pdf2 = to_pandas(sv_df)      # pyscivex DataFrame -> pandas

scikit-learn compatibility
--------------------------

Wrap any pyscivex ML model so it works with scikit-learn utilities:

.. code-block:: python

   from pyscivex import LinearRegression
   from pyscivex.compat import SklearnWrapper
   from sklearn.model_selection import cross_val_score

   est = SklearnWrapper(LinearRegression())
   scores = cross_val_score(est, X, y, cv=5)

Gymnasium compatibility
-----------------------

Use standard Gymnasium environments with pyscivex RL agents:

.. code-block:: python

   import gymnasium as gym
   from pyscivex.compat import GymnasiumEnvWrapper

   env = GymnasiumEnvWrapper(gym.make("CartPole-v1"))
   obs, info = env.reset()
   obs, reward, terminated, truncated, info = env.step(0)
