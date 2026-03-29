# pyscivex — Complete Python Use Case Guide

> **One package. Every data science workflow. Rust-powered Python.**
>
> This document walks through every capability of `pyscivex` — from basic tensor
> manipulation to training neural networks, building ML pipelines, and deploying
> models. Each section includes working Python code examples showing how pyscivex
> replaces numpy, pandas, scipy, scikit-learn, matplotlib, pytorch, and 15+ other
> packages with a single `pip install pyscivex`.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Tensor Operations (NumPy Replacement)](#2-tensor-operations)
3. [DataFrames (Pandas/Polars Replacement)](#3-dataframes)
4. [Data I/O — Files, Databases, Cloud](#4-data-io)
5. [Statistics & Distributions (SciPy.stats Replacement)](#5-statistics--distributions)
6. [Time Series Analysis (statsmodels Replacement)](#6-time-series-analysis)
7. [Bayesian Inference & MCMC (PyMC Replacement)](#7-bayesian-inference--mcmc)
8. [Optimization & Solvers (SciPy.optimize Replacement)](#8-optimization--solvers)
9. [Signal Processing & Audio (SciPy.signal + librosa Replacement)](#9-signal-processing--audio)
10. [Image Processing & Computer Vision (OpenCV + Pillow Replacement)](#10-image-processing--computer-vision)
11. [Natural Language Processing (NLTK + spaCy Replacement)](#11-natural-language-processing)
12. [Graph & Network Analysis (NetworkX Replacement)](#12-graph--network-analysis)
13. [Symbolic Mathematics (SymPy Replacement)](#13-symbolic-mathematics)
14. [Classical Machine Learning (scikit-learn Replacement)](#14-classical-machine-learning)
15. [Neural Networks & Deep Learning (PyTorch Replacement)](#15-neural-networks--deep-learning)
16. [Reinforcement Learning (Stable-Baselines3 Replacement)](#16-reinforcement-learning)
17. [GPU Acceleration (CuPy Replacement)](#17-gpu-acceleration)
18. [Visualization & Plotting (Matplotlib + Seaborn Replacement)](#18-visualization--plotting)
19. [ML Pipelines & AutoML](#19-ml-pipelines--automl)
20. [Model Deployment & Interop](#20-model-deployment--interop)
21. [Jupyter Notebook Integration](#21-jupyter-notebook-integration)
22. [Python Ecosystem Interop](#22-python-ecosystem-interop)
23. [What pyscivex Does NOT Do](#23-what-pyscivex-does-not-do)

---

## Python ↔ Library Mapping

Before diving in, here's what pyscivex replaces:

| You currently use | pyscivex equivalent | Import |
|---|---|---|
| `numpy` | Tensor, linalg, fft, random | `import pyscivex as sv` |
| `pandas` / `polars` | DataFrame, Series, GroupBy | `sv.DataFrame(...)` |
| `scipy.stats` | stats module | `sv.stats` |
| `scipy.optimize` | optim module | `sv.optim` |
| `scipy.signal` | signal module | `sv.signal` |
| `scikit-learn` | ml module | `sv.ml` |
| `pytorch` / `tensorflow` | nn module | `sv.nn` |
| `matplotlib` / `seaborn` | viz module, Figure, Chart | `sv.Figure()` / `sv.Chart()` |
| `pillow` / `opencv` | image module | `sv.image` |
| `nltk` / `spacy` / `gensim` | nlp module | `sv.nlp` |
| `networkx` | graph module | `sv.Graph()` |
| `sympy` | sym module | `sv.sym` |
| `stable-baselines3` | rl module | `sv.rl` |
| `cupy` | gpu module | `sv.gpu` |

**Total: 15+ Python packages → 1 `pip install pyscivex`**

---

## 1. Getting Started

### Installation

```bash
# From PyPI (precompiled wheels for Linux, macOS, Windows)
pip install pyscivex

# With NumPy interop (recommended)
pip install pyscivex[numpy]

# Full install with all optional dependencies
pip install pyscivex[full]

# Development install from source
git clone https://github.com/scivex/scivex.git
cd scivex/crates/pyscivex
pip install maturin
maturin develop --release
```

### Hello World

```python
import pyscivex as sv

# Create a 2x3 tensor
a = sv.Tensor([[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]])
print(f"Shape: {a.shape}")    # (2, 3)
print(f"Sum: {a.sum()}")      # 21.0
print(f"Mean: {a.mean()}")    # 3.5

# Quick stats
data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
print(f"mean={sv.stats.mean(data):.2f}")       # 5.00
print(f"std={sv.stats.std_dev(data):.2f}")     # 2.00
print(f"median={sv.stats.median(data):.2f}")   # 4.50

# One-line ML
model = sv.ml.LinearRegression()
X = sv.Tensor([[1.0], [2.0], [3.0], [4.0]])
y = sv.Tensor([2.1, 3.9, 6.1, 8.0])
model.fit(X, y)
print(model.predict(sv.Tensor([[5.0]])))  # ~10.0
```

### Import Patterns

```python
# Full import (recommended)
import pyscivex as sv

# Submodule imports
from pyscivex import Tensor, DataFrame
from pyscivex.stats import Normal, ttest_ind
from pyscivex.ml import RandomForestClassifier, Pipeline
from pyscivex.nn import Sequential, Linear, Adam
from pyscivex.optim import minimize, curve_fit
from pyscivex.signal import stft, mfcc
from pyscivex.image import Image
from pyscivex.graph import Graph, DiGraph
from pyscivex.nlp import WordTokenizer, TfidfVectorizer
from pyscivex.sym import var, diff, integrate
from pyscivex.rl import DQN, CartPole
```

---

## 2. Tensor Operations

> **Replaces:** numpy, numpy.linalg, numpy.fft, numpy.random

### Creating Tensors

```python
import pyscivex as sv

# From Python lists
a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])          # 2x2
b = sv.Tensor([1.0, 2.0, 3.0])                    # 1-D

# From flat data + shape
c = sv.Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))  # 2x3

# Factory constructors
zeros = sv.Tensor.zeros((3, 4))                    # 3x4 zeros
ones  = sv.Tensor.ones((3, 4))                     # 3x4 ones
eye   = sv.Tensor.eye(3)                           # 3x3 identity
rng   = sv.Tensor.arange(10)                       # [0, 1, ..., 9]
lin   = sv.Tensor.linspace(0.0, 1.0, 100)         # 100 points in [0, 1]
full  = sv.Tensor.full((2, 3), 42.0)              # all 42s

# Random tensors
sv.random.seed(42)
u = sv.random.uniform((3, 3))                      # uniform [0, 1)
n = sv.random.normal((3, 3), mean=0.0, std=1.0)   # standard normal
r = sv.random.randint((5,), low=0, high=10)        # random integers

# Dtype support
f32 = sv.Tensor([1.0, 2.0], dtype="float32")
f64 = sv.Tensor([1.0, 2.0], dtype="float64")       # default
i64 = sv.Tensor([1, 2, 3], dtype="int64")
```

### Element-wise Arithmetic

```python
a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
b = sv.Tensor([[5.0, 6.0], [7.0, 8.0]])

c = a + b         # element-wise add
d = a * b         # element-wise multiply
e = a - b         # subtract
f = a / b         # divide
g = -a            # negate
h = a + 2.0       # scalar broadcast
i = a ** 2        # power

# In-place operations
a += b
a *= 2.0
```

### Element-wise Math Functions

```python
t = sv.Tensor([0.5, 1.0, 1.5, 2.0])

sv.abs(t)         # absolute value
sv.sqrt(t)        # square root
sv.sin(t)         # sine
sv.cos(t)         # cosine
sv.exp(t)         # exponential
sv.log(t)         # natural log
sv.floor(t)       # floor
sv.ceil(t)        # ceiling
sv.round(t)       # round
t.clamp(0.0, 1.0) # clamp to range
```

### Reductions

```python
a = sv.Tensor([[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]])

a.sum()                 # 21.0 — sum all
a.sum(axis=0)           # [5.0, 7.0, 9.0] — sum along rows
a.sum(axis=1)           # [6.0, 15.0] — sum along columns
a.mean()                # 3.5
a.min()                 # 1.0
a.max()                 # 6.0
a.prod()                # 720.0
a.argmin()              # index of minimum
a.argmax()              # index of maximum
a.std()                 # standard deviation
a.var()                 # variance
```

### Reshaping and Indexing

```python
a = sv.Tensor.arange(12).reshape((3, 4))

# Reshaping
flat = a.flatten()                   # 1-D [0, 1, ..., 11]
r = a.reshape((4, 3))               # 4x3
t = a.transpose()                   # 4x3
p = a.permute((1, 0))               # axis permutation

# Indexing (NumPy-compatible)
a[0]                                 # row 0 → [0, 1, 2, 3]
a[0, 1]                             # scalar → 1.0
a[:, 1]                             # column 1 → [1, 5, 9]
a[1:3, 0:2]                         # submatrix → [[4,5],[8,9]]
a[a > 5]                            # boolean mask → [6, 7, ..., 11]

# Squeeze / unsqueeze
u = a.unsqueeze(0)                   # add dim → (1, 3, 4)
s = u.squeeze()                      # remove size-1 dims → (3, 4)

# Concatenation and stacking
c = sv.concat([a, a], axis=0)        # (6, 4)
s = sv.stack([a, a], axis=0)         # (2, 3, 4)

# Split
parts = sv.split(a, 3, axis=0)      # 3 tensors of shape (1, 4)
```

### Linear Algebra

```python
import pyscivex as sv

a = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
b = sv.Tensor([[5.0, 6.0], [7.0, 8.0]])

# Matrix multiply
c = a @ b                            # operator
c = sv.linalg.matmul(a, b)          # function

# Dot product
x = sv.Tensor([1.0, 2.0, 3.0])
y = sv.Tensor([4.0, 5.0, 6.0])
d = sv.linalg.dot(x, y)             # 32.0

# Solve Ax = b
A = sv.Tensor([[3.0, 1.0], [1.0, 2.0]])
b = sv.Tensor([9.0, 8.0])
x = sv.linalg.solve(A, b)           # [2.0, 3.0]

# Inverse and determinant
inv_a = sv.linalg.inv(a)
det_a = sv.linalg.det(a)            # -2.0

# Least squares
x = sv.linalg.lstsq(A, b)

# Norm
n = sv.linalg.norm(x)               # L2 norm
n = sv.linalg.norm(x, ord=1)        # L1 norm
n = sv.linalg.norm(a, ord="fro")    # Frobenius norm

# Decompositions
L, U, P = sv.linalg.lu(a)
Q, R = sv.linalg.qr(a)
U, S, Vt = sv.linalg.svd(a)
eigenvalues, eigenvectors = sv.linalg.eig(a)
L = sv.linalg.cholesky(a)           # Cholesky (for positive definite)

# Use decompositions
rank = sv.linalg.matrix_rank(a)
cond = sv.linalg.cond(a)            # condition number
```

### Einstein Summation

```python
a = sv.Tensor.ones((3, 4))
b = sv.Tensor.ones((4, 5))
x = sv.Tensor.ones((3,))
y = sv.Tensor.ones((3,))

c = sv.einsum("ij,jk->ik", a, b)    # matmul → (3, 5)
t = sv.einsum("ii->", a)             # trace
o = sv.einsum("i,j->ij", x, y)      # outer product
d = sv.einsum("i,i->", x, y)        # dot product
```

### Sparse Matrices

```python
# Build from triplets (row, col, value)
rows = [0, 0, 1, 2, 2]
cols = [0, 2, 1, 0, 2]
vals = [1.0, 2.0, 3.0, 4.0, 5.0]
csr = sv.sparse.CsrMatrix(rows, cols, vals, shape=(3, 3))

# From dense
dense = sv.Tensor.eye(100)
sparse = sv.sparse.CsrMatrix.from_dense(dense)

# Operations
y = csr.matvec(x)                    # sparse-dense multiply
d = csr.to_dense()                   # convert back

# Random sparse matrix
sv_csr = sv.sparse.CsrMatrix.random(100, 100, density=0.1, seed=42)
```

### FFT

```python
signal = sv.Tensor([1.0, 0.0, -1.0, 0.0] * 16)

# 1-D FFT
spectrum = sv.fft.fft(signal)        # complex output (n, 2)
recovered = sv.fft.ifft(spectrum)    # inverse FFT

# Real FFT (faster for real input)
rfft_out = sv.fft.rfft(signal)
recovered = sv.fft.irfft(rfft_out)

# 2-D FFT (for images)
image = sv.Tensor.ones((64, 64))
freq = sv.fft.fft2(image)
```

### Random Number Generation

```python
sv.random.seed(42)                    # reproducible results

# Distributions
u = sv.random.uniform((100,))                     # U(0, 1)
n = sv.random.normal((100,), mean=0, std=1)       # N(0, 1)
r = sv.random.randint((50,), low=0, high=10)      # integers in [0, 10)

# Utilities
sv.random.shuffle(tensor)                          # in-place shuffle
choice = sv.random.choice(tensor, n=5)             # random selection
perm = sv.random.permutation(10)                   # random permutation
```

---

## 3. DataFrames

> **Replaces:** pandas, polars

### Creating DataFrames

```python
import pyscivex as sv

# From dict (primary — like pandas)
df = sv.DataFrame({
    "name": ["Alice", "Bob", "Carol", "Dave"],
    "age": [30, 25, 35, 28],
    "salary": [95000.0, 87500.0, 92000.0, 78000.0],
    "department": ["Engineering", "Marketing", "Engineering", "Sales"],
})

print(df)
# ┌───────┬─────┬──────────┬─────────────┐
# │ name  │ age │ salary   │ department  │
# ├───────┼─────┼──────────┼─────────────┤
# │ Alice │  30 │ 95000.00 │ Engineering │
# │ Bob   │  25 │ 87500.00 │ Marketing   │
# │ Carol │  35 │ 92000.00 │ Engineering │
# │ Dave  │  28 │ 78000.00 │ Sales       │
# └───────┴─────┴──────────┴─────────────┘

# From records (list of dicts)
df = sv.DataFrame.from_records([
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
])

# From random tensor
data = sv.random.normal((100, 3))
df = sv.DataFrame.from_tensor(data, columns=["a", "b", "c"])

# Shape info
print(df.shape)          # (4, 4)
print(df.nrows)          # 4
print(df.ncols)          # 4
print(df.column_names)   # ["name", "age", "salary", "department"]
print(df.dtypes)         # {"name": "str", "age": "i64", "salary": "f64", ...}
```

### Column Access and Selection

```python
# Single column → Series
ages = df["age"]                      # Series
print(ages.mean())                    # 29.5

# Multiple columns → DataFrame
subset = df[["name", "salary"]]

# Attribute access
df.age                                # same as df["age"]

# Select / drop
selected = df.select(["name", "age"])
dropped = df.drop(["department"])
```

### Filtering and Querying

```python
# Boolean indexing (pandas-style)
seniors = df[df["age"] > 30]
engineers = df[df["department"] == "Engineering"]

# Combined conditions
high_paid_eng = df[(df["department"] == "Engineering") & (df["salary"] > 90000)]

# String query (SQL-like)
result = df.query("age > 28 and salary > 85000")

# Head / tail / sample
df.head(5)
df.tail(5)
df.sample(3)
```

### Sorting

```python
# Sort by column
sorted_df = df.sort_values("salary", ascending=False)

# Sort by multiple columns
sorted_df = df.sort_values(["department", "salary"], ascending=[True, False])
```

### GroupBy and Aggregation

```python
# Basic groupby
grouped = df.groupby("department")
result = grouped.mean()                # mean of numeric columns per group
result = grouped.sum()
result = grouped.count()

# Multiple aggregations
result = df.groupby("department").agg({
    "salary": "mean",
    "age": ["min", "max", "count"],
})

# Custom aggregation
result = df.groupby("department").apply(
    lambda group: group["salary"].max() - group["salary"].min()
)
```

### Joins and Merging

```python
employees = sv.DataFrame({
    "emp_id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Carol", "Dave"],
    "dept_id": [10, 20, 10, 30],
})

departments = sv.DataFrame({
    "dept_id": [10, 20, 30],
    "dept_name": ["Engineering", "Marketing", "Sales"],
})

# Inner join (default)
merged = employees.merge(departments, on="dept_id")

# Left join
merged = employees.merge(departments, on="dept_id", how="left")

# Right / outer join
merged = employees.merge(departments, on="dept_id", how="outer")
```

### Pivot and Melt

```python
# Pivot (long → wide)
sales = sv.DataFrame({
    "quarter": ["Q1", "Q1", "Q2", "Q2"],
    "product": ["A", "B", "A", "B"],
    "revenue": [100.0, 200.0, 150.0, 250.0],
})
wide = sales.pivot(index="quarter", columns="product", values="revenue")
# quarter | A     | B
# Q1      | 100.0 | 200.0
# Q2      | 150.0 | 250.0

# Melt (wide → long)
long = wide.melt(id_vars=["quarter"], value_vars=["A", "B"],
                 var_name="product", value_name="revenue")
```

### Null Handling

```python
df_nulls = sv.DataFrame({
    "x": [1.0, None, 3.0, None, 5.0],
    "y": [10.0, 20.0, None, 40.0, 50.0],
})

df_nulls.dropna()                     # drop rows with any null
df_nulls.fillna(0.0)                  # fill nulls with 0
df_nulls.fillna(method="ffill")       # forward fill
df_nulls.fillna(method="bfill")       # backward fill
df_nulls.interpolate()                # linear interpolation
df_nulls.isna()                       # boolean mask of nulls
```

### Rolling Windows

```python
prices = sv.DataFrame({"price": [100, 102, 101, 105, 108, 107, 110, 115]})

# Rolling calculations
prices["ma_3"] = prices["price"].rolling(3).mean()
prices["std_3"] = prices["price"].rolling(3).std()
prices["min_3"] = prices["price"].rolling(3).min()
prices["max_3"] = prices["price"].rolling(3).max()

# Exponential weighted moving average
prices["ewma"] = prices["price"].ewm(alpha=0.3).mean()
```

### String and DateTime Operations

```python
# String operations
df["name"].str.upper()
df["name"].str.lower()
df["name"].str.contains("ali", case=False)
df["name"].str.replace("Bob", "Robert")
df["name"].str.len()
df["name"].str.split(" ")

# DateTime operations
df["date"].dt.year
df["date"].dt.month
df["date"].dt.day
df["date"].dt.hour
df["date"].dt.dayofweek
```

### Lazy API (Deferred Execution)

```python
# Build query lazily, execute at collect()
result = (
    df.lazy()
      .filter(sv.col("age") > 25)
      .select(["name", "salary"])
      .sort("salary", descending=True)
      .limit(10)
      .collect()
)
```

### SQL Queries on DataFrames

```python
# Query DataFrames with SQL
result = sv.sql("""
    SELECT department, AVG(salary) as avg_salary, COUNT(*) as count
    FROM employees
    WHERE age > 25
    GROUP BY department
    ORDER BY avg_salary DESC
""", employees=df)

# Multi-table SQL
result = sv.sql("""
    SELECT e.name, d.dept_name, e.salary
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
    WHERE e.salary > 85000
""", employees=employees, departments=departments)
```

### Describe and Summary

```python
print(df.describe())
# ┌──────────┬───────┬──────────┐
# │          │ age   │ salary   │
# ├──────────┼───────┼──────────┤
# │ count    │ 4.0   │ 4.0      │
# │ mean     │ 29.5  │ 88125.0  │
# │ std      │ 4.2   │ 7395.0   │
# │ min      │ 25.0  │ 78000.0  │
# │ 25%      │ 27.25 │ 85125.0  │
# │ 50%      │ 29.0  │ 89750.0  │
# │ 75%      │ 31.25 │ 92750.0  │
# │ max      │ 35.0  │ 95000.0  │
# └──────────┴───────┴──────────┘
```

---

## 4. Data I/O

> **Replaces:** pandas.read_csv, sqlalchemy, pyarrow, openpyxl, h5py

### CSV

```python
# Read
df = sv.read_csv("data.csv")
df = sv.read_csv("data.csv", delimiter=";", has_header=True, skip_rows=1)
df = sv.read_csv("data.csv", columns=["name", "age"])  # select columns

# Write
df.to_csv("output.csv")
df.to_csv("output.csv", delimiter="\t", include_header=True)
```

### JSON

```python
# Read
df = sv.read_json("data.json")
df = sv.read_json("data.json", orient="records")    # [{...}, {...}]
df = sv.read_json("data.json", orient="columns")    # {"col": [...]}

# Write
df.to_json("output.json", orient="records", pretty=True)
```

### Parquet

```python
# Read
df = sv.read_parquet("data.parquet")
df = sv.read_parquet("data.parquet", columns=["name", "age"])

# Write
df.to_parquet("output.parquet", compression="snappy")
```

### Excel

```python
# Read
df = sv.read_excel("data.xlsx", sheet="Sheet1")

# Write
df.to_excel("output.xlsx", sheet="Results")
```

### SQL Databases

```python
# PostgreSQL
df = sv.read_sql("SELECT * FROM users WHERE age > 25",
                 "postgresql://user:pass@localhost:5432/mydb")

# MySQL
df = sv.read_sql("SELECT * FROM orders",
                 "mysql://user:pass@localhost:3306/shop")

# SQL Server (MSSQL)
df = sv.read_sql("SELECT * FROM inventory",
                 "mssql://user:pass@server:1433/warehouse")

# SQLite
df = sv.read_sql("SELECT * FROM logs",
                 "sqlite:///local.db")

# DuckDB
df = sv.read_sql("SELECT * FROM analytics",
                 "duckdb:///analytics.ddb")

# Write to database
df.to_sql("processed_users", "postgresql://user:pass@localhost:5432/mydb",
          if_exists="replace")
```

### Arrow IPC

```python
# Read/write Apache Arrow IPC format
df = sv.read_arrow("data.arrow")
df.to_arrow_file("output.arrow")

# pyscivex handles Arrow natively — no pyarrow needed
df.to_arrow_file("output.arrow")
```

### Other Formats

```python
# Avro
df = sv.read_avro("data.avro")
df.to_avro("output.avro")

# ORC
df = sv.read_orc("data.orc")

# HDF5
df = sv.read_hdf5("data.h5", dataset="measurements")
df.to_hdf5("output.h5", dataset="results")

# NumPy .npy / .npz
tensor = sv.io.read_npy("weights.npy")
sv.io.write_npy("output.npy", tensor)

# Delta Lake
df = sv.read_delta("s3://bucket/delta-table/")
```

### Complete I/O Workflow

```python
import pyscivex as sv

# 1. Load from PostgreSQL
raw = sv.read_sql(
    "SELECT * FROM sensor_data WHERE ts > '2025-01-01'",
    "postgresql://admin:secret@db.example.com/iot"
)

# 2. Clean and transform
clean = (
    raw.dropna()
       .sort_values("ts")
       .query("value > 0")
)

# 3. Feature engineering
clean["hour"] = clean["ts"].dt.hour
clean["ma_5"] = clean["value"].rolling(5).mean()

# 4. Save processed data
clean.to_parquet("processed_sensor_data.parquet")

# 5. Convert to tensors for ML
X = clean[["hour", "ma_5", "value"]].to_tensor()
```

---

## 5. Statistics & Distributions

> **Replaces:** scipy.stats, statsmodels

### Descriptive Statistics

```python
import pyscivex as sv

data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

sv.stats.mean(data)         # 5.0
sv.stats.std_dev(data)      # 2.0
sv.stats.variance(data)     # 4.0
sv.stats.median(data)       # 4.5
sv.stats.quantile(data, 0.25)  # Q1
sv.stats.skewness(data)     # skewness
sv.stats.kurtosis(data)     # excess kurtosis

# Full summary
summary = sv.stats.describe(data)
print(summary)
# DescriptiveStats { mean: 5.0, std: 2.0, min: 2.0, max: 9.0, ... }
```

### Probability Distributions

```python
# Normal distribution
norm = sv.stats.Normal(mu=0.0, sigma=1.0)
norm.pdf(0.0)              # 0.3989... (density at x=0)
norm.cdf(1.96)             # 0.975 (cumulative probability)
norm.ppf(0.975)            # 1.96 (inverse CDF / quantile)
samples = norm.sample(1000) # 1000 random samples
norm.mean()                # 0.0
norm.var()                 # 1.0

# Student's t
t = sv.stats.StudentT(df=10)
t.pdf(0.0)
t.cdf(2.228)

# Other distributions
gamma   = sv.stats.Gamma(alpha=2.0, beta=1.0)
beta    = sv.stats.Beta(alpha=2.0, beta=5.0)
poisson = sv.stats.Poisson(lambda_=3.0)
binom   = sv.stats.Binomial(n=10, p=0.5)
exp     = sv.stats.Exponential(lambda_=1.0)
chi2    = sv.stats.ChiSquared(df=5)
weibull = sv.stats.Weibull(k=1.5, lambda_=1.0)
pareto  = sv.stats.Pareto(alpha=2.0, x_m=1.0)
uniform = sv.stats.Uniform(a=0.0, b=1.0)

# All have: .pdf(), .cdf(), .ppf(), .sample(n), .mean(), .var()
```

### Hypothesis Tests

```python
# One-sample t-test
result = sv.stats.ttest_1samp(data, popmean=5.0)
print(f"t={result.statistic:.4f}, p={result.p_value:.4f}")

# Two-sample t-test
group_a = [23.0, 25.0, 28.0, 30.0, 32.0]
group_b = [19.0, 22.0, 24.0, 26.0, 28.0]
result = sv.stats.ttest_ind(group_a, group_b)
print(f"t={result.statistic:.4f}, p={result.p_value:.4f}")

# Chi-square test of independence
observed = [[10, 20, 30], [6, 9, 17]]
result = sv.stats.chi2_test(observed)
print(f"chi2={result.statistic:.4f}, p={result.p_value:.4f}")

# One-way ANOVA
result = sv.stats.anova_oneway([group_a, group_b])
print(f"F={result.statistic:.4f}, p={result.p_value:.4f}")

# Kolmogorov-Smirnov test
result = sv.stats.ks_2samp(group_a, group_b)

# Mann-Whitney U test (non-parametric)
result = sv.stats.mann_whitney_u(group_a, group_b)
```

### Correlation

```python
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.1, 4.0, 5.8, 8.1, 9.9]

r = sv.stats.pearsonr(x, y)         # Pearson correlation
rho = sv.stats.spearmanr(x, y)      # Spearman rank correlation
tau = sv.stats.kendalltau(x, y)      # Kendall tau

# Correlation matrix from DataFrame
corr = df[["age", "salary"]].corr()
```

### Regression

```python
# OLS regression
X = sv.Tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 7.0]])
y = sv.Tensor([3.0, 5.0, 8.0, 11.0])

result = sv.stats.ols(X, y)
print(result.coefficients)           # fitted coefficients
print(result.r_squared)              # R-squared
print(result.p_values)               # p-values for each coefficient
print(result.summary())              # full summary table

# GLM (generalized linear model)
result = sv.stats.glm(X, y_binary, family="binomial")
print(result.coefficients)
```

### Confidence Intervals and Effect Sizes

```python
# Confidence interval for mean
ci = sv.stats.ci_mean(data, confidence=0.95)
print(f"95% CI: [{ci.lower:.2f}, {ci.upper:.2f}]")

# Confidence interval for proportion
ci = sv.stats.ci_proportion(successes=45, n=100, confidence=0.95)

# Effect sizes
d = sv.stats.cohens_d(group_a, group_b)        # Cohen's d
g = sv.stats.hedges_g(group_a, group_b)        # Hedges' g

# Multiple comparison corrections
p_values = [0.01, 0.04, 0.05, 0.10, 0.30]
adjusted = sv.stats.bonferroni(p_values)         # Bonferroni
adjusted = sv.stats.benjamini_hochberg(p_values) # FDR
```

### Survival Analysis

```python
# Kaplan-Meier estimator
times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
events = [True, True, False, True, False, True]  # True = event, False = censored
km = sv.stats.kaplan_meier(times, events)

# Log-rank test
result = sv.stats.log_rank_test(times_a, events_a, times_b, events_b)

# Cox proportional hazards
result = sv.stats.cox_ph(X, times, events)
```

---

## 6. Time Series Analysis

> **Replaces:** statsmodels.tsa, prophet, pmdarima

### Autocorrelation and Stationarity

```python
ts = sv.Tensor([...])  # time series data

acf_vals = sv.stats.acf(ts, max_lag=20)
pacf_vals = sv.stats.pacf(ts, max_lag=20)

# Augmented Dickey-Fuller test
result = sv.stats.adf_test(ts)
print(f"ADF stat={result.statistic:.4f}, p={result.p_value:.4f}")
if result.p_value < 0.05:
    print("Series is stationary")
```

### Seasonal Decomposition

```python
result = sv.stats.seasonal_decompose(ts, period=12)
trend = result.trend
seasonal = result.seasonal
residual = result.residual
```

### ARIMA / SARIMAX

```python
# ARIMA(1,1,1)
model = sv.stats.ARIMA(p=1, d=1, q=1)
model.fit(ts)
forecast = model.predict(h=12)         # 12-step forecast
print(forecast)

# SARIMAX (seasonal ARIMA)
model = sv.stats.SARIMAX(p=1, d=1, q=1, P=1, D=1, Q=1, s=12)
model.fit(ts)
forecast = model.predict(h=24)
```

### Exponential Smoothing

```python
model = sv.stats.ExponentialSmoothing(
    method="triple",                    # Holt-Winters
    alpha=0.3, beta=0.1, gamma=0.1,
    season_length=12
)
model.fit(ts)
forecast = model.predict(h=6)
```

### Prophet-style Forecasting

```python
model = sv.stats.Prophet(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    n_changepoints=25,
)
model.fit(dates, values)
future_forecast = model.predict(future_dates)
```

### Anomaly Detection

```python
# Z-score based
anomalies = sv.stats.zscore_anomaly(ts, threshold=3.0)

# Seasonal residual based
anomalies = sv.stats.seasonal_anomaly(ts, period=24, threshold=2.5)

# EWMA-based
anomalies = sv.stats.ewma_anomaly(ts, alpha=0.3, threshold=3.0)
```

### Kalman Filter

```python
kf = sv.stats.KalmanFilter(dim_x=2, dim_z=1)
# Configure state transition, measurement matrices, etc.
kf.F = [[1, 1], [0, 1]]    # state transition
kf.H = [[1, 0]]             # measurement function
kf.Q = [[0.1, 0], [0, 0.1]] # process noise
kf.R = [[1.0]]              # measurement noise

for measurement in measurements:
    kf.predict()
    kf.update(measurement)
    print(f"State: {kf.x}, Covariance: {kf.P}")
```

---

## 7. Bayesian Inference & MCMC

> **Replaces:** PyMC, emcee, pystan

### MCMC Sampling

```python
import pyscivex as sv

# Define log-posterior
def log_posterior(params):
    mu, sigma = params[0], params[1]
    if sigma <= 0:
        return float("-inf")
    # Prior: N(0, 10) for mu, HalfNormal(5) for sigma
    log_prior = -0.5 * (mu / 10) ** 2 - 0.5 * (sigma / 5) ** 2
    # Likelihood: data ~ N(mu, sigma)
    log_lik = sum(-0.5 * ((x - mu) / sigma) ** 2 - sv.log(sigma) for x in data)
    return log_prior + log_lik

# Metropolis-Hastings
sampler = sv.stats.MetropolisHastings(
    n_samples=10000,
    warmup=2000,
    seed=42,
)
chains = sampler.sample(log_posterior, initial=[0.0, 1.0])

# Hamiltonian Monte Carlo (more efficient)
sampler = sv.stats.HMC(
    n_samples=5000,
    warmup=1000,
    step_size=0.01,
    n_leapfrog=20,
    seed=42,
)
chains = sampler.sample(log_posterior, grad_log_posterior, initial=[0.0, 1.0])

# NUTS (No U-Turn Sampler — gold standard)
sampler = sv.stats.NUTS(
    n_samples=5000,
    warmup=1000,
    target_accept=0.8,
    seed=42,
)
chains = sampler.sample(log_posterior, grad_log_posterior, initial=[0.0, 1.0])
```

### Diagnostics

```python
# Effective sample size
ess = sv.stats.effective_sample_size(chains)

# R-hat convergence diagnostic
rhat = sv.stats.rhat(chains)

# Trace summary
summary = sv.stats.trace_summary(chains)
print(summary)  # mean, std, HDI, ESS, R-hat per parameter
```

### Bayesian Optimization

```python
optimizer = sv.stats.BayesianOptimizer(
    kernel="matern",
    acquisition="expected_improvement",
    n_initial=5,
    n_iter=50,
    seed=42,
)

def objective(x):
    return -(x[0] ** 2 + x[1] ** 2)  # maximize (minimize negative)

bounds = [(-5.0, 5.0), (-5.0, 5.0)]
result = optimizer.optimize(objective, bounds)
print(f"Best params: {result.x}, Best value: {result.fun}")
```

---

## 8. Optimization & Solvers

> **Replaces:** scipy.optimize, scipy.integrate, scipy.interpolate

### Minimization

```python
import pyscivex as sv

# BFGS (quasi-Newton, with gradient)
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return [dx, dy]

result = sv.optim.minimize(rosenbrock, x0=[0.0, 0.0],
                           method="bfgs", jac=rosenbrock_grad)
print(f"Minimum at {result.x}, f = {result.fun}")
# Minimum at [1.0, 1.0], f ≈ 0.0

# Nelder-Mead (derivative-free)
result = sv.optim.minimize(rosenbrock, x0=[0.0, 0.0],
                           method="nelder-mead")

# L-BFGS-B (with box constraints)
result = sv.optim.minimize(rosenbrock, x0=[0.0, 0.0],
                           method="l-bfgs-b", jac=rosenbrock_grad,
                           bounds=[(0, 10), (0, 10)])

# Gradient descent
result = sv.optim.minimize(rosenbrock, x0=[0.0, 0.0],
                           method="gradient-descent", jac=rosenbrock_grad,
                           options={"learning_rate": 0.001, "max_iter": 5000})
```

### Root Finding

```python
# Brent's method (bracketed)
root = sv.optim.brentq(lambda x: x**3 - 2*x - 5, a=1.0, b=3.0)
print(f"Root: {root}")  # ~2.0946

# Newton's method (with derivative)
root = sv.optim.newton(lambda x: x**2 - 2, x0=1.0,
                       fprime=lambda x: 2*x)

# Bisection
root = sv.optim.bisect(lambda x: x**3 - 1, a=0.0, b=2.0)
```

### Numerical Integration

```python
import math

# Adaptive quadrature (most accurate)
result = sv.optim.quad(lambda x: math.exp(-x**2), a=0.0, b=1.0)
print(f"Integral: {result.value:.10f}, error: {result.error:.2e}")

# Trapezoid rule
result = sv.optim.trapezoid(lambda x: x**2, a=0.0, b=1.0, n=1000)

# Simpson's rule
result = sv.optim.simpson(lambda x: math.sin(x), a=0.0, b=math.pi, n=100)
```

### Interpolation

```python
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [0.0, 0.8, 0.9, 0.1, -0.8]

# Linear interpolation
f = sv.optim.interp1d(x, y, kind="linear")
print(f(1.5))   # interpolated value at x=1.5

# Cubic spline
f = sv.optim.interp1d(x, y, kind="cubic")
y_smooth = [f(xi) for xi in sv.Tensor.linspace(0, 4, 100).to_list()]

# CubicSpline object
cs = sv.optim.CubicSpline(x, y)
print(cs(2.5))

# 2-D interpolation
f2d = sv.optim.interp2d(x_grid, y_grid, z_values, kind="bicubic")
```

### ODE Solvers

```python
# dy/dt = -y (exponential decay)
def decay(t, y):
    return [-y[0]]

result = sv.optim.solve_ivp(decay, t_span=(0.0, 5.0), y0=[1.0],
                             method="rk45", max_step=0.01)
print(result.t)   # time points
print(result.y)   # solution values

# Lorenz attractor (3-D system)
def lorenz(t, y, sigma=10, rho=28, beta=8/3):
    return [
        sigma * (y[1] - y[0]),
        y[0] * (rho - y[2]) - y[1],
        y[0] * y[1] - beta * y[2],
    ]

result = sv.optim.solve_ivp(lorenz, t_span=(0, 50), y0=[1, 1, 1],
                             method="rk45")
```

### PDE Solvers

```python
# Heat equation
result = sv.optim.heat_equation_1d(
    alpha=0.01, length=1.0, nx=100, nt=1000, dt=0.001,
    initial=lambda x: math.sin(math.pi * x),
    bc_left=sv.optim.BoundaryCondition.dirichlet(0.0),
    bc_right=sv.optim.BoundaryCondition.dirichlet(0.0),
)

# Laplace equation (2-D steady state)
result = sv.optim.laplace_2d(nx=50, ny=50, max_iter=10000, tol=1e-6)
```

### Linear & Quadratic Programming

```python
# Linear programming: minimize c^T x subject to A_ub x <= b_ub, x >= 0
c = [-1.0, -2.0]              # minimize -x - 2y (= maximize x + 2y)
A_ub = [[1, 1], [1, -1]]      # x+y <= 4, x-y <= 2
b_ub = [4.0, 2.0]

result = sv.optim.linprog(c, A_ub, b_ub)
print(f"Optimal: x={result.x}, value={result.fun}")

# Quadratic programming: minimize 0.5 x^T Q x + c^T x
Q = [[2.0, 0.0], [0.0, 2.0]]
c = [-2.0, -5.0]
A = [[1.0, -2.0], [-1.0, -2.0], [-1.0, 2.0]]
b = [2.0, 6.0, 2.0]
result = sv.optim.quadprog(Q, c, A, b)
```

### Curve Fitting

```python
# Fit y = a * exp(-b * x)
def model(x, a, b):
    return a * math.exp(-b * x)

x_data = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
y_data = [5.0, 3.4, 2.2, 1.5, 1.0, 0.7, 0.5]
p0 = [5.0, 1.0]  # initial guess

result = sv.optim.curve_fit(model, x_data, y_data, p0)
print(f"Fitted: a={result.params[0]:.3f}, b={result.params[1]:.3f}")
print(f"Residuals: {result.residuals}")
```

---

## 9. Signal Processing & Audio

> **Replaces:** scipy.signal, librosa

### Filtering

```python
import pyscivex as sv

# FIR low-pass filter
b = sv.signal.firwin(numtaps=51, cutoff=0.3)       # normalized freq
filtered = sv.signal.lfilter(b, [1.0], signal)

# Zero-phase filtering
filtered = sv.signal.filtfilt(b, [1.0], signal)

# FIR band-pass
b = sv.signal.firwin(numtaps=101, cutoff=[0.1, 0.4], pass_zero=False)

# IIR Butterworth
b, a = sv.signal.butter(order=4, cutoff=0.3, btype="low")
filtered = sv.signal.lfilter(b, a, signal)
```

### Spectral Analysis

```python
# Short-time Fourier Transform
spec = sv.signal.stft(signal, n_fft=1024, hop_length=256,
                       window="hann")

# Inverse STFT
reconstructed = sv.signal.istft(spec, hop_length=256)

# Spectrogram (magnitude)
Sxx, freqs, times = sv.signal.spectrogram(signal, fs=16000,
                                            n_fft=1024, hop_length=256)

# Power spectral density
freqs, Pxx = sv.signal.welch(signal, fs=16000, nperseg=256)
freqs, Pxx = sv.signal.periodogram(signal, fs=16000)
```

### Wavelets

```python
# Discrete wavelet transform
cA, cD = sv.signal.dwt(signal, wavelet="haar")

# Inverse DWT
reconstructed = sv.signal.idwt(cA, cD, wavelet="haar")
```

### Audio Features

```python
# Mel spectrogram
mel = sv.signal.mel_spectrogram(audio, sr=16000, n_mels=80,
                                  n_fft=1024, hop_length=256)

# MFCC (Mel-frequency cepstral coefficients)
mfccs = sv.signal.mfcc(audio, sr=16000, n_mfcc=13)

# Chroma features
chroma = sv.signal.chroma_stft(audio, sr=16000)

# Pitch detection
pitches = sv.signal.pitch_yin(audio, sr=16000, fmin=50, fmax=500)
```

### Peak Detection

```python
peaks = sv.signal.find_peaks(signal, height=0.5, distance=10,
                              prominence=0.1)
print(f"Found {len(peaks.indices)} peaks")
print(f"Peak positions: {peaks.indices}")
print(f"Peak heights: {peaks.heights}")

prominences = sv.signal.peak_prominences(signal, peaks.indices)
```

### Audio I/O

```python
# Read WAV file
audio, sr = sv.signal.read_wav("audio.wav")
print(f"Sample rate: {sr}, Duration: {len(audio)/sr:.2f}s")

# Write WAV file
sv.signal.write_wav("output.wav", audio, sr=16000)
```

### Window Functions

```python
hann    = sv.signal.hann(1024)
hamming = sv.signal.hamming(1024)
blackman = sv.signal.blackman(1024)
```

### Resampling and Convolution

```python
# Resample signal to new sample rate
resampled = sv.signal.resample(signal, target_length=8000)

# Decimate (downsample with anti-aliasing)
decimated = sv.signal.decimate(signal, factor=4)

# Convolution / correlation
conv = sv.signal.convolve(signal, kernel, mode="full")
corr = sv.signal.correlate(signal1, signal2)
```

### Beat Detection

```python
bpm, beat_times = sv.signal.detect_beats(audio, sr=16000)
print(f"Tempo: {bpm:.1f} BPM")
print(f"Beat positions: {beat_times}")
```

---

## 10. Image Processing & Computer Vision

> **Replaces:** Pillow, OpenCV, scikit-image

### Loading and Saving

```python
import pyscivex as sv

# Load image
img = sv.image.open("photo.bmp")
print(f"Size: {img.width}x{img.height}, Channels: {img.channels}")

# Save
img.save("output.bmp")

# From/to NumPy
# Convert to/from tensor
t = img.to_tensor()                  # → Tensor (H, W, C)
img = sv.image.from_tensor(t)

# Display in Jupyter
img  # auto-renders via _repr_png_()
```

### Transforms

```python
# Resize
resized = img.resize(224, 224, method="bilinear")
resized = img.resize(224, 224, method="lanczos")

# Crop
cropped = img.crop(x=10, y=10, width=100, height=100)

# Flip and rotate
flipped = img.flip_horizontal()
rotated = img.rotate90()
rotated = img.rotate(45)             # arbitrary angle

# Pad
padded = img.pad(top=10, bottom=10, left=10, right=10, value=0)
```

### Filters

```python
# Blur
blurred = img.gaussian_blur(sigma=1.5)
blurred = img.median_filter(kernel_size=5)

# Edge detection
edges = img.sobel()
edges = img.laplacian()
edges = img.canny(low=50, high=150)

# Custom convolution
kernel = sv.Tensor([[-1, -1, -1],
                     [-1,  8, -1],
                     [-1, -1, -1]], dtype="float64")
filtered = img.convolve2d(kernel)
```

### Color Space Conversion

```python
gray = img.to_grayscale()
hsv = img.to_hsv()
rgb = hsv.to_rgb()
```

### Feature Detection

```python
# Harris corner detection
corners = img.harris_corners(k=0.04, threshold=0.01)

# FAST corner detection
corners = img.fast_corners(threshold=20)

# ORB features (Oriented FAST + Rotated BRIEF)
detector = sv.image.OrbDetector(n_features=500)
keypoints = detector.detect(gray)
descriptors = detector.compute(gray, keypoints)

# Feature matching
matcher = sv.image.BruteForceMatcher()
matches = matcher.match(desc1, desc2)
```

### Morphological Operations

```python
kernel = sv.image.StructuringElement.rect(5, 5)

eroded  = img.erode(kernel)
dilated = img.dilate(kernel)
opened  = img.opening(kernel)         # erode then dilate
closed  = img.closing(kernel)         # dilate then erode
```

### Segmentation

```python
# Connected components
labels, n_components = img.connected_components()

# Region growing
labels = img.region_growing(seed=(100, 100), threshold=15)

# Watershed
labels = img.watershed(markers)
```

### Hough Transforms

```python
# Line detection
lines = sv.image.hough_lines(edges, threshold=100)
for rho, theta in lines:
    print(f"Line: rho={rho:.1f}, theta={theta:.3f}")

# Circle detection
circles = sv.image.hough_circles(gray, min_radius=10, max_radius=50)
```

### Optical Flow

```python
# Sparse (Lucas-Kanade)
flow = sv.image.lucas_kanade(prev_frame, next_frame, points)

# Dense (Farneback)
flow = sv.image.farneback(prev_frame, next_frame)
```

### Augmentation Pipeline

```python
# Build augmentation pipeline for ML training
pipeline = sv.image.AugmentPipeline() \
    .random_flip(horizontal=True, p=0.5) \
    .random_rotation(max_degrees=15) \
    .color_jitter(brightness=0.2, contrast=0.2, saturation=0.2) \
    .random_crop(224, 224) \
    .cutout(n_holes=1, size=32) \
    .normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Apply to images during training
augmented = pipeline.apply(img)
```

### Drawing

```python
img.draw_line(x1=10, y1=10, x2=100, y2=100, color=(255, 0, 0))
img.draw_rect(x=50, y=50, w=100, h=80, color=(0, 255, 0))
img.draw_circle(cx=150, cy=150, r=30, color=(0, 0, 255))
img.fill_rect(x=200, y=200, w=50, h=50, color=(255, 255, 0))
```

---

## 11. Natural Language Processing

> **Replaces:** NLTK, spaCy (basic), Gensim

### Tokenization

```python
import pyscivex as sv

text = "The quick brown fox jumps over the lazy dog."

# Word tokenization
tokens = sv.nlp.word_tokenize(text)
# ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

# Character tokenization
chars = sv.nlp.char_tokenize(text)

# N-gram tokenization
bigrams = sv.nlp.ngram_tokenize(text, n=2)
# [("The", "quick"), ("quick", "brown"), ...]

# WordPiece tokenization (subword)
tokenizer = sv.nlp.WordPieceTokenizer(vocab_size=5000)
tokenizer.train(corpus)
tokens = tokenizer.tokenize("unbelievable")
# ["un", "##believ", "##able"]
```

### Text Preprocessing

```python
# Stopword removal
cleaned = sv.nlp.remove_stopwords(tokens)

# Stemming
stemmer = sv.nlp.PorterStemmer()
stems = [stemmer.stem(word) for word in tokens]
# ["the", "quick", "brown", "fox", "jump", "over", "the", "lazi", "dog"]

# Edit distance
dist = sv.nlp.edit_distance("kitten", "sitting")  # 3

# Text normalization
clean = sv.nlp.normalize(text)   # lowercase, strip punctuation, etc.
```

### Vectorization

```python
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends",
]

# Bag-of-words (CountVectorizer)
cv = sv.nlp.CountVectorizer()
cv.fit(corpus)
X = cv.transform(corpus)           # sparse term-document matrix
print(cv.vocabulary)                # {"the": 0, "cat": 1, ...}

# TF-IDF
tfidf = sv.nlp.TfidfVectorizer()
tfidf.fit(corpus)
X = tfidf.transform(corpus)
```

### Word Embeddings (Word2Vec)

```python
# Train Word2Vec model
config = sv.nlp.Word2VecConfig(
    embedding_dim=100,
    window=5,
    min_count=1,
    learning_rate=0.025,
    epochs=5,
)
model = sv.nlp.Word2Vec(config)
model.train(corpus)                  # list of tokenized sentences

# Use embeddings
embeddings = model.embeddings()
vec = embeddings.get_vector("king")
similar = embeddings.most_similar("king", top_k=10)
# [("queen", 0.92), ("prince", 0.85), ...]

# Analogy: king - man + woman = queen
result = embeddings.analogy("king", "man", "woman")

# Similarity
sim = embeddings.similarity("cat", "dog")
```

### Sentiment Analysis

```python
analyzer = sv.nlp.SentimentAnalyzer()
result = analyzer.analyze("This movie was absolutely wonderful!")
print(f"Score: {result.score:.2f}")    # 0.85 (positive)
print(f"Label: {result.label}")        # "positive"
print(f"Positive words: {result.positive_words}")
```

### POS Tagging

```python
tagger = sv.nlp.HmmPosTagger()
tagger.train(tagged_corpus)           # list of (word, tag) pairs
tags = tagger.tag(["The", "cat", "sat"])
# [("The", "DT"), ("cat", "NN"), ("sat", "VBD")]
```

### Named Entity Recognition

```python
ner = sv.nlp.RuleBasedNer()
ner.add_pattern("PERSON", r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
ner.add_pattern("DATE", r"\b\d{4}-\d{2}-\d{2}\b")
ner.add_pattern("EMAIL", r"\b[\w.]+@[\w]+\.[\w]+\b")

entities = ner.extract("John Smith joined on 2025-01-15.")
# [Entity(text="John Smith", label="PERSON", start=0, end=10),
#  Entity(text="2025-01-15", label="DATE", start=21, end=31)]
```

### Topic Modeling (LDA)

```python
model = sv.nlp.LDA(
    n_topics=5,
    max_iter=100,
    alpha=0.1,
    beta=0.01,
)
model.fit(document_term_matrix)

# Get topic-word distributions
for topic_id in range(5):
    top_words = model.top_words(topic_id, n=10)
    print(f"Topic {topic_id}: {top_words}")

# Get document-topic distributions
doc_topics = model.transform(new_documents)
```

### Similarity Metrics

```python
# Cosine similarity
sim = sv.nlp.cosine_similarity(vec_a, vec_b)

# Jaccard similarity
sim = sv.nlp.jaccard_similarity(set_a, set_b)

# Edit distance (normalized)
dist = sv.nlp.edit_distance("python", "pytorch", normalize=True)
```

---

## 12. Graph & Network Analysis

> **Replaces:** NetworkX

### Building Graphs

```python
import pyscivex as sv

# Undirected graph
g = sv.Graph()
g.add_node(0)
g.add_node(1)
g.add_edge(0, 1, weight=1.5)
g.add_edge(1, 2, weight=2.0)
g.add_edge(0, 2, weight=3.0)

# Directed graph
dg = sv.DiGraph()
dg.add_edge(0, 1, weight=1.0)
dg.add_edge(1, 2, weight=2.0)

# From edge list
g = sv.Graph.from_edges([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)])

# From adjacency matrix
adj = sv.Tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype="float64")
g = sv.Graph.from_adjacency_matrix(adj)

# Graph info
print(f"Nodes: {g.num_nodes()}")
print(f"Edges: {g.num_edges()}")
print(f"Neighbors of 0: {g.neighbors(0)}")
```

### Shortest Paths

```python
# Dijkstra's algorithm (single source)
distances, predecessors = sv.graph.dijkstra(g, source=0)
print(f"Distance 0→2: {distances[2]}")

# Bellman-Ford (handles negative weights)
distances = sv.graph.bellman_ford(g, source=0)

# Floyd-Warshall (all pairs)
dist_matrix = sv.graph.floyd_warshall(g)
```

### Traversals

```python
# BFS
order = sv.graph.bfs(g, start=0)          # [0, 1, 2, ...]

# DFS
order = sv.graph.dfs(g, start=0)          # [0, 1, 2, ...]

# Topological sort (DAGs only)
topo = sv.graph.topological_sort(dg)
```

### Connectivity

```python
# Connected components (undirected)
components = sv.graph.connected_components(g)
print(f"Number of components: {len(components)}")

is_connected = sv.graph.is_connected(g)

# Strongly connected components (directed)
sccs = sv.graph.strongly_connected_components(dg)
```

### Minimum Spanning Trees

```python
mst_edges = sv.graph.kruskal(g)
# [(0, 1, 1.5), (1, 2, 2.0)]

mst_edges = sv.graph.prim(g)
```

### Centrality Measures

```python
# Degree centrality
dc = sv.graph.degree_centrality(g)

# Betweenness centrality (Brandes algorithm)
bc = sv.graph.betweenness_centrality(g)
print(f"Most central node: {max(bc, key=bc.get)}")

# PageRank
pr = sv.graph.pagerank(g, damping=0.85, max_iter=100, tol=1e-6)
print(f"PageRank scores: {pr}")
```

### Community Detection

```python
communities = sv.graph.label_propagation(g)
print(f"Communities: {communities}")
```

### Network Flow

```python
# Maximum flow (Edmonds-Karp)
flow_value, flow_dict = sv.graph.max_flow(g, source=0, sink=5)
print(f"Max flow: {flow_value}")

# Bipartite matching (Hopcroft-Karp)
matching = sv.graph.bipartite_matching(g, left_nodes=[0,1,2],
                                         right_nodes=[3,4,5])
```

---

## 13. Symbolic Mathematics

> **Replaces:** SymPy

### Creating Expressions

```python
import pyscivex as sv

x = sv.sym.var("x")
y = sv.sym.var("y")
pi = sv.sym.pi
e = sv.sym.e

# Build expressions with operators
expr = x**2 + 2*x + 1
expr2 = sv.sym.sin(x) * sv.sym.cos(y)
expr3 = sv.sym.exp(-x**2)

print(expr)           # x^2 + 2*x + 1
print(expr.latex())   # x^{2} + 2x + 1
```

### Evaluation and Substitution

```python
# Evaluate at a point
result = expr.eval(x=3.0)            # 16.0

# Substitute
simplified = expr.substitute(x=y + 1)
print(simplified)                     # (y+1)^2 + 2*(y+1) + 1
```

### Calculus

```python
# Differentiation
f = x**3 + 2*x**2 + x
df = sv.sym.diff(f, x)               # 3*x^2 + 4*x + 1
d2f = sv.sym.diff(f, x, n=2)         # 6*x + 4

# Partial derivatives
g = x**2 * y + sv.sym.sin(y)
dg_dx = sv.sym.diff(g, x)            # 2*x*y
dg_dy = sv.sym.diff(g, y)            # x^2 + cos(y)

# Integration
F = sv.sym.integrate(f, x)           # antiderivative
val = sv.sym.definite_integral(f, x, a=0.0, b=1.0)  # definite integral
```

### Simplification

```python
expr = x**2 + 2*x + 1
simplified = sv.sym.simplify(expr)    # (x + 1)^2
expanded = sv.sym.expand((x + 1)**2) # x^2 + 2*x + 1
factored = sv.sym.factor(x**2 - 1)   # (x - 1)*(x + 1)
```

### Equation Solving

```python
# Linear equation
roots = sv.sym.solve(2*x + 3, x)     # [-1.5]

# Quadratic equation
roots = sv.sym.solve(x**2 - 5*x + 6, x)  # [2.0, 3.0]
```

### Polynomials

```python
# Create polynomial from coefficients [a_n, ..., a_1, a_0]
p = sv.sym.Polynomial([1, -5, 6])    # x^2 - 5x + 6

p.eval(3.0)                          # 0.0
p.roots()                            # [2.0, 3.0]
p.derivative()                       # 2x - 5
p.degree()                           # 2
```

### Taylor Series

```python
# Taylor expansion of sin(x) around x=0, order 7
series = sv.sym.taylor(sv.sym.sin(x), x, center=0.0, order=7)
print(series)  # x - x^3/6 + x^5/120 - x^7/5040

# Maclaurin series (center=0)
series = sv.sym.maclaurin(sv.sym.exp(x), x, order=5)
print(series)  # 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120
```

---

## 14. Classical Machine Learning

> **Replaces:** scikit-learn

### Data Preparation

```python
import pyscivex as sv

# Train/test split
X_train, X_test, y_train, y_test = sv.ml.train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cross-validation
scores = sv.ml.cross_val_score(model, X, y, cv=5)
print(f"Mean: {scores.mean():.4f} +/- {scores.std():.4f}")

# K-Fold
kfold = sv.ml.KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kfold.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
```

### Preprocessing

```python
# Standard scaling (zero mean, unit variance)
scaler = sv.ml.StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Min-max scaling to [0, 1]
scaler = sv.ml.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encoding
encoder = sv.ml.OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical)

# Label encoding
le = sv.ml.LabelEncoder()
y_encoded = le.fit_transform(labels)

# Polynomial features
poly = sv.ml.PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

### Linear Models

```python
# Linear regression
model = sv.ml.LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"R2: {model.score(X_test, y_test):.4f}")
print(f"Weights: {model.weights}")
print(f"Bias: {model.bias}")

# Ridge regression (L2 regularization)
model = sv.ml.Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Lasso regression (L1 regularization)
model = sv.ml.Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Logistic regression (classification)
model = sv.ml.LogisticRegression(learning_rate=0.01, max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Decision Trees

```python
# Classification
tree = sv.ml.DecisionTreeClassifier(max_depth=10, min_samples_split=5)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
print(f"Feature importances: {tree.feature_importances}")

# Regression
tree = sv.ml.DecisionTreeRegressor(max_depth=8)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
```

### Ensemble Methods

```python
# Random Forest
rf = sv.ml.RandomForestClassifier(n_trees=100, max_depth=10, seed=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
probabilities = rf.predict_proba(X_test)
importances = rf.feature_importances

# Gradient Boosting
gb = sv.ml.GradientBoostingClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1
)
gb.fit(X_train, y_train)
predictions = gb.predict(X_test)

# Random Forest Regressor
rf_reg = sv.ml.RandomForestRegressor(n_trees=100, max_depth=10)
rf_reg.fit(X_train, y_train)
```

### Support Vector Machines

```python
# SVC with RBF kernel
svm = sv.ml.SVC(kernel="rbf", C=1.0, gamma=0.1)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)

# SVR (regression)
svr = sv.ml.SVR(kernel="rbf", C=1.0, epsilon=0.1)
svr.fit(X_train, y_train)

# Kernel options: "linear", "polynomial", "rbf"
```

### K-Nearest Neighbors

```python
# Classification
knn = sv.ml.KNeighborsClassifier(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Regression
knn_reg = sv.ml.KNeighborsRegressor(k=5)
knn_reg.fit(X_train, y_train)
```

### Naive Bayes

```python
nb = sv.ml.GaussianNB()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
probabilities = nb.predict_proba(X_test)
```

### Clustering

```python
# K-Means
kmeans = sv.ml.KMeans(n_clusters=3, max_iter=100, n_init=10, seed=42)
kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.centroids
inertia = kmeans.inertia

# DBSCAN
db = sv.ml.DBSCAN(eps=0.5, min_samples=5)
db.fit(X)
labels = db.labels
n_clusters = db.n_clusters
```

### Dimensionality Reduction

```python
# PCA
pca = sv.ml.PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)
print(f"Explained variance: {pca.explained_variance_ratio}")

# t-SNE
tsne = sv.ml.TSNE(n_components=2, perplexity=30.0)
X_embedded = tsne.fit_transform(X)
```

### Metrics

```python
# Classification metrics
acc = sv.ml.accuracy_score(y_true, y_pred)
prec = sv.ml.precision_score(y_true, y_pred)
rec = sv.ml.recall_score(y_true, y_pred)
f1 = sv.ml.f1_score(y_true, y_pred)
auc = sv.ml.roc_auc_score(y_true, y_proba)
cm = sv.ml.confusion_matrix(y_true, y_pred)

# Classification report
report = sv.ml.classification_report(y_true, y_pred)
print(report)

# Regression metrics
mse = sv.ml.mean_squared_error(y_true, y_pred)
rmse = sv.ml.rmse(y_true, y_pred)
mae = sv.ml.mean_absolute_error(y_true, y_pred)
r2 = sv.ml.r2_score(y_true, y_pred)
```

### Approximate Nearest Neighbors

```python
# Build ANN index (fast similarity search)
index = sv.ml.AnnoyIndex(dim=128, n_trees=10)
for i, vec in enumerate(embeddings):
    index.add(i, vec)
index.build()

# Query nearest neighbors
neighbors = index.query(query_vector, k=10)
# [(idx, distance), ...]
```

### Model Persistence

```python
# Save model
model.save("model.svex")

# Load model
model = sv.ml.load("model.svex")
predictions = model.predict(X_new)
```

---

## 15. Neural Networks & Deep Learning

> **Replaces:** PyTorch, TensorFlow, Keras

### Tensors with Autograd

```python
import pyscivex as sv

# Create tensors with gradient tracking
x = sv.nn.tensor([2.0, 3.0], requires_grad=True)
y = sv.nn.tensor([4.0, 5.0], requires_grad=True)

# Forward computation
z = (x * y).sum()
print(f"z = {z.data}")     # 23.0

# Backward pass (automatic differentiation)
z.backward()
print(f"dz/dx = {x.grad}")  # [4.0, 5.0]
print(f"dz/dy = {y.grad}")  # [2.0, 3.0]
```

### Layers

```python
# Linear (fully connected)
linear = sv.nn.Linear(in_features=784, out_features=128)

# Conv2d
conv = sv.nn.Conv2d(in_channels=3, out_channels=64,
                     kernel_size=3, stride=1, padding=1)

# Batch normalization
bn = sv.nn.BatchNorm1d(num_features=128)
bn2d = sv.nn.BatchNorm2d(num_features=64)

# Dropout
dropout = sv.nn.Dropout(p=0.5)

# Recurrent
lstm = sv.nn.LSTM(input_size=50, hidden_size=128)
gru = sv.nn.GRU(input_size=50, hidden_size=128)

# Attention
mha = sv.nn.MultiHeadAttention(embed_dim=512, num_heads=8)

# Embedding
embed = sv.nn.Embedding(num_embeddings=10000, embedding_dim=128)

# Layer norm
ln = sv.nn.LayerNorm(normalized_shape=128)
```

### Activations

```python
sv.nn.relu(x)
sv.nn.sigmoid(x)
sv.nn.tanh(x)
sv.nn.softmax(x, dim=-1)
sv.nn.gelu(x)
sv.nn.leaky_relu(x, alpha=0.01)
```

### Building Models

```python
# Sequential model (simple)
model = sv.nn.Sequential([
    sv.nn.Linear(784, 256),
    sv.nn.ReLU(),
    sv.nn.Dropout(0.3),
    sv.nn.Linear(256, 128),
    sv.nn.ReLU(),
    sv.nn.Dropout(0.3),
    sv.nn.Linear(128, 10),
])

# Forward pass
output = model.forward(input_tensor)
```

### Loss Functions

```python
# Mean squared error (regression)
loss = sv.nn.mse_loss(predictions, targets)

# Cross entropy (multi-class classification)
loss = sv.nn.cross_entropy_loss(logits, labels)

# Binary cross entropy
loss = sv.nn.bce_loss(predictions, targets)

# Huber loss (robust regression)
loss = sv.nn.huber_loss(predictions, targets, delta=1.0)

# Focal loss (imbalanced classification)
loss = sv.nn.focal_loss(predictions, targets, alpha=0.25, gamma=2.0)
```

### Optimizers

```python
# SGD with momentum
optimizer = sv.nn.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer = sv.nn.Adam(model.parameters(), lr=0.001,
                        betas=(0.9, 0.999))

# AdamW (weight decay)
optimizer = sv.nn.AdamW(model.parameters(), lr=0.001,
                         weight_decay=0.01)
```

### Learning Rate Schedulers

```python
scheduler = sv.nn.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = sv.nn.CosineAnnealingLR(optimizer, T_max=100)
scheduler = sv.nn.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
```

### Training Loop

```python
# Complete training example
model = sv.nn.Sequential([
    sv.nn.Linear(784, 128),
    sv.nn.ReLU(),
    sv.nn.Dropout(0.2),
    sv.nn.Linear(128, 10),
])

optimizer = sv.nn.Adam(model.parameters(), lr=0.001)
scheduler = sv.nn.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
for epoch in range(50):
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        # Forward pass
        output = model.forward(X_batch)
        loss = sv.nn.cross_entropy_loss(output, y_batch)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch}: loss={total_loss:.4f}")
```

### CNN Example

```python
# Convolutional neural network for image classification
cnn = sv.nn.Sequential([
    # Block 1
    sv.nn.Conv2d(3, 32, kernel_size=3, padding=1),
    sv.nn.BatchNorm2d(32),
    sv.nn.ReLU(),
    sv.nn.MaxPool2d(2),

    # Block 2
    sv.nn.Conv2d(32, 64, kernel_size=3, padding=1),
    sv.nn.BatchNorm2d(64),
    sv.nn.ReLU(),
    sv.nn.MaxPool2d(2),

    # Classifier
    sv.nn.Flatten(),
    sv.nn.Linear(64 * 8 * 8, 128),
    sv.nn.ReLU(),
    sv.nn.Dropout(0.5),
    sv.nn.Linear(128, 10),
])

output = cnn.forward(image_batch)  # (B, 3, 32, 32) → (B, 10)
```

### Data Loading

```python
# DataLoader for batching
dataset = list(zip(X_train, y_train))
dataloader = sv.nn.DataLoader(dataset, batch_size=32, shuffle=True)

for X_batch, y_batch in dataloader:
    output = model.forward(X_batch)
    loss = sv.nn.cross_entropy_loss(output, y_batch)
    # ...
```

### ONNX Model Loading

```python
# Load and run pre-trained ONNX model
model = sv.nn.load_onnx("resnet50.onnx")
output = model.forward(input_tensor)

# With optimization (Conv-BatchNorm fusion)
model = sv.nn.load_onnx("model.onnx", optimize=True)
```

### Weight Saving and Loading

```python
# Save model weights
sv.nn.save_weights(model, "model_weights.bin")
sv.nn.load_weights(model, "model_weights.bin")

# SafeTensors format (safer, no pickle)
sv.nn.save_safetensors(model, "model.safetensors")
sv.nn.load_safetensors(model, "model.safetensors")
```

### Graph Neural Networks

```python
# GCN layer
gcn = sv.nn.GCNConv(in_features=64, out_features=32)
out = gcn.forward(node_features, adjacency_matrix)

# GAT (Graph Attention)
gat = sv.nn.GATConv(in_features=64, out_features=32, num_heads=4)

# GraphSAGE
sage = sv.nn.SAGEConv(in_features=64, out_features=32)
```

---

## 16. Reinforcement Learning

> **Replaces:** Stable-Baselines3, Gymnasium

### Environments

```python
import pyscivex as sv

# Built-in environments (Gym-compatible API)
env = sv.rl.CartPole()
env = sv.rl.MountainCar()
env = sv.rl.GridWorld(width=10, height=10)

# Environment API
obs = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

### DQN (Deep Q-Network)

```python
config = sv.rl.DQNConfig(
    hidden_dim=64,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_size=10000,
    batch_size=32,
    target_update_freq=100,
)

agent = sv.rl.DQN(env, config)
agent.learn(n_steps=50000)

# Use trained agent
obs = env.reset()
action = agent.predict(obs)
```

### PPO (Proximal Policy Optimization)

```python
config = sv.rl.PPOConfig(
    hidden_dim=64,
    learning_rate=0.0003,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    n_epochs=10,
    batch_size=64,
)

agent = sv.rl.PPO(env, config)
agent.learn(n_steps=100000)
```

### SAC (Soft Actor-Critic)

```python
config = sv.rl.SACConfig(
    hidden_dim=256,
    learning_rate=0.0003,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    buffer_size=100000,
    batch_size=256,
)

agent = sv.rl.SAC(env, config)
agent.learn(n_steps=100000)
```

### TD3 (Twin Delayed DDPG)

```python
config = sv.rl.TD3Config(
    hidden_dim=256,
    learning_rate=0.001,
    gamma=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_delay=2,
)

agent = sv.rl.TD3(env, config)
agent.learn(n_steps=100000)
```

### Hindsight Experience Replay

```python
buffer = sv.rl.HerReplayBuffer(
    capacity=100000,
    goal_strategy="final",    # "final", "future", "episode"
)
```

### Episode Logging

```python
logger = sv.rl.EpisodeLogger()

for episode in range(1000):
    obs = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    logger.log(episode_reward)

# Export training curves
logger.to_csv("training_log.csv")
print(logger.summary())
```

---

## 17. GPU Acceleration

> **Replaces:** CuPy, torch.cuda

### GPU Device Management

```python
import pyscivex as sv

# Auto-detect GPU (Vulkan/Metal/DX12)
device = sv.gpu.Device()
print(device.info())   # "Apple M2 Pro (Metal)"

# Check availability
if sv.gpu.is_available():
    print("GPU available!")
```

### GPU Tensors

```python
# Create GPU tensor
g = sv.gpu.Tensor([1.0, 2.0, 3.0, 4.0], shape=(2, 2))

# Move CPU tensor to GPU
cpu_tensor = sv.Tensor([[1.0, 2.0], [3.0, 4.0]])
gpu_tensor = cpu_tensor.to_gpu()

# Move back to CPU
cpu_tensor = gpu_tensor.to_cpu()
```

### GPU Operations

```python
a = sv.gpu.Tensor.ones((1024, 1024))
b = sv.gpu.Tensor.ones((1024, 1024))

# Matrix multiply (GPU-accelerated)
c = sv.gpu.matmul(a, b)

# Element-wise operations
d = sv.gpu.add(a, b)
e = sv.gpu.mul(a, b)

# Reductions
s = sv.gpu.sum(a)
m = sv.gpu.mean(a)
```

### GPU Training

```python
# GPU-accelerated optimizer
optimizer = sv.gpu.Adam(model.parameters(), lr=0.001)

# Training on GPU
for X_batch, y_batch in dataloader:
    X_gpu = X_batch.to_gpu()
    y_gpu = y_batch.to_gpu()

    output = model.forward(X_gpu)
    loss = sv.nn.cross_entropy_loss(output, y_gpu)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 18. Visualization & Plotting

> **Replaces:** matplotlib, seaborn, plotly

### Basic Plots

```python
import pyscivex as sv

# Line plot
fig = sv.Figure()
fig.line(x, y, label="train loss")
fig.line(x, y2, label="val loss")
fig.title("Training Progress")
fig.x_label("Epoch")
fig.y_label("Loss")
fig.legend()
fig.save_svg("training.svg")

# In Jupyter — auto-renders
fig  # displays inline via _repr_svg_()
```

### Scatter Plot

```python
fig = sv.Figure()
fig.scatter(x, y, label="data points", color="blue", marker="o")
fig.scatter(x2, y2, label="outliers", color="red", marker="x")
fig.title("Scatter Plot")
fig
```

### Bar Chart

```python
fig = sv.Figure()
fig.bar(categories, values, color="steelblue")
fig.title("Sales by Region")
fig.x_label("Region")
fig.y_label("Revenue ($)")
fig
```

### Histogram

```python
fig = sv.Figure()
fig.hist(data, bins=30, color="green", alpha=0.7)
fig.title("Distribution of Values")
fig
```

### Statistical Plots

```python
# Box plot
fig = sv.Figure()
fig.boxplot([group_a, group_b, group_c],
            labels=["A", "B", "C"])
fig

# Violin plot
fig = sv.Figure()
fig.violin([group_a, group_b],
           labels=["Treatment", "Control"])
fig

# Heatmap
fig = sv.Figure()
fig.heatmap(correlation_matrix,
            x_labels=columns, y_labels=columns,
            colormap="viridis", annotate=True)
fig

# QQ plot
fig = sv.Figure()
fig.qqplot(data)
fig

# Regression plot
fig = sv.Figure()
fig.regplot(x, y, ci=0.95)
fig
```

### Pie Chart

```python
fig = sv.Figure()
fig.pie(values, labels=["A", "B", "C", "D"],
        colors=["#ff6384", "#36a2eb", "#ffce56", "#4bc0c0"])
fig
```

### Contour and Surface Plots

```python
# Contour plot
fig = sv.Figure()
fig.contour(X, Y, Z, levels=20, colormap="coolwarm")
fig

# 3-D surface plot
fig = sv.Figure()
fig.surface(X, Y, Z, colormap="viridis")
fig
```

### Grammar of Graphics (Vega-Lite style)

```python
# Declarative plotting API
chart = sv.Chart(df) \
    .mark_point() \
    .encode(
        x="age",
        y="salary",
        color="department",
        size="experience",
    )
chart  # auto-renders in Jupyter

# Line chart
chart = sv.Chart(df) \
    .mark_line() \
    .encode(x="date", y="value", color="series")

# Bar chart
chart = sv.Chart(df) \
    .mark_bar() \
    .encode(x="category", y="count", color="group")
```

### Styling and Themes

```python
# Set theme
sv.viz.set_theme("dark")       # dark, minimal, publication, default

# Custom colors
fig = sv.Figure()
fig.line(x, y, color="#ff6384", linewidth=2.0, linestyle="dashed")

# Multiple y-axes, grid, etc.
fig.grid(True)
fig.xlim(0, 100)
fig.ylim(-1, 1)
```

### Rendering and Export

```python
# SVG (vector — best for publications)
fig.save_svg("plot.svg")

# HTML (interactive in browser)
fig.save_html("plot.html")

# Terminal (braille art)
fig.to_terminal()
# ⠀⠀⠀⠀⠀⠀⣠⣶⣿⣿⣿⣶⣄⡀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⠀

# PNG (rasterized)
fig.save_png("plot.png", width=800, height=600)
```

### Animation

```python
anim = sv.viz.Animation()
for frame_data in simulation:
    fig = sv.Figure()
    fig.scatter(frame_data.x, frame_data.y)
    fig.title(f"Step {frame_data.step}")
    anim.add_frame(fig)

anim.save_gif("simulation.gif", fps=30)
```

---

## 19. ML Pipelines & AutoML

### Building Pipelines

```python
import pyscivex as sv

# Scikit-learn-style pipeline
pipe = sv.ml.Pipeline([
    ("scaler", sv.ml.StandardScaler()),
    ("pca", sv.ml.PCA(n_components=50)),
    ("model", sv.ml.RandomForestClassifier(n_trees=100, max_depth=10)),
])

# Fit and predict
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
score = pipe.score(X_test, y_test)
print(f"Pipeline accuracy: {score:.4f}")

# Cross-validate the whole pipeline
scores = sv.ml.cross_val_score(pipe, X, y, cv=5)
```

### Hyperparameter Search

```python
# Grid search
param_grid = {
    "model__n_trees": [50, 100, 200],
    "model__max_depth": [5, 10, 20],
    "pca__n_components": [20, 50, 100],
}

grid = sv.ml.GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params}")
print(f"Best score: {grid.best_score:.4f}")
best_model = grid.best_estimator

# Random search
random_search = sv.ml.RandomizedSearchCV(
    pipe, param_distributions=param_grid,
    n_iter=20, cv=5, scoring="accuracy", random_state=42
)
random_search.fit(X_train, y_train)
```

### AutoML (Automated Pipeline Optimization)

```python
# Automatic model selection and hyperparameter tuning
auto = sv.ml.PipelineOptimizer(
    transformers=[sv.ml.StandardScaler, sv.ml.MinMaxScaler, sv.ml.PCA],
    predictors=[sv.ml.RandomForestClassifier, sv.ml.GradientBoostingClassifier,
                sv.ml.SVC, sv.ml.LogisticRegression],
    cv=5,
    n_iter=50,
    scoring="f1",
    seed=42,
)

result = auto.search(X_train, y_train)
print(f"Best pipeline: {result.best_pipeline}")
print(f"Best score: {result.best_score:.4f}")

# Use the best pipeline
best = result.best_pipeline
predictions = best.predict(X_test)
```

### Feature Selection

```python
# Select K best features
selector = sv.ml.SelectKBest(k=10, scoring="mutual_info")
selector.fit(X_train, y_train)
X_selected = selector.transform(X_train)

# Recursive feature elimination
rfe = sv.ml.RFE(estimator=sv.ml.RandomForestClassifier(), n_features=20)
rfe.fit(X_train, y_train)
selected = rfe.support   # boolean mask of selected features
```

### Model Explainability

```python
# Permutation importance
importance = sv.ml.permutation_importance(model, X_test, y_test, n_repeats=10)
print(importance)

# Partial dependence plots
pdp = sv.ml.partial_dependence(model, X_train, feature=0)

# LIME-style local explanations
explanation = sv.ml.lime(model, X_test[0], X_train, num_features=10)

# SHAP-style (Kernel SHAP)
shap_values = sv.ml.kernel_shap(model, X_test[0], X_train)
```

---

## 20. Model Deployment & Interop

### ONNX Import and Inference

```python
import pyscivex as sv

# Load pre-trained ONNX model
model = sv.nn.load_onnx("resnet50.onnx")

# Prepare input
img = sv.image.open("cat.jpg").resize(224, 224)
input_tensor = sv.Tensor(img.to_numpy()).unsqueeze(0)  # (1, 3, 224, 224)

# Run inference
output = model.forward(input_tensor)
predicted_class = output.argmax()
print(f"Predicted class: {predicted_class}")
```

### REST API Serving

```python
# FastAPI example with pyscivex model
from fastapi import FastAPI
import pyscivex as sv

app = FastAPI()
model = sv.nn.load_onnx("model.onnx")

@app.post("/predict")
async def predict(features: list[float]):
    input_tensor = sv.Tensor(features).reshape((1, -1))
    output = model.forward(input_tensor)
    return {"prediction": output.to_list()}
```

### Flask API Serving

```python
from flask import Flask, request, jsonify
import pyscivex as sv

app = Flask(__name__)
model = sv.ml.load("trained_model.svex")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    X = sv.Tensor(data).reshape((1, -1))
    prediction = model.predict(X).to_list()
    return jsonify({"prediction": prediction})
```

### Model Save/Load

```python
# Save scikit-learn-style model
model.save("classifier.svex")

# Load anywhere
model = sv.ml.load("classifier.svex")

# Save neural network (SafeTensors format)
sv.nn.save_safetensors(nn_model, "network.safetensors")
sv.nn.load_safetensors(nn_model, "network.safetensors")
```

### WebAssembly (Browser Deployment)

```python
# pyscivex models can be exported for browser use
# See scivex-wasm crate for JavaScript/WASM interop:
#
#   import init, { Tensor, LinearRegression } from 'scivex-wasm';
#   await init();
#   const model = new LinearRegression();
#   model.fit(x_train, y_train);
#   const pred = model.predict(x_test);
```

---

## 21. Jupyter Notebook Integration

pyscivex is designed as a Jupyter-first library. All major types have rich display methods.

### Auto-Rendering

```python
import pyscivex as sv

# DataFrames render as HTML tables
df = sv.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
df  # rich HTML table in Jupyter

# Figures render as inline SVG
fig = sv.Figure()
fig.line([1, 2, 3], [4, 5, 6])
fig  # inline SVG plot

# Charts render as inline SVG
sv.Chart(df).mark_point().encode(x="x", y="y")  # inline

# Images render as inline PNG
img = sv.image.open("photo.jpg")
img  # inline image
```

### Display Methods

```python
# DataFrame
df._repr_html_()      # HTML table for Jupyter
str(df)                # text table for terminal

# Figure
fig._repr_svg_()       # SVG for Jupyter
fig.to_terminal()      # braille art for terminal

# Image
img._repr_png_()       # PNG bytes for Jupyter

# Symbolic expressions
expr._repr_latex_()    # LaTeX for Jupyter rendering
```

### Progress Bars

```python
# Training with progress bar (tqdm integration)
from tqdm import tqdm

for epoch in tqdm(range(100), desc="Training"):
    loss = train_one_epoch(model, dataloader)
```

---

## 22. Python Ecosystem Interop (Optional Migration Helpers)

> **You do NOT need any of these libraries.** pyscivex is fully self-contained.
> These converters exist **only** as migration helpers for teams transitioning
> existing codebases from numpy/pandas/scipy/etc. to pyscivex. Once migrated,
> you can remove all external dependencies and use pure pyscivex.

### NumPy

```python
import numpy as np
import pyscivex as sv

# NumPy → pyscivex (zero-copy when C-contiguous)
arr = np.array([[1.0, 2.0], [3.0, 4.0]])
t = sv.Tensor.from_numpy(arr)

# pyscivex → NumPy (zero-copy)
arr = t.numpy()                    # returns np.ndarray

# Accept numpy anywhere a Tensor is expected
model.fit(np.array(X), np.array(y))  # just works
```

### pandas

```python
import pandas as pd
import pyscivex as sv

# pandas → pyscivex
pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
df = sv.DataFrame.from_pandas(pdf)

# pyscivex → pandas
pdf = df.to_pandas()
```

### scipy.sparse

```python
import scipy.sparse as sp
import pyscivex as sv

# scipy → pyscivex
scipy_csr = sp.random(100, 100, density=0.1, format="csr")
sv_csr = sv.sparse.CsrMatrix.from_scipy(scipy_csr)

# pyscivex → scipy
scipy_csr = sv_csr.to_scipy()
```

### pyarrow

```python
import pyarrow as pa
import pyscivex as sv

# pyarrow → pyscivex
table = pa.table({"x": [1, 2, 3], "y": [4, 5, 6]})
df = sv.DataFrame.from_arrow(table)

# pyscivex → pyarrow
table = df.to_arrow()
```

### NetworkX

```python
import networkx as nx
import pyscivex as sv

# networkx → pyscivex
nx_g = nx.karate_club_graph()
g = sv.Graph.from_networkx(nx_g)

# pyscivex → networkx
nx_g = g.to_networkx()
```

### SymPy

```python
import sympy
import pyscivex as sv

# SymPy → pyscivex
x_sp = sympy.Symbol("x")
expr_sp = x_sp**2 + 2*x_sp + 1
expr = sv.sym.Expr.from_sympy(expr_sp)

# pyscivex → SymPy
expr_sp = expr.to_sympy()
```

### Gymnasium (RL Environments)

```python
import gymnasium as gym
import pyscivex as sv

# Use Gym environments with pyscivex RL agents
env = gym.make("CartPole-v1")
agent = sv.rl.DQN(env, config)
agent.learn(n_steps=50000)
```

---

## 23. What pyscivex Does NOT Do

pyscivex focuses on being the core data science toolkit. Here's what's **out of scope**:

| Category | What it does NOT do | What to use instead |
|----------|-------------------|-------------------|
| **LLM Inference** | Run GPT/Llama/Claude models | Use HuggingFace Transformers, vLLM, llama.cpp |
| **LLM Frameworks** | Build LLM applications | LangChain, LlamaIndex, Claude SDK |
| **RAG Pipelines** | Retrieval-augmented generation | LangChain, Haystack |
| **Distributed Training** | Multi-GPU / multi-node training | PyTorch DDP, Horovod, DeepSpeed |
| **MLOps** | Model registry, A/B testing | MLflow, Weights & Biases, SageMaker |
| **Data Orchestration** | ETL pipelines, DAGs | Airflow, Prefect, Dagster |
| **Web Frameworks** | HTTP servers, REST APIs | FastAPI, Flask, Django |
| **Database Engines** | Storing/querying data | PostgreSQL, DuckDB, SQLite |
| **Cloud Storage** | S3/GCS/Azure object access | boto3, gcsfs, azure-storage |

### What pyscivex DOES provide as building blocks

- **Tensor operations** — the foundation for all numerical computing
- **Neural network layers** — build custom architectures
- **ONNX runtime** — import and run pre-trained models from any framework
- **Embeddings** — Word2Vec, TF-IDF for text representation
- **Approximate nearest neighbors** — fast vector similarity search
- **Graph neural network layers** — GCN, GAT, GraphSAGE
- **GPU acceleration** — hardware-accelerated tensor operations
- **Visualization** — publication-quality plots and charts

You CAN combine pyscivex with specialized tools:

```python
# Use pyscivex for preprocessing, a specialized tool for the rest
import pyscivex as sv
from transformers import AutoModel

# pyscivex for data loading and preprocessing
df = sv.read_sql("SELECT * FROM documents", conn_str)
texts = df["content"].to_list()

# pyscivex for tokenization
tokenizer = sv.nlp.WordPieceTokenizer(vocab_size=30000)
tokens = [tokenizer.tokenize(text) for text in texts]

# pyscivex for embeddings
tfidf = sv.nlp.TfidfVectorizer()
features = tfidf.fit_transform(texts)

# Use HuggingFace for LLM inference
model = AutoModel.from_pretrained("bert-base-uncased")
```

---

## End-to-End Workflow Example

Here's a complete data science project using only pyscivex:

```python
import pyscivex as sv

# ============================================================
# 1. Data Loading
# ============================================================
df = sv.read_sql(
    "SELECT * FROM customer_data WHERE signup_date > '2024-01-01'",
    "postgresql://user:pass@db.prod.example.com/analytics"
)
print(f"Loaded {df.nrows} rows, {df.ncols} columns")
print(df.describe())

# ============================================================
# 2. Exploratory Data Analysis
# ============================================================
# Summary statistics
print(df[["age", "income", "purchases"]].describe())

# Correlation matrix
corr = df[["age", "income", "purchases", "satisfaction"]].corr()
fig = sv.Figure()
fig.heatmap(corr, annotate=True, colormap="coolwarm")
fig.title("Feature Correlations")
fig.save_svg("correlation.svg")

# Distribution plots
fig = sv.Figure()
fig.hist(df["income"].to_list(), bins=50, color="steelblue")
fig.title("Income Distribution")
fig.save_svg("income_dist.svg")

# ============================================================
# 3. Feature Engineering
# ============================================================
# Rolling features
df["purchases_ma7"] = df["purchases"].rolling(7).mean()

# Categorical encoding
df["region_encoded"] = sv.ml.LabelEncoder().fit_transform(df["region"])

# Standard scaling
scaler = sv.ml.StandardScaler()
numeric_cols = ["age", "income", "purchases", "purchases_ma7"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ============================================================
# 4. Model Training
# ============================================================
X = df[["age", "income", "purchases", "purchases_ma7", "region_encoded"]].to_tensor()
y = df["churned"].to_tensor()

X_train, X_test, y_train, y_test = sv.ml.train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline with multiple models
pipe = sv.ml.Pipeline([
    ("scaler", sv.ml.StandardScaler()),
    ("model", sv.ml.GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1
    )),
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)

# ============================================================
# 5. Evaluation
# ============================================================
print(f"Accuracy: {sv.ml.accuracy_score(y_test, predictions):.4f}")
print(f"F1 Score: {sv.ml.f1_score(y_test, predictions):.4f}")
print(f"AUC-ROC:  {sv.ml.roc_auc_score(y_test, pipe.predict_proba(X_test)):.4f}")
print(sv.ml.classification_report(y_test, predictions))

# Confusion matrix
cm = sv.ml.confusion_matrix(y_test, predictions)
fig = sv.Figure()
fig.heatmap(cm, annotate=True, colormap="blues")
fig.title("Confusion Matrix")
fig.save_svg("confusion_matrix.svg")

# ============================================================
# 6. Cross-Validation
# ============================================================
scores = sv.ml.cross_val_score(pipe, X, y, cv=5, scoring="f1")
print(f"CV F1: {scores.mean():.4f} +/- {scores.std():.4f}")

# ============================================================
# 7. Feature Importance
# ============================================================
importance = sv.ml.permutation_importance(pipe, X_test, y_test, n_repeats=10)
fig = sv.Figure()
fig.bar(feature_names, importance, color="teal")
fig.title("Feature Importance")
fig.save_svg("feature_importance.svg")

# ============================================================
# 8. Save and Deploy
# ============================================================
pipe.save("churn_model.svex")

# Later: load and serve
# model = sv.ml.load("churn_model.svex")
# predictions = model.predict(new_data)
```

---

## Performance: pyscivex vs Pure Python Libraries

pyscivex achieves significant speedups over pure Python libraries because all
computation happens in compiled Rust:

| Operation | NumPy/pandas/sklearn | pyscivex | Speedup |
|-----------|---------------------|----------|---------|
| Matrix multiply (1000x1000) | baseline | Rust BLAS with NEON/AVX | 1-3x |
| DataFrame filter (1M rows) | baseline | zero-copy columnar | 2-5x |
| Random Forest train (100k samples) | baseline | parallel Rust trees | 2-4x |
| FFT (4096 points) | baseline | pure Rust radix-2 | comparable |
| CSV parse (100MB) | baseline | zero-alloc Rust parser | 3-5x |
| K-Means (100k points) | baseline | SIMD distance + parallel | 2-5x |

**Key advantages:**
- Single `pip install` — no compile-time dependencies, no BLAS/LAPACK issues
- Consistent API — no switching between numpy/pandas/sklearn conventions
- Memory efficient — Rust ownership model prevents leaks and copies
- Thread-safe — GIL released during Rust computation
- Cross-platform — same behavior on Linux, macOS (Intel + ARM), Windows

---

*pyscivex v0.1.0 — MIT License*
*Powered by [Scivex](https://github.com/scivex/scivex) (120K+ LOC of pure Rust)*
