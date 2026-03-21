# API Summary

Quick reference for all Scivex crates. For full documentation, run `cargo doc --workspace --no-deps --open`.

## scivex-core — Tensors & Linear Algebra

**Import:** `use scivex_core::prelude::*;`

### Tensor Creation

| Function | Description |
|----------|-------------|
| `Tensor::from_vec(data, shape)` | Create from `Vec<T>` with given shape |
| `Tensor::from_slice(data, shape)` | Create from `&[T]` with given shape |
| `Tensor::zeros(shape)` | All zeros |
| `Tensor::ones(shape)` | All ones |
| `Tensor::full(shape, value)` | Filled with a constant |
| `Tensor::eye(n)` | Identity matrix |
| `Tensor::arange(n)` | Range `[0, n)` |
| `Tensor::linspace(start, end, n)` | Evenly spaced values |
| `Tensor::scalar(value)` | 0-d scalar tensor |

### Tensor Methods

| Method | Description |
|--------|-------------|
| `shape()` | Shape as `&[usize]` |
| `ndim()` | Number of dimensions |
| `numel()` | Total element count |
| `get(index)` / `set(index, val)` | Element access |
| `as_slice()` / `into_vec()` | Data access |
| `reshape(new_shape)` | Reshape (consuming) |
| `transpose()` | Matrix transpose |
| `permute(axes)` | Arbitrary axis permutation |
| `squeeze()` / `unsqueeze(axis)` | Remove/add length-1 axes |
| `flatten()` | Flatten to 1-D |
| `slice(ranges)` | Sub-tensor by ranges |
| `select(axis, index)` | Select along axis |
| `index_select(axis, indices)` | Gather indices along axis |
| `masked_select(mask)` | Boolean masking |
| `concat(tensors, axis)` | Concatenate tensors |
| `stack(tensors, axis)` | Stack tensors along new axis |
| `map(f)` / `zip_map(other, f)` | Element-wise transformation |
| `cast::<U>()` | Type cast |
| `sort()` / `argsort()` | Sort elements |

### Arithmetic & Reductions

| Operation | Description |
|-----------|-------------|
| `a + b`, `a - b`, `a * b`, `a / b` | Element-wise (same shape or scalar broadcast) |
| `sum()`, `mean()`, `product()` | Global reductions |
| `min_element()`, `max_element()` | Min/max values |
| `sum_axis(axis)` | Reduce along axis |
| `matmul(other)` | Matrix multiplication |

### Linear Algebra (`linalg`)

| Function | Description |
|----------|-------------|
| `dot(x, y)` | Inner product |
| `axpy(alpha, x, y)` | `y += alpha * x` |
| `nrm2(x)` / `asum(x)` | L2 / L1 norms |
| `gemv(...)` / `gemm(...)` | Matrix-vector / matrix-matrix product |
| `solve(a, b)` | Solve `Ax = b` |
| `inv(a)` | Matrix inverse |
| `det(a)` | Determinant |
| `lstsq(a, b)` | Least squares |
| `LuDecomposition` | LU with partial pivoting |
| `QrDecomposition` | QR decomposition |
| `CholeskyDecomposition` | Cholesky decomposition |
| `SvdDecomposition` | Singular value decomposition |
| `EigDecomposition` | Eigenvalue decomposition |

### Sparse Matrices (`linalg::sparse`)

| Type | Description |
|------|-------------|
| `CooMatrix` | Coordinate format |
| `CsrMatrix` | Compressed sparse row |
| `CscMatrix` | Compressed sparse column |

### Numeric Traits

| Trait | Key Methods |
|-------|-------------|
| `Scalar` | `zero()`, `one()`, `from_usize()` |
| `Float` | `pi()`, `epsilon()`, `abs()`, `sqrt()`, `sin()`, `cos()`, `exp()`, `ln()`, `powf()` |
| `Integer` | `rem()` |
| `Real` | Real-valued subset of Float |

### Other Modules

| Module | Contents |
|--------|----------|
| `complex` | `Complex<T>` type |
| `fft` | FFT, IFFT, RFFT |
| `random` | Pseudo-random number generation |
| `promote` | `DType`, `CastFrom`, type promotion |

---

## scivex-frame — DataFrames

**Import:** `use scivex_frame::prelude::*;`

### Types

| Type | Description |
|------|-------------|
| `DataFrame` | Columnar data frame with type-erased columns |
| `DataFrameBuilder` | Builder for constructing DataFrames |
| `Series<T>` | Generic typed column |
| `StringSeries` | String column |
| `CategoricalSeries` | Categorical column |
| `DateTimeSeries` | DateTime column |
| `DateTime` / `Duration` | Temporal types |
| `RollingWindow` | Rolling window aggregations |
| `GroupBy` | Group-by-aggregate operations |
| `AggFunc` | Aggregation function enum (Sum, Mean, Count, Min, Max, Std, Var) |
| `LazyFrame` | Lazy evaluation with expression trees |
| `MultiIndex` | Hierarchical indexing |
| `JoinType` | Join types (Inner, Left, Right, Outer) |
| `SqlContext` / `sql!` | SQL query engine on DataFrames |
| `DType` | Column type introspection |

### DataFrame Operations

```rust
DataFrame::builder().add_column("x", vec![1, 2, 3]).build()?;
df.column("name")?;          // get column
df.shape();                    // (rows, cols)
df.join(&other, ..., JoinType::Inner)?;
df.groupby(&["col"])?;
```

### Parallel (with `parallel` feature)

`par_apply()`, `par_filter()`, `par_groupby_agg()`, `par_sort()`

---

## scivex-stats — Statistics

**Import:** `use scivex_stats::prelude::*;`

### Descriptive Statistics

`mean()`, `variance()`, `std_dev()`, `median()`, `quantile()`, `skewness()`, `kurtosis()`, `describe()`

### Distributions

`Normal`, `Uniform`, `Exponential`, `Beta`, `Gamma`, `Binomial`, `Poisson`, `StudentT`, `ChiSquared`, `Cauchy`, `Laplace`, `LogNormal`, `Pareto`, `Weibull`, `Bernoulli`, `NegativeBinomial`, `Hypergeometric`

All implement `Distribution` trait: `pdf()`, `cdf()`, `sample()`, `mean()`, `variance()`

### Hypothesis Tests

`t_test_one_sample()`, `t_test_two_sample()`, `chi_square_test()`, `anova_oneway()`, `mann_whitney_u()`, `ks_test_two_sample()`

### Correlation

`pearson()`, `spearman()`, `kendall()`, `corr_matrix()`

### Regression

`ols()` (OLS), `glm()` (Generalized Linear Model with `Family` / `LinkFunction`)

### Effect Size

`cohens_d()`, `hedges_g()`, `glass_delta()`, `eta_squared()`, `omega_squared()`, `cramers_v()`, `cohens_w()`, `point_biserial()`

### Confidence Intervals

`ci_mean()`, `ci_mean_z()`, `ci_proportion()`

### Time Series

`Arima`, `Sarimax`, `ExponentialSmoothing`, `acf()`, `pacf()`, `adf_test()`, `seasonal_decompose()`

### Survival Analysis

`kaplan_meier()`, `cox_ph()`, `log_rank_test()`, `median_survival_time()`

### Bayesian

`MetropolisHastings`, `HamiltonianMC`, `McmcConfig`, `rhat()`, `effective_sample_size()`

### Other

`Garch` (volatility), `KalmanFilter`, `VarModel` (vector autoregression), `bonferroni()`, `benjamini_hochberg()`

---

## scivex-optim — Optimization

**Import:** `use scivex_optim::prelude::*;`

### Root Finding

`bisection()`, `newton()`, `brent_root()` — all return `RootResult`

### 1-D Minimization

`golden_section()`, `brent_min()` — return `Minimize1dResult`

### Multi-D Minimization

| Function | Description |
|----------|-------------|
| `gradient_descent()` | Basic gradient descent |
| `bfgs()` | Quasi-Newton BFGS |
| `lbfgsb()` | L-BFGS-B with box constraints |
| `nelder_mead()` | Derivative-free simplex |
| `numerical_gradient()` | Finite-difference gradient |

Options: `MinimizeOptions`, `Bounds`. Returns `MinimizeResult`.

### Integration

`trapezoid()`, `simpson()`, `quad()` — return `QuadResult`

### Interpolation

`interp1d()`, `interp2d()`, `CubicSpline`, `BSpline`, `Linear1d`, `Bilinear2d`, `Bicubic2d`

### ODE Solvers

`euler()`, `rk45()`, `bdf2()`, `solve_ivp()` — return `OdeResult`

### PDE Solvers

`heat_equation_1d()`, `wave_equation_1d()`, `laplace_2d()` — return `PdeResult`

### Linear / Quadratic Programming

`linprog()` (revised simplex), `quadprog()` (active set)

### Curve Fitting

`curve_fit()`, `levenberg_marquardt()` — return `LeastSquaresResult`

---

## scivex-io — Data I/O

**Import:** `use scivex_io::prelude::*;`

All modules are feature-gated.

| Feature | Functions / Types |
|---------|-------------------|
| `csv` (default) | `read_csv()`, `read_csv_path()`, `write_csv()`, `CsvReaderBuilder`, `CsvWriterBuilder` |
| `json` | `read_json()`, `read_json_path()`, `write_json()`, `JsonReaderBuilder`, `JsonWriterBuilder` |
| `parquet` | `read_parquet()`, `write_parquet()`, `ParquetReaderBuilder`, `ParquetWriterBuilder` |
| `arrow` | `read_arrow()`, `write_arrow()`, tensor conversions |
| `npy` | `read_npy()`, `write_npy()`, `read_npz()`, `write_npz()` |
| `excel` | `read_excel()`, `write_excel()`, `ExcelReaderBuilder` |
| `hdf5` | `read_hdf5_dataset()`, `write_hdf5_dataset()`, `list_hdf5_datasets()` |
| `avro` | `read_avro()`, `write_avro()` |
| `orc` | `read_orc()`, `write_orc()` |
| `sqlite` | `SqliteConnection` |
| `postgres` | `PostgresConnection` |
| `mysql` | `MysqlConnection` |
| `mssql` | `MssqlConnection`, `MssqlConfig` |
| `duckdb` | `DuckDbConnection` |
| `mmap` | `MmapTensorReader`, `mmap_npy()` |
| `delta` | `read_delta()`, `DeltaSnapshot` |
| `cloud` | `cloud_read_bytes()`, `CloudPath`, `CloudConfig` |

---

## scivex-viz — Visualization

**Import:** `use scivex_viz::prelude::*;`

### Core Types

| Type | Description |
|------|-------------|
| `Figure` | Top-level container, holds `Axes` |
| `Axes` | Plot area with title, labels, ticks |
| `Layout` | Grid layout for subplots |
| `Element` | Backend-agnostic drawing primitive |

### Plot Builders

`LinePlot`, `ScatterPlot`, `BarPlot`, `Histogram`, `AreaPlot`, `BoxPlotBuilder`, `ViolinPlot`, `HeatmapBuilder`, `ContourPlot`, `PieChart`, `PolarPlot`, `ErrorBarPlot`, `ConfidenceBand`, `SurfacePlot`

### Statistical Plots

`RegressionPlot`, `ResidualPlot`, `QQPlot`, `CorrelationHeatmap`

### Composite Plots

`PairPlot` (n*n scatter matrix), `JointPlot` (scatter + marginal histograms)

### Styling

`Color`, `ColorMap`, `Theme`, `Fill`, `Stroke`, `Font`, `Marker`, `MarkerShape`

### Backends

`SvgBackend`, `TerminalBackend`, `HtmlBackend`, `BitmapBackend`

### Output

```rust
fig.to_svg()?;           // SVG string
fig.save_svg("out.svg")?; // Save to file
fig.to_terminal()?;       // Terminal braille art
fig.to_html()?;           // Interactive HTML
```

### Special Features

- `Animation` — GIF89a export with `to_gif()` / `save_gif()`
- `SurfacePlot` — 3D surface with isometric projection
- `parse_latex()` / `contains_math()` — LaTeX math in labels
- `Layout::weighted_grid()` — Proportional subplot sizing
- `Figure::share_x()` / `share_y()` — Shared axes across subplots

---

## scivex-ml — Machine Learning

**Import:** `use scivex_ml::prelude::*;`

### Traits

| Trait | Methods |
|-------|---------|
| `Transformer` | `fit()`, `transform()`, `fit_transform()` |
| `Predictor` | `fit()`, `predict()` |
| `Classifier` | extends `Predictor` + `predict_proba()` |

### Models

| Category | Types |
|----------|-------|
| Linear | `LinearRegression`, `Ridge`, `LogisticRegression` |
| Trees | `DecisionTreeClassifier`, `DecisionTreeRegressor` |
| Ensembles | `RandomForestClassifier/Regressor`, `GradientBoostingClassifier/Regressor`, `HistGradientBoostingClassifier/Regressor` |
| SVM | `SVC`, `SVR`, `Kernel` |
| Neighbors | `KNNClassifier`, `KNNRegressor`, `DistanceMetric` |
| Clustering | `KMeans`, `DBSCAN`, `AgglomerativeClustering` |
| Naive Bayes | `GaussianNB` |
| Reduction | `PCA`, `TruncatedSVD`, `TSNE` |

### Preprocessing

`StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OneHotEncoder`

### Pipelines

`Pipeline`, `FeatureUnion`, `ColumnTransformer`

### Model Selection

`train_test_split()`, `KFold`, `cross_val_score()`, `grid_search_cv()`, `random_search_cv()`

### Metrics (in `metrics` module)

Classification: accuracy, precision, recall, F1, confusion matrix, ROC-AUC
Regression: MSE, RMSE, MAE, R2, MAPE

### Explainability

`permutation_importance()`, `partial_dependence()`, `lime()`, `kernel_shap()`, `tree_shap()`

### Online Learning

`SGDClassifier`, `SGDRegressor`, `OnlineKMeans`, `StreamingMean`, `StreamingVariance`

### Persistence

`Persistable` trait — `save()` / `load()` for all models

---

## scivex-nn — Neural Networks

**Import:** `use scivex_nn::prelude::*;`

### Autograd

`Variable<T>` — computation graph node supporting reverse-mode autodiff via `.backward()`

### Layers

| Layer | Description |
|-------|-------------|
| `Linear` | Fully connected |
| `Conv1d`, `Conv2d`, `Conv3d` | Convolution |
| `BatchNorm1d`, `BatchNorm2d` | Batch normalization |
| `LayerNorm` | Layer normalization |
| `SimpleRNN`, `LSTM`, `GRU` | Recurrent |
| `MultiHeadAttention` | Multi-head attention |
| `TransformerEncoderLayer`, `TransformerDecoderLayer` | Transformer blocks |
| `MaxPool1d/2d`, `AvgPool1d/2d` | Pooling |
| `Dropout` | Dropout regularization |
| `Embedding` | Embedding lookup |
| `Flatten` | Reshape to 1-D |
| `Sequential` | Sequential container |
| `ReLU`, `Sigmoid`, `Tanh` | Activation layers |

### Loss Functions

`mse_loss()`, `cross_entropy_loss()`, `bce_loss()`, `focal_loss()`, `huber_loss()`, `smooth_l1_loss()`, `kl_divergence()`, `hinge_loss()`

### Optimizers

`SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`

### LR Schedulers

`StepLR`, `ExponentialLR`, `LinearLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, `WarmupCosineDecay`

### Initialization

`xavier_normal()`, `xavier_uniform()`, `kaiming_normal()`, `kaiming_uniform()`

### Training Utilities

`Trainer`, `TrainingHistory`, `EarlyStopping`, `ModelCheckpoint`, `LossLogger`, `LrFinder`, `clip_grad_norm()`, `clip_grad_value()`

### Data

`Dataset` trait, `TensorDataset`, `DataLoader`

### Persistence & Serialization

`save_weights()` / `load_weights()`, `save_safetensors()` / `load_safetensors()`, `save_gguf()` / `load_gguf()`

### ONNX

`load_onnx()`, `OnnxModel`, `OnnxInferenceSession`

---

## scivex-signal — Signal Processing

**Import:** `use scivex_signal::prelude::*;`

| Module | Functions |
|--------|-----------|
| `window` | Hann, Hamming, Blackman, Bartlett |
| `filter` | FIR design, `lfilter()`, `filtfilt()` |
| `spectral` | STFT, spectrogram, periodogram, Welch |
| `resample` | `resample()`, `decimate()`, `interpolate()` |
| `peak` | `find_peaks()`, prominence |
| `convolution` | `convolve()`, `ConvolveMode` |
| `wavelet` | DWT/IDWT, `Wavelet` |
| `audio` | WAV read/write |
| `features` | `mel_spectrogram()`, `mfcc()`, `chroma_stft()`, `pitch_yin()` |

---

## scivex-image — Image Processing

**Import:** `use scivex_image::prelude::*;`

| Type / Function | Description |
|-----------------|-------------|
| `Image<T>` | Core image type |
| `PixelFormat` | RGB, Grayscale, HSV |
| Color | `grayscale()`, `rgb_to_hsv()`, `hsv_to_rgb()` |
| Transforms | `resize()`, `crop()`, `flip()`, `rotate()`, `pad()` |
| Filters | `gaussian_blur()`, `sobel_edges()`, convolution |
| Histogram | `histogram()`, `equalize()` |
| Morphology | `erode()`, `dilate()`, `opening()`, `closing()` |
| Features | `harris_corners()`, `fast_features()`, `OrbDetector` |
| Matching | `BruteForceMatcher`, `FlannMatcher` |
| Segmentation | `connected_components()`, `region_growing()`, `watershed()` |
| Hough | `HoughLine`, `HoughCircle` |

---

## scivex-graph — Graphs

**Import:** `use scivex_graph::prelude::*;`

| Type / Function | Description |
|-----------------|-------------|
| `Graph<N, E>` | Undirected weighted graph |
| `DiGraph<N, E>` | Directed weighted graph |
| Traversals | BFS, DFS, topological sort |
| Shortest paths | Dijkstra, Bellman-Ford, Floyd-Warshall |
| Connectivity | Connected/strongly-connected components |
| Centrality | Degree, betweenness, PageRank |
| MST | Kruskal, Prim |
| Community | Label propagation |
| Flow | Max-flow, bipartite matching |

---

## scivex-nlp — NLP

**Import:** `use scivex_nlp::prelude::*;`

| Type / Function | Description |
|-----------------|-------------|
| Tokenizers | `WhitespaceTokenizer`, `WordTokenizer`, `CharTokenizer`, `NGramTokenizer`, `WordPieceTokenizer`, `UnigramTokenizer` |
| `PorterStemmer` | Porter stemming |
| Vectorization | `CountVectorizer`, `TfidfVectorizer` |
| Embeddings | `WordEmbeddings`, `Word2VecModel`, `Word2VecTrainer` |
| Similarity | `cosine_similarity()`, `jaccard_similarity()`, `edit_distance_normalized()` |
| Sentiment | `SentimentAnalyzer` |
| POS Tagging | `HmmPosTagger`, `PosTag` |
| NER | `RuleBasedNer`, `Entity`, `EntityType` |
| Topics | `LdaModel`, `LdaConfig` |

---

## scivex-sym — Symbolic Math

**Import:** `use scivex_sym::prelude::*;`

| Function / Type | Description |
|-----------------|-------------|
| `Expr` | Symbolic expression AST |
| `var(name)` | Create variable |
| `constant(val)` | Create constant |
| `sin()`, `cos()`, `tan()`, `exp()`, `ln()`, `sqrt()`, `abs()` | Math functions |
| `pi()`, `e()`, `zero()`, `one()` | Constants |
| `diff(expr, var)` | Symbolic derivative |
| `diff_n(expr, var, n)` | nth derivative |
| `integrate(expr, var)` | Symbolic integral |
| `definite_integral(expr, var, a, b)` | Definite integral |
| `simplify(expr)` | Algebraic simplification |
| `expand(expr)` | Expand products |
| `factor_out(expr)` | Factor common terms |
| `solve_linear(expr, var)` | Solve linear equation |
| `solve_quadratic(expr, var)` | Solve quadratic equation |
| `taylor(expr, var, point, n)` | Taylor series |
| `maclaurin(expr, var, n)` | Maclaurin series |
| `Polynomial` | Coefficient-based polynomial |
