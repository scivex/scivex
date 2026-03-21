# Type Mapping: Python to Scivex

This reference maps common Python data science types to their Scivex Rust equivalents.

## Primitive Types

| Python | Scivex Rust |
|--------|-------------|
| `int` | `i32`, `i64` |
| `float` | `f64` |
| `complex` | `Complex<f64>` |
| `bool` | `bool` |
| `str` | `String` / `&str` |
| `None` | `Option<T>::None` |
| `list[T]` | `Vec<T>` |
| `dict[K, V]` | `HashMap<K, V>` |
| `tuple[A, B]` | `(A, B)` |

## NumPy to scivex-core

| NumPy | Scivex |
|-------|--------|
| `numpy.ndarray` | `Tensor<T>` |
| `numpy.float32` | `f32` as `T` in `Tensor<f32>` |
| `numpy.float64` | `f64` as `T` in `Tensor<f64>` |
| `numpy.int32` | `i32` as `T` in `Tensor<i32>` |
| `numpy.int64` | `i64` as `T` in `Tensor<i64>` |
| `numpy.complex128` | `Complex<f64>` |
| `numpy.bool_` | `bool` |
| `numpy.dtype` | `DType` enum |
| `numpy.linalg` functions | `scivex_core::linalg::*` |
| `numpy.fft` functions | `scivex_core::fft::*` |
| `numpy.random` | `scivex_core::random::*` |

### Tensor Shape Comparison

| NumPy | Scivex |
|-------|--------|
| `a.shape` | `a.shape()` returns `&[usize]` |
| `a.ndim` | `a.ndim()` |
| `a.size` | `a.numel()` |
| `a.dtype` | Type parameter `T` (compile-time) |
| `a.reshape(2, 3)` | `a.reshape(vec![2, 3])` |
| `a.T` | `a.transpose()` |
| `a.flatten()` | `a.flatten()` |
| `a.astype(np.float32)` | `a.cast::<f32>()` |

## Pandas to scivex-frame

| Pandas | Scivex |
|--------|--------|
| `pandas.DataFrame` | `DataFrame` |
| `pandas.Series` (int) | `Series<i64>` |
| `pandas.Series` (float) | `Series<f64>` |
| `pandas.Series` (str) | `StringSeries` |
| `pandas.Series` (bool) | `Series<bool>` |
| `pandas.Categorical` | `CategoricalSeries` |
| `pandas.Timestamp` | `DateTime` |
| `pandas.Timedelta` | `Duration` |
| `pandas.DatetimeIndex` | `DateTimeSeries` |
| `pandas.MultiIndex` | `MultiIndex` |
| `pandas.GroupBy` | `GroupBy` |
| `pandas.api.types.CategoricalDtype` | `DType::Categorical` |
| `pandas.NaT` / `numpy.nan` | `Option<T>::None` (typed columns) |

### DataFrame Operation Comparison

| Pandas | Scivex |
|--------|--------|
| `pd.DataFrame({"a": [1,2]})` | `DataFrame::builder().add_column("a", vec![1,2]).build()?` |
| `df["col"]` | `df.column("col")?` |
| `df.shape` | `df.shape()` |
| `df.groupby("col")` | `df.groupby(&["col"])?` |
| `df.merge(other, on="col")` | `df.join(&other, ..., JoinType::Inner)?` |
| `df.to_csv("f.csv")` | `write_csv(&df, "f.csv")?` |
| `pd.read_csv("f.csv")` | `read_csv_path("f.csv")?` |

## scikit-learn to scivex-ml

| sklearn | Scivex |
|---------|--------|
| `sklearn.base.BaseEstimator` | `Predictor` / `Transformer` traits |
| `sklearn.base.ClassifierMixin` | `Classifier` trait |
| `sklearn.linear_model.LinearRegression` | `LinearRegression` |
| `sklearn.linear_model.Ridge` | `Ridge` |
| `sklearn.linear_model.LogisticRegression` | `LogisticRegression` |
| `sklearn.tree.DecisionTreeClassifier` | `DecisionTreeClassifier` |
| `sklearn.tree.DecisionTreeRegressor` | `DecisionTreeRegressor` |
| `sklearn.ensemble.RandomForestClassifier` | `RandomForestClassifier` |
| `sklearn.ensemble.RandomForestRegressor` | `RandomForestRegressor` |
| `sklearn.ensemble.GradientBoostingClassifier` | `GradientBoostingClassifier` |
| `sklearn.ensemble.HistGradientBoostingClassifier` | `HistGradientBoostingClassifier` |
| `sklearn.svm.SVC` | `SVC` |
| `sklearn.svm.SVR` | `SVR` |
| `sklearn.neighbors.KNeighborsClassifier` | `KNNClassifier` |
| `sklearn.neighbors.KNeighborsRegressor` | `KNNRegressor` |
| `sklearn.cluster.KMeans` | `KMeans` |
| `sklearn.cluster.DBSCAN` | `DBSCAN` |
| `sklearn.cluster.AgglomerativeClustering` | `AgglomerativeClustering` |
| `sklearn.naive_bayes.GaussianNB` | `GaussianNB` |
| `sklearn.decomposition.PCA` | `PCA` |
| `sklearn.decomposition.TruncatedSVD` | `TruncatedSVD` |
| `sklearn.manifold.TSNE` | `TSNE` |
| `sklearn.preprocessing.StandardScaler` | `StandardScaler` |
| `sklearn.preprocessing.MinMaxScaler` | `MinMaxScaler` |
| `sklearn.preprocessing.LabelEncoder` | `LabelEncoder` |
| `sklearn.preprocessing.OneHotEncoder` | `OneHotEncoder` |
| `sklearn.pipeline.Pipeline` | `Pipeline` |
| `sklearn.pipeline.FeatureUnion` | `FeatureUnion` |
| `sklearn.compose.ColumnTransformer` | `ColumnTransformer` |
| `sklearn.model_selection.train_test_split` | `train_test_split()` |
| `sklearn.model_selection.KFold` | `KFold` |
| `sklearn.model_selection.cross_val_score` | `cross_val_score()` |
| `sklearn.model_selection.GridSearchCV` | `grid_search_cv()` |
| `sklearn.model_selection.RandomizedSearchCV` | `random_search_cv()` |
| `sklearn.metrics.accuracy_score` | `metrics::accuracy()` |
| `sklearn.metrics.f1_score` | `metrics::f1()` |
| `joblib.dump` / `joblib.load` | `Persistable::save()` / `Persistable::load()` |

## PyTorch to scivex-nn

| PyTorch | Scivex |
|---------|--------|
| `torch.Tensor` | `Variable<T>` (with autograd) / `Tensor<T>` (without) |
| `torch.nn.Module` | `Layer` trait |
| `torch.nn.Linear` | `Linear` |
| `torch.nn.Conv1d` | `Conv1d` |
| `torch.nn.Conv2d` | `Conv2d` |
| `torch.nn.Conv3d` | `Conv3d` |
| `torch.nn.BatchNorm1d` | `BatchNorm1d` |
| `torch.nn.BatchNorm2d` | `BatchNorm2d` |
| `torch.nn.LayerNorm` | `LayerNorm` |
| `torch.nn.RNN` | `SimpleRNN` |
| `torch.nn.LSTM` | `LSTM` |
| `torch.nn.GRU` | `GRU` |
| `torch.nn.MultiheadAttention` | `MultiHeadAttention` |
| `torch.nn.TransformerEncoderLayer` | `TransformerEncoderLayer` |
| `torch.nn.TransformerDecoderLayer` | `TransformerDecoderLayer` |
| `torch.nn.MaxPool1d` | `MaxPool1d` |
| `torch.nn.MaxPool2d` | `MaxPool2d` |
| `torch.nn.AvgPool1d` | `AvgPool1d` |
| `torch.nn.AvgPool2d` | `AvgPool2d` |
| `torch.nn.Dropout` | `Dropout` |
| `torch.nn.Embedding` | `Embedding` |
| `torch.nn.Flatten` | `Flatten` |
| `torch.nn.Sequential` | `Sequential` |
| `torch.nn.ReLU` | `ReLU` / `relu()` |
| `torch.nn.Sigmoid` | `Sigmoid` / `sigmoid()` |
| `torch.nn.Tanh` | `Tanh` / `tanh_fn()` |
| `torch.nn.functional.softmax` | `softmax()` |
| `torch.nn.functional.log_softmax` | `log_softmax()` |
| `torch.nn.MSELoss` | `mse_loss()` |
| `torch.nn.CrossEntropyLoss` | `cross_entropy_loss()` |
| `torch.nn.BCELoss` | `bce_loss()` |
| `torch.optim.SGD` | `SGD` |
| `torch.optim.Adam` | `Adam` |
| `torch.optim.AdamW` | `AdamW` |
| `torch.optim.RMSprop` | `RMSprop` |
| `torch.optim.Adagrad` | `Adagrad` |
| `torch.optim.lr_scheduler.StepLR` | `StepLR` |
| `torch.optim.lr_scheduler.CosineAnnealingLR` | `CosineAnnealingLR` |
| `torch.utils.data.Dataset` | `Dataset` trait |
| `torch.utils.data.DataLoader` | `DataLoader` |
| `torch.save` / `torch.load` | `save_weights()` / `load_weights()` |
| `safetensors` | `save_safetensors()` / `load_safetensors()` |
| ONNX Runtime | `load_onnx()`, `OnnxInferenceSession` |

## SciPy to scivex-optim / scivex-stats / scivex-signal

| SciPy | Scivex |
|-------|--------|
| `scipy.optimize.minimize` (BFGS) | `bfgs()` |
| `scipy.optimize.minimize` (L-BFGS-B) | `lbfgsb()` |
| `scipy.optimize.minimize` (Nelder-Mead) | `nelder_mead()` |
| `scipy.optimize.curve_fit` | `curve_fit()` |
| `scipy.optimize.least_squares` | `levenberg_marquardt()` |
| `scipy.optimize.linprog` | `linprog()` |
| `scipy.optimize.brentq` | `brent_root()` |
| `scipy.optimize.bisect` | `bisection()` |
| `scipy.optimize.newton` | `newton()` |
| `scipy.integrate.quad` | `quad()` |
| `scipy.integrate.trapezoid` | `trapezoid()` |
| `scipy.integrate.simpson` | `simpson()` |
| `scipy.integrate.solve_ivp` | `solve_ivp()` |
| `scipy.interpolate.interp1d` | `interp1d()` |
| `scipy.interpolate.CubicSpline` | `CubicSpline` |
| `scipy.interpolate.BSpline` | `BSpline` |
| `scipy.stats.norm` | `Normal` |
| `scipy.stats.t` | `StudentT` |
| `scipy.stats.chi2` | `ChiSquared` |
| `scipy.stats.ttest_1samp` | `t_test_one_sample()` |
| `scipy.stats.ttest_ind` | `t_test_two_sample()` |
| `scipy.stats.pearsonr` | `pearson()` |
| `scipy.stats.spearmanr` | `spearman()` |
| `scipy.signal.stft` | `spectral::stft()` |
| `scipy.signal.spectrogram` | `spectral::spectrogram()` |
| `scipy.signal.lfilter` | `filter::lfilter()` |
| `scipy.signal.filtfilt` | `filter::filtfilt()` |
| `scipy.signal.find_peaks` | `peak::find_peaks()` |
| `scipy.signal.resample` | `resample::resample()` |
| `scipy.signal.convolve` | `convolution::convolve()` |

## Matplotlib to scivex-viz

| Matplotlib | Scivex |
|------------|--------|
| `matplotlib.figure.Figure` | `Figure` |
| `matplotlib.axes.Axes` | `Axes` |
| `plt.plot()` | `LinePlot::new(x, y)` |
| `plt.scatter()` | `ScatterPlot::new(x, y)` |
| `plt.bar()` | `BarPlot::new(labels, values)` |
| `plt.hist()` | `Histogram::new(data, n_bins)` |
| `plt.imshow()` | `HeatmapBuilder` |
| `plt.contour()` | `ContourPlot` |
| `plt.pie()` | `PieChart` |
| `plt.boxplot()` | `BoxPlotBuilder` |
| `plt.violinplot()` | `ViolinPlot` |
| `mpl_toolkits.mplot3d` | `SurfacePlot` |
| `seaborn.pairplot()` | `PairPlot` |
| `seaborn.jointplot()` | `JointPlot` |
| `plt.subplots(2, 2)` | `Layout::grid(2, 2)` |
| `plt.savefig("out.svg")` | `fig.save_svg("out.svg")?` |
| `plt.show()` | `fig.to_terminal()?` |
| `FuncAnimation` | `Animation` with `to_gif()` |

## Error Handling Pattern

| Python | Scivex Rust |
|--------|-------------|
| `raise ValueError(...)` | `Err(CoreError::InvalidParameter(...))` |
| `try: ... except:` | `match result { Ok(v) => ..., Err(e) => ... }` |
| Implicit `None` return | `Result<T, Error>` — must handle |
| Runtime `TypeError` | Caught at compile time via generics |
| `assert x.shape == (2, 3)` | `assert_eq!(x.shape(), &[2, 3])` |
