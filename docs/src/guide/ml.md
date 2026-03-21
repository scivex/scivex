# Machine Learning

The `scivex-ml` crate provides classical machine learning algorithms with a
consistent, trait-based API. Every estimator follows the same patterns, making
it easy to swap models without changing surrounding code.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;
```

---

## The Trait Hierarchy: Transformer, Predictor, Classifier

All estimators in `scivex-ml` implement one of three traits.

### Transformer -- unsupervised fit/transform

```rust
pub trait Transformer<T: Float> {
    fn fit(&mut self, x: &Tensor<T>) -> Result<()>;
    fn transform(&self, x: &Tensor<T>) -> Result<Tensor<T>>;
    fn fit_transform(&mut self, x: &Tensor<T>) -> Result<Tensor<T>>;
}
```

Types that implement `Transformer`: `StandardScaler`, `MinMaxScaler`,
`OneHotEncoder`, `PCA`, `TruncatedSVD`, `TSNE`.

### Predictor -- supervised fit/predict

```rust
pub trait Predictor<T: Float> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()>;
    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>>;
}
```

Types that implement `Predictor`: `LinearRegression`, `Ridge`,
`LogisticRegression`, `DecisionTreeClassifier`, `DecisionTreeRegressor`,
`RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingRegressor`,
`GradientBoostingClassifier`, `SVC`, `SVR`, `KNNClassifier`, `KNNRegressor`,
`GaussianNB`, `Pipeline`.

### Classifier -- adds probability predictions

```rust
pub trait Classifier<T: Float>: Predictor<T> {
    fn predict_proba(&self, x: &Tensor<T>) -> Result<Tensor<T>>;
}
```

### Data format conventions

Throughout `scivex-ml`, feature matrices `x` are 2-D tensors of shape
`[n_samples, n_features]` and target vectors `y` are 1-D tensors of shape
`[n_samples]`.

```rust
let x = Tensor::from_vec(
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    vec![3, 2], // 3 samples, 2 features
).unwrap();
let y = Tensor::from_vec(vec![0.0, 1.0, 1.0], vec![3]).unwrap();
```

### Error handling

All fallible operations return `Result<T, MlError>`. Key error variants:

- `MlError::NotFitted` -- calling `predict`/`transform` before `fit`
- `MlError::DimensionMismatch` -- shape disagreements between x and y
- `MlError::InvalidParameter` -- illegal hyperparameter values
- `MlError::EmptyInput` -- zero-length input data
- `MlError::ConvergenceFailure` -- iterative solver did not converge
- `MlError::SingularMatrix` -- non-invertible matrix encountered

---

## Decision Trees

CART decision trees for classification (Gini impurity) and regression (MSE
criterion). Both support `max_depth` and `min_samples_split` parameters.

### Classification

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    vec![4, 2],
).unwrap();
let y = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], vec![4]).unwrap();

let mut tree = DecisionTreeClassifier::<f64>::new(Some(4), 1);
tree.fit(&x, &y).unwrap();
let preds = tree.predict(&x).unwrap();
assert_eq!(preds.as_slice(), &[0.0, 1.0, 1.0, 0.0]);
```

Parameters:
- `max_depth: Option<usize>` -- maximum tree depth (`None` for unlimited)
- `min_samples_split: usize` -- minimum samples required to split a node

### Regression

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3, 1]).unwrap();
let y = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();

let mut tree = DecisionTreeRegressor::<f64>::new(None, 1);
tree.fit(&x, &y).unwrap();
let preds = tree.predict(&x).unwrap();
// With unlimited depth, the tree memorises the training data.
```

### Feature importance (split-count based)

```rust
// After fitting a DecisionTreeRegressor:
let importances: Vec<f64> = tree.feature_importances(2);
// Returns normalised split counts per feature.
```

---

## Random Forests

Bagging of decision trees with bootstrap sampling and random feature subsets.

### Random Forest Classifier

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![1.0_f64, 1.0, 2.0, 2.0, 1.5, 0.5,
         8.0, 8.0, 9.0, 9.0, 8.5, 7.5],
    vec![6, 2],
).unwrap();
let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

let mut rf = RandomForestClassifier::<f64>::new(
    100,        // n_trees
    Some(5),    // max_depth
    None,       // max_features (defaults to sqrt(n_features))
    42,         // seed
).unwrap();

rf.fit(&x, &y).unwrap();
let preds = rf.predict(&x).unwrap();
```

### Random Forest Regressor

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let mut rf = RandomForestRegressor::<f64>::new(
    50,         // n_trees
    Some(10),   // max_depth
    None,       // max_features
    42,         // seed
).unwrap();

rf.fit(&x, &y).unwrap();
let preds = rf.predict(&x).unwrap();
```

### Parallel training

With the `parallel` feature flag enabled, use `par_fit` for multi-threaded
tree construction via Rayon:

```rust
rf.par_fit(&x, &y).unwrap();
```

---

## Gradient Boosting

Sequential fitting of decision tree regressors to pseudo-residuals. Supports
MSE, MAE, and Huber loss functions.

### Gradient Boosting Regressor

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    vec![8, 1],
).unwrap();
let y = Tensor::from_vec(
    vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
    vec![8],
).unwrap();

let mut gbr = GradientBoostingRegressor::<f64>::new(
    100,        // n_estimators
    0.1,        // learning_rate
    Some(3),    // max_depth per tree
    GBLoss::Mse,
).unwrap();

// Optional: enable stochastic gradient boosting
gbr.set_subsample(0.8).set_seed(123);

gbr.fit(&x, &y).unwrap();
let preds = gbr.predict(&x).unwrap();
```

**Loss functions** (`GBLoss`):
- `GBLoss::Mse` -- L2, gradient = -(y - F(x))
- `GBLoss::Mae` -- L1, gradient = -sign(y - F(x))
- `GBLoss::Huber` -- smooth L1/L2 blend; configure delta with `set_huber_delta(1.35)`

**Staged predictions** for early-stopping analysis:

```rust
let stages: Vec<Tensor<f64>> = gbr.staged_predict(&x).unwrap();
// stages[i] contains predictions after boosting round i.
```

**Feature importances** (aggregated impurity reduction):

```rust
let importances: Tensor<f64> = gbr.feature_importances(n_features).unwrap();
```

### Gradient Boosting Classifier

Multi-class classification using log-loss with one-vs-all decomposition. Each
class gets its own sequence of regression trees.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![1.0_f64, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0],
    vec![4, 2],
).unwrap();
let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();

let mut gbc = GradientBoostingClassifier::<f64>::new(
    50,         // n_estimators
    0.1,        // learning_rate
    Some(3),    // max_depth
).unwrap();

gbc.fit(&x, &y).unwrap();
let preds = gbc.predict(&x).unwrap();

// Class probabilities (shape [n_samples, n_classes]):
let proba = gbc.predict_proba(&x).unwrap();
```

### Histogram-based Gradient Boosting

`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` bin
continuous features into discrete histograms before tree building, giving
faster training on large datasets. Available via:

```rust
use scivex_ml::prelude::{HistGradientBoostingClassifier, HistGradientBoostingRegressor};
```

---

## Support Vector Machines

SMO-based SVM with multiple kernel functions. Binary SVM uses direct SMO;
multi-class uses one-vs-one decomposition with voting.

### Kernels

```rust
use scivex_ml::prelude::Kernel;

let linear = Kernel::<f64>::Linear;
let rbf    = Kernel::<f64>::Rbf { gamma: 0.1 };
let poly   = Kernel::<f64>::Poly { degree: 3, gamma: 0.01, coef0: 1.0 };
let sig    = Kernel::<f64>::Sigmoid { gamma: 1.0, coef0: 0.0 };

// Convenience constructors:
let rbf_auto = Kernel::<f64>::rbf_auto(4);    // gamma = 1/n_features
let poly_auto = Kernel::<f64>::poly(3, 4);    // degree=3, gamma=1/n_features
```

### SVC (Support Vector Classifier)

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![1.0_f64, 1.0, 2.0, 2.0, 1.5, 0.5,
         8.0, 8.0, 9.0, 9.0, 8.5, 7.5],
    vec![6, 2],
).unwrap();
let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

let mut svc = SVC::<f64>::new(Kernel::Rbf { gamma: 0.1 }, 10.0).unwrap();
svc.set_tol(1e-4);
svc.set_max_iter(2000);
svc.fit(&x, &y).unwrap();
let preds = svc.predict(&x).unwrap();

// Access support vectors:
let svs: Vec<&[f64]> = svc.support_vectors().unwrap();

// Decision function (distance from hyperplane, binary only):
let distances = svc.decision_function(&x).unwrap();
```

### SVR (Support Vector Regressor)

Epsilon-insensitive SVR. Only errors larger than `epsilon` contribute to the
loss.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    vec![8, 1],
).unwrap();
let y = Tensor::from_vec(
    vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
    vec![8],
).unwrap();

let mut svr = SVR::<f64>::new(
    Kernel::Linear,
    100.0,    // C (regularisation)
    0.5,      // epsilon (tube width)
).unwrap();
svr.set_max_iter(2000);
svr.fit(&x, &y).unwrap();
let preds = svr.predict(&x).unwrap();
```

---

## K-Nearest Neighbours

Brute-force KNN with Euclidean distance. Fitting stores the training data;
prediction computes distances on demand.

### KNN Classifier

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![0.0_f64, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1],
    vec![4, 2],
).unwrap();
let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();

let mut knn = KNNClassifier::<f64>::new(3).unwrap();
knn.fit(&x, &y).unwrap();

let test_x = Tensor::from_vec(vec![0.05, 0.05, 10.05, 10.05], vec![2, 2]).unwrap();
let preds = knn.predict(&test_x).unwrap();
assert_eq!(preds.as_slice(), &[0.0, 1.0]);
```

### KNN Regressor

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(vec![1.0_f64, 3.0, 5.0], vec![3, 1]).unwrap();
let y = Tensor::from_vec(vec![2.0, 6.0, 10.0], vec![3]).unwrap();

let mut knn = KNNRegressor::<f64>::new(2).unwrap();
knn.fit(&x, &y).unwrap();

let test_x = Tensor::from_vec(vec![2.0], vec![1, 1]).unwrap();
let preds = knn.predict(&test_x).unwrap();
// Averages the 2 nearest neighbours: (2.0 + 6.0) / 2 = 4.0
assert!((preds.as_slice()[0] - 4.0).abs() < 1e-10);
```

### Advanced nearest-neighbour structures

The `neighbors` module also provides:

- `BruteForceIndex` -- standalone brute-force k-NN index returning
  `NearestNeighborResult`
- `HnswIndex` -- Hierarchical Navigable Small World graph for approximate
  nearest-neighbour search
- `ProductQuantizer` -- product quantisation for compressed distance
  computation
- `DistanceMetric` -- configurable distance functions

---

## Clustering

### KMeans

Lloyd's algorithm with multiple random initialisations (`n_init`).

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![0.0_f64, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1],
    vec![4, 2],
).unwrap();

let mut km = KMeans::<f64>::new(
    2,      // n_clusters
    100,    // max_iter
    1e-6,   // tol (convergence threshold)
    3,      // n_init (number of random restarts)
    42,     // seed
).unwrap();

km.fit(&x).unwrap();
let labels = km.predict(&x).unwrap();

// Inspect results:
let inertia: f64 = km.inertia().unwrap();
let centroids: &[f64] = km.centroids().unwrap();
```

### DBSCAN

Density-based clustering. Points in sparse regions are labelled as noise (-1).

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![
        0.0_f64, 0.0,  0.1, 0.1,  0.0, 0.1,
        10.0, 10.0,  10.1, 10.1,  10.0, 10.1,
    ],
    vec![6, 2],
).unwrap();

let mut db = DBSCAN::<f64>::new(
    1.0,    // eps (neighbourhood radius)
    2,      // min_samples (core point threshold)
).unwrap();

db.fit(&x).unwrap();

let n_clusters: usize = db.n_clusters().unwrap();
let labels: &[i64] = db.labels().unwrap();
// labels[i] == -1 means noise

// Or use fit_predict to get labels as a Tensor:
let mut db2 = DBSCAN::<f64>::new(1.0, 2).unwrap();
let label_tensor = db2.fit_predict(&x).unwrap();

// Find which points are core samples:
let cores: Vec<usize> = db.core_sample_indices(&x).unwrap();
```

### Agglomerative Clustering

Hierarchical clustering with configurable linkage.

```rust
use scivex_ml::prelude::{AgglomerativeClustering, Linkage};
```

Supported linkage types: `Linkage::Single`, `Linkage::Complete`,
`Linkage::Average`, etc.

---

## Pipelines

Chain preprocessing transformers with a final predictor into a single
composite estimator.

### Basic Pipeline

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
let y = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0], vec![5]).unwrap();

let mut pipe = Pipeline::new()
    .add_step("scaler", Box::new(StandardScaler::<f64>::new()))
    .set_predictor("lr", Box::new(LinearRegression::<f64>::new()));

pipe.fit(&x, &y).unwrap();
let preds = pipe.predict(&x).unwrap();
```

`Pipeline` itself implements `Predictor`, so it can be used anywhere a
predictor is expected (including cross-validation and grid search).

### FeatureUnion

Runs multiple transformers in parallel and concatenates their outputs
column-wise.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();

let mut fu = FeatureUnion::new()
    .add("std", Box::new(StandardScaler::<f64>::new()))
    .add("minmax", Box::new(MinMaxScaler::<f64>::new()));

let combined = fu.fit_transform(&x).unwrap();
assert_eq!(combined.shape(), &[3, 4]); // 2 + 2 columns
```

### ColumnTransformer

Applies different transformers to different column subsets, then concatenates.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![1.0_f64, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0],
    vec![3, 3],
).unwrap();

let mut ct = ColumnTransformer::new()
    .add("numeric", Box::new(StandardScaler::<f64>::new()), vec![0])
    .add("other", Box::new(MinMaxScaler::<f64>::new()), vec![1, 2]);

let out = ct.fit_transform(&x).unwrap();
assert_eq!(out.shape(), &[3, 3]); // 1 + 2 columns
```

---

## Metrics

### Classification Metrics

All classification metrics take `&[T]` slices (not tensors). Use
`.as_slice()` on predictions.

```rust
use scivex_ml::metrics::classification::{accuracy, precision, recall, f1_score, confusion_matrix};

let y_true = [1.0_f64, 0.0, 1.0, 1.0, 0.0];
let y_pred = [1.0, 0.0, 0.0, 1.0, 0.0];

let acc = accuracy(&y_true, &y_pred).unwrap();    // 0.8
let prec = precision(&y_true, &y_pred).unwrap();  // TP / (TP + FP)
let rec = recall(&y_true, &y_pred).unwrap();      // TP / (TP + FN)
let f1 = f1_score(&y_true, &y_pred).unwrap();     // 2 * P * R / (P + R)

// Confusion matrix: row = true class, col = predicted class
let cm = confusion_matrix(&y_true, &y_pred, 2).unwrap();
// cm[0][0] = TN, cm[0][1] = FP, cm[1][0] = FN, cm[1][1] = TP
```

Note: for binary classification, the positive class is automatically detected
as the maximum label value in `y_true`.

### Regression Metrics

```rust
use scivex_ml::metrics::regression::{mse, rmse, mae, r2_score};

let y_true = [3.0_f64, -0.5, 2.0, 7.0];
let y_pred = [2.5, 0.0, 2.0, 8.0];

let mean_sq_err = mse(&y_true, &y_pred).unwrap();
let root_mse    = rmse(&y_true, &y_pred).unwrap();
let mean_abs    = mae(&y_true, &y_pred).unwrap();
let r_squared   = r2_score(&y_true, &y_pred).unwrap();
// r_squared ~ 0.9486 for this example
```

---

## Preprocessing

### StandardScaler

Zero-mean, unit-variance scaling per feature.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![1.0_f64, 10.0, 2.0, 20.0, 3.0, 30.0],
    vec![3, 2],
).unwrap();

let mut scaler = StandardScaler::<f64>::new();
let scaled = scaler.fit_transform(&x).unwrap();
// Each column now has mean ~ 0 and std ~ 1.

// Reverse the transformation:
let original = scaler.inverse_transform(&scaled).unwrap();
```

### MinMaxScaler

Scales features to the `[0, 1]` range.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![1.0_f64, 10.0, 3.0, 30.0, 5.0, 50.0],
    vec![3, 2],
).unwrap();

let mut scaler = MinMaxScaler::<f64>::new();
let scaled = scaler.fit_transform(&x).unwrap();
// col 0: min=1 -> 0.0, max=5 -> 1.0
// col 1: min=10 -> 0.0, max=50 -> 1.0
```

### LabelEncoder

Maps arbitrary label values to contiguous integers `0..n_classes`.

```rust
use scivex_ml::prelude::*;

let labels = [3.0_f64, 1.0, 2.0, 1.0, 3.0];

let mut enc = LabelEncoder::new();
enc.fit(&labels).unwrap();

let encoded: Vec<usize> = enc.transform(&labels).unwrap();
// Sorted classes [1.0, 2.0, 3.0] -> indices [2, 0, 1, 0, 2]

let decoded: Vec<f64> = enc.inverse_transform(&encoded).unwrap();
assert_eq!(decoded, vec![3.0, 1.0, 2.0, 1.0, 3.0]);

assert_eq!(enc.n_classes(), Some(3));
```

### OneHotEncoder

Expands integer-valued categorical features into binary indicator columns.
Implements the `Transformer` trait.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

// 2 features: col0 has categories {0,1,2}, col1 has categories {0,1}
let x = Tensor::from_vec(
    vec![0.0_f64, 0.0, 1.0, 1.0, 2.0, 0.0],
    vec![3, 2],
).unwrap();

let mut enc = OneHotEncoder::<f64>::new();
let encoded = enc.fit_transform(&x).unwrap();
// Output shape: [3, 5] (3 + 2 indicator columns)

assert_eq!(enc.n_output_features(), Some(5));
let cats = enc.categories().unwrap();
// cats[0] == [0.0, 1.0, 2.0], cats[1] == [0.0, 1.0]
```

---

## Cross-Validation and Model Selection

### Train/Test Split

```rust
use scivex_core::{Tensor, random::Rng};
use scivex_ml::prelude::*;

let x = Tensor::from_vec(
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    vec![5, 2],
).unwrap();
let y = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0], vec![5]).unwrap();

let mut rng = Rng::new(42);
let (x_train, x_test, y_train, y_test) = train_test_split(
    &x, &y,
    0.4_f64,   // test_ratio (must be in (0, 1))
    &mut rng,
).unwrap();

assert_eq!(x_train.shape()[0], 3); // 60% train
assert_eq!(x_test.shape()[0], 2);  // 40% test
```

### K-Fold Cross-Validation

```rust
use scivex_core::{Tensor, random::Rng};
use scivex_ml::prelude::*;
use scivex_ml::metrics::regression::r2_score;

let x = Tensor::from_vec(
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    vec![10, 1],
).unwrap();
let y = Tensor::from_vec(
    vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0],
    vec![10],
).unwrap();

let model = LinearRegression::<f64>::new();
let mut rng = Rng::new(42);

let scores: Vec<f64> = cross_val_score(
    &model, &x, &y,
    3,           // n_folds
    r2_score,    // metric function: fn(&[T], &[T]) -> Result<T>
    &mut rng,
).unwrap();

// scores contains one R2 value per fold
for &s in &scores {
    assert!(s > 0.8);
}
```

### Using KFold directly

For more control, use `KFold` to iterate over train/test index splits:

```rust
use scivex_core::random::Rng;
use scivex_ml::prelude::*;

let mut rng = Rng::new(42);
let kfold = KFold::new(5, 100, &mut rng).unwrap();

for (train_indices, test_indices) in &kfold {
    assert_eq!(train_indices.len() + test_indices.len(), 100);
    // Build tensors from indices and train/evaluate...
}
```

### Hyperparameter Search

Grid search and random search with cross-validation:

```rust
use scivex_ml::prelude::{grid_search_cv, random_search_cv, SearchResult};
```

---

## Explainability: SHAP, Permutation Importance, LIME, and PDP

### Tree SHAP

Exact Shapley values for `DecisionTreeRegressor`. Returns a tensor of shape
`[n_samples, n_features]` representing each feature's contribution to the
prediction.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
let y = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();

let mut tree = DecisionTreeRegressor::<f64>::new(None, 1);
tree.fit(&x, &y).unwrap();

let shap_values = tree_shap(&tree, &x).unwrap();
assert_eq!(shap_values.shape(), &[4, 1]);
```

### Kernel SHAP

Model-agnostic SHAP approximation. Works with any `Predictor`. Uses a weighted
sampling approach with coalition vectors.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

// model: any fitted Predictor
// x_background: background dataset for marginalisation
// x_explain: instances to explain

let shap_values = kernel_shap(
    &model,
    &x_background,
    &x_explain,
    100,    // n_samples (number of coalition samples)
    42,     // seed
).unwrap();
// Shape: [n_explain, n_features]
```

### Permutation Importance

Measures feature importance by shuffling each feature column and observing
the decrease in a scoring metric. Works with any `Predictor`.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

// scorer: fn(&Tensor<T>, &Tensor<T>) -> T
let result: PermutationImportanceResult<f64> = permutation_importance(
    &model,
    &x_test,
    &y_test,
    5,          // n_repeats
    scorer_fn,  // scoring function
    42,         // seed
).unwrap();

// result.importances_mean  -- shape [n_features]
// result.importances_std   -- shape [n_features]
// result.importances_raw   -- shape [n_features, n_repeats]
```

### Partial Dependence

Computes the average model prediction as a single feature varies across a
grid of values, while other features are held at their observed values.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let pd: PartialDependence<f64> = partial_dependence(
    &model,
    &x,
    0,      // feature_index
    50,     // grid_resolution
).unwrap();

// pd.feature_values       -- grid points [grid_resolution]
// pd.average_predictions   -- avg prediction at each point [grid_resolution]
```

### LIME

Local Interpretable Model-agnostic Explanations. Perturbs an instance, weights
perturbations by proximity, and fits a local linear model.

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let explanations: Vec<LimeExplanation<f64>> = lime(
    &model,
    &x_train,        // training data (for feature statistics)
    &x_explain,      // instances to explain
    500,             // n_perturbations
    None,            // kernel_width (None for automatic)
    42,              // seed
).unwrap();

for exp in &explanations {
    // exp.weights     -- per-feature importance weights
    // exp.intercept   -- intercept of local linear model
    // exp.score       -- local R2 (goodness of fit)
    // exp.prediction  -- model prediction for the explained instance
}
```

---

## Model Persistence

Save and load trained models to/from binary format using the `Persistable`
trait. Requires the `serde-support` feature flag.

```rust
use scivex_ml::prelude::Persistable;
```

---

## Online / Streaming Algorithms

The `online` module provides incremental estimators for streaming data:

- `SGDClassifier` -- stochastic gradient descent classifier
- `SGDRegressor` -- stochastic gradient descent regressor
- `OnlineKMeans` -- mini-batch K-Means
- `StreamingMean` -- running mean computation
- `StreamingVariance` -- running variance computation

All online predictors implement the `IncrementalPredictor` trait for
partial fitting:

```rust
use scivex_ml::prelude::*;
```

---

## Putting It All Together

A complete workflow: preprocess, train, evaluate, and explain.

```rust
use scivex_core::{Tensor, random::Rng};
use scivex_ml::prelude::*;
use scivex_ml::metrics::classification::accuracy;
use scivex_ml::metrics::regression::r2_score;

// 1. Create data
let x = Tensor::from_vec(
    vec![
        1.0_f64, 2.0,  1.5, 1.8,  2.0, 1.0,
        8.0, 9.0,  9.0, 8.5,  8.5, 9.5,
    ],
    vec![6, 2],
).unwrap();
let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

// 2. Split
let mut rng = Rng::new(42);
let (x_train, x_test, y_train, y_test) = train_test_split(
    &x, &y, 0.33, &mut rng,
).unwrap();

// 3. Build a pipeline: scale + classify
let mut pipe = Pipeline::new()
    .add_step("scaler", Box::new(StandardScaler::<f64>::new()))
    .set_predictor("rf", Box::new(
        RandomForestClassifier::<f64>::new(50, Some(5), None, 42).unwrap()
    ));

pipe.fit(&x_train, &y_train).unwrap();
let preds = pipe.predict(&x_test).unwrap();

// 4. Evaluate
let acc = accuracy(y_test.as_slice(), preds.as_slice()).unwrap();
println!("Test accuracy: {acc:.2}");

// 5. Cross-validate
let model = DecisionTreeClassifier::<f64>::new(Some(3), 2);
let scores = cross_val_score(
    &model, &x, &y, 3, accuracy, &mut rng,
).unwrap();
let mean_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
println!("CV accuracy: {mean_score:.2}");
```
