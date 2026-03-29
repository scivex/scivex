# Migrating from scikit-learn to Scivex

This guide maps scikit-learn's Python API to the equivalent Scivex Rust API.
Every function, type, and method referenced here exists in the `scivex-ml` crate.

---

## Core Concepts

### Trait-based estimator hierarchy

scikit-learn uses duck typing and informal interfaces (`fit`, `predict`, `transform`).
Scivex encodes the same contracts as explicit Rust traits defined in `scivex_ml::traits`:

| scikit-learn concept | Scivex trait | Methods |
|---|---|---|
| Transformer (`fit` / `transform`) | `Transformer<T: Float>` | `fit(&mut self, x: &Tensor<T>)`, `transform(&self, x: &Tensor<T>)`, `fit_transform(&mut self, x: &Tensor<T>)` |
| Estimator (`fit` / `predict`) | `Predictor<T: Float>` | `fit(&mut self, x: &Tensor<T>, y: &Tensor<T>)`, `predict(&self, x: &Tensor<T>)` |
| Classifier with probabilities | `Classifier<T: Float>` (extends `Predictor`) | `predict_proba(&self, x: &Tensor<T>)` |

All trait methods return `Result<T, MlError>` rather than panicking.

### Generic numeric types

scikit-learn always works with NumPy `float64` arrays.
Scivex models are generic over any type that implements the `Float` trait (typically `f32` or `f64`).
You choose the precision at the type level:

```rust
let mut model = LinearRegression::<f64>::new();
// or
let mut model = LinearRegression::<f32>::new();
```

### Error handling

scikit-learn raises Python exceptions.
Scivex returns `Result<T, MlError>` from every fallible operation.
The `MlError` enum covers:

| Variant | Meaning |
|---|---|
| `NotFitted` | Called `predict` before `fit` |
| `EmptyInput` | Passed an empty tensor |
| `DimensionMismatch { expected, got }` | Shape mismatch |
| `InvalidParameter { name, reason }` | Bad hyperparameter value |
| `ConvergenceFailure { iterations }` | Iterative algorithm did not converge |
| `SingularMatrix` | Matrix inversion failed |
| `CoreError(_)` | Propagated from `scivex-core` |

---

## Quick-reference Table

| scikit-learn | Scivex (`scivex_ml::prelude`) |
|---|---|
| `LinearRegression` | `LinearRegression::<f64>::new()` |
| `Ridge(alpha=1.0)` | `Ridge::<f64>::new(1.0)?` |
| `LogisticRegression` | `LogisticRegression::<f64>::new(lr, max_iter, tol)?` |
| `DecisionTreeClassifier` | `DecisionTreeClassifier::<f64>::new(Some(max_depth), min_samples_split)` |
| `DecisionTreeRegressor` | `DecisionTreeRegressor::<f64>::new(Some(max_depth), min_samples_split)` |
| `RandomForestClassifier` | `RandomForestClassifier::<f64>::new(n_trees, Some(max_depth), Some(max_features), seed)?` |
| `RandomForestRegressor` | `RandomForestRegressor::<f64>::new(n_trees, Some(max_depth), Some(max_features), seed)?` |
| `GradientBoostingRegressor` | `GradientBoostingRegressor::<f64>::new(n_estimators, learning_rate, Some(max_depth), Loss::Mse)?` |
| `GradientBoostingClassifier` | `GradientBoostingClassifier::<f64>::new(n_estimators, learning_rate, Some(max_depth))?` |
| `SVC(kernel='rbf')` | `SVC::<f64>::new(Kernel::Rbf { gamma: T::from_f64(0.1) }, 1.0)?` |
| `KNeighborsClassifier(n_neighbors=5)` | `KNNClassifier::<f64>::new(5)?` |
| `KNeighborsRegressor(n_neighbors=5)` | `KNNRegressor::<f64>::new(5)?` |
| `GaussianNB` | `GaussianNB::<f64>::new()` |
| `KMeans(n_clusters=3)` | `KMeans::<f64>::new(3, 100, 1e-6, 3, 42)?` |
| `DBSCAN(eps=0.5, min_samples=5)` | `DBSCAN::<f64>::new(0.5, 5)?` |
| `AgglomerativeClustering` | `AgglomerativeClustering::<f64>::new(n_clusters, Linkage::Ward)` |
| `PCA(n_components=2)` | `PCA::<f64>::new(2)?` |
| `TruncatedSVD` | `TruncatedSVD::<f64>::new(n_components)?` |
| `TSNE` | `TSNE::<f64>::new(n_components)?` |
| `StandardScaler` | `StandardScaler::<f64>::new()` |
| `MinMaxScaler` | `MinMaxScaler::<f64>::new()` |
| `LabelEncoder` | `LabelEncoder::<f64>::new()` |
| `OneHotEncoder` | `OneHotEncoder::<f64>::new()` |
| `Pipeline` | `Pipeline::<f64>::new()` |
| `FeatureUnion` | `FeatureUnion::<f64>::new()` |
| `ColumnTransformer` | `ColumnTransformer::<f64>::new()` |

---

## Model Creation and Training

### Linear regression

**scikit-learn:**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([3, 5, 7, 9])

model = LinearRegression()
model.fit(X, y)
preds = model.predict(X)
print(model.coef_, model.intercept_)
```

**Scivex:**

```rust
use scivex_core::Tensor;
use scivex_ml::prelude::*;

let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4, 1])?;
let y = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0], vec![4])?;

let mut model = LinearRegression::<f64>::new();
model.fit(&x, &y)?;
let preds = model.predict(&x)?;

// Access fitted parameters
let weights = model.weights();   // Option<&[f64]>
let bias = model.bias();         // Option<f64>
```

### Ridge regression

**scikit-learn:**

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=0.1)
model.fit(X, y)
```

**Scivex:**

```rust
let mut model = Ridge::<f64>::new(0.1)?;
model.fit(&x, &y)?;
```

### Logistic regression

**scikit-learn:**

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
probs = model.predict_proba(X)
```

**Scivex:**

```rust
let mut model = LogisticRegression::<f64>::new(
    0.01,   // learning_rate
    1000,   // max_iter
    1e-6,   // tol
)?;
model.fit(&x, &y)?;
let preds = model.predict(&x)?;
let probs = model.predict_proba(&x)?;  // via Classifier trait
```

---

## Decision Trees and Ensembles

### Decision tree

**scikit-learn:**

```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
tree.fit(X, y)
```

**Scivex:**

```rust
let mut tree = DecisionTreeClassifier::<f64>::new(
    Some(5),  // max_depth (None for unlimited)
    2,        // min_samples_split
);
tree.fit(&x, &y)?;
let preds = tree.predict(&x)?;
```

### Random forest

**scikit-learn:**

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X, y)
```

**Scivex:**

```rust
let mut rf = RandomForestClassifier::<f64>::new(
    100,       // n_trees
    Some(10),  // max_depth
    None,      // max_features (None = sqrt(n_features))
    42,        // seed
)?;
rf.fit(&x, &y)?;

// With the `parallel` feature, use par_fit for multi-threaded training:
// rf.par_fit(&x, &y)?;
```

### Gradient boosting

**scikit-learn:**

```python
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)
gb.fit(X, y)
```

**Scivex:**

```rust
use scivex_ml::ensemble::GBLoss;

let mut gb = GradientBoostingRegressor::<f64>::new(
    200,       // n_estimators
    0.1,       // learning_rate
    Some(3),   // max_depth
    GBLoss::Mse,  // loss function: Mse, Mae, or Huber
)?;
gb.set_subsample(0.8);  // stochastic gradient boosting
gb.set_seed(42);
gb.fit(&x, &y)?;

// Feature importances (impurity-based)
let importances = gb.feature_importances(n_features)?;
```

---

## Support Vector Machines

**scikit-learn:**

```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, gamma=0.1)
model.fit(X, y)
```

**Scivex:**

```rust
use scivex_ml::svm::{Kernel, SVC};

let mut model = SVC::<f64>::new(
    Kernel::Rbf { gamma: 0.1_f64 },
    1.0,  // C (regularisation)
)?;
model.set_tol(1e-4);
model.set_max_iter(2000);
model.fit(&x, &y)?;
let preds = model.predict(&x)?;
```

Available kernels:

| scikit-learn | Scivex |
|---|---|
| `kernel='linear'` | `Kernel::Linear` |
| `kernel='rbf'` | `Kernel::Rbf { gamma }` |
| `kernel='poly'` | `Kernel::Poly { degree, gamma, coef0 }` |
| `kernel='sigmoid'` | `Kernel::Sigmoid { gamma, coef0 }` |

---

## K-Nearest Neighbours

**scikit-learn:**

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
```

**Scivex:**

```rust
let mut knn = KNNClassifier::<f64>::new(5)?;
knn.fit(&x, &y)?;
let preds = knn.predict(&x)?;
```

---

## Clustering

### KMeans

**scikit-learn:**

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=42)
km.fit(X)
labels = km.predict(X)
print(km.inertia_, km.cluster_centers_)
```

**Scivex:**

```rust
let mut km = KMeans::<f64>::new(
    3,      // n_clusters
    300,    // max_iter
    1e-6,   // tol
    10,     // n_init
    42,     // seed
)?;
km.fit(&x)?;
let labels = km.predict(&x)?;

let inertia = km.inertia();        // Option<f64>
let centroids = km.centroids();    // Option<&[f64]>
```

Note: `KMeans` uses its own `fit` and `predict` methods directly, rather than the
`Transformer` or `Predictor` traits, because clustering is neither purely supervised
nor a feature transformation.

### DBSCAN

**scikit-learn:**

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)
db.fit(X)
labels = db.labels_
```

**Scivex:**

```rust
let mut db = DBSCAN::<f64>::new(0.5, 5)?;
db.fit(&x)?;
let n = db.n_clusters();  // Option<usize>
```

---

## Preprocessing

### StandardScaler

**scikit-learn:**

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_original = scaler.inverse_transform(X_scaled)
```

**Scivex:**

```rust
let mut scaler = StandardScaler::<f64>::new();
let x_scaled = scaler.fit_transform(&x)?;  // from Transformer trait
let x_original = scaler.inverse_transform(&x_scaled)?;
```

`MinMaxScaler`, `LabelEncoder`, and `OneHotEncoder` follow the same `Transformer`
trait pattern with `fit`, `transform`, and `fit_transform`.

---

## Pipelines

### Basic pipeline

**scikit-learn:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression()),
])
pipe.fit(X, y)
preds = pipe.predict(X)
```

**Scivex:**

```rust
let mut pipe = Pipeline::<f64>::new()
    .add_step("scaler", Box::new(StandardScaler::<f64>::new()))
    .set_predictor("lr", Box::new(LinearRegression::<f64>::new()));

pipe.fit(&x, &y)?;    // fits all transformers, then the predictor
let preds = pipe.predict(&x)?;
```

Key differences:
- Transformer steps are added with `add_step`, the final predictor with `set_predictor`.
- Steps are boxed trait objects (`Box<dyn Transformer<T>>` and `Box<dyn Predictor<T>>`).
- `Pipeline` itself implements `Predictor<T>`, so it composes with other pipelines.
- Use `n_steps()` to query the number of transformer steps.

### FeatureUnion

**scikit-learn:**

```python
from sklearn.pipeline import FeatureUnion
fu = FeatureUnion([
    ('scaler1', StandardScaler()),
    ('scaler2', MinMaxScaler()),
])
X_out = fu.fit_transform(X)
```

**Scivex:**

```rust
let mut fu = FeatureUnion::<f64>::new()
    .add("scaler1", Box::new(StandardScaler::<f64>::new()))
    .add("scaler2", Box::new(MinMaxScaler::<f64>::new()));

let x_out = fu.fit_transform(&x)?;
// Output has columns from both transformers concatenated horizontally
```

### ColumnTransformer

**scikit-learn:**

```python
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([
    ('num', StandardScaler(), [0, 1]),
    ('cat', OneHotEncoder(), [2]),
])
X_out = ct.fit_transform(X)
```

**Scivex:**

```rust
let mut ct = ColumnTransformer::<f64>::new()
    .add("num", Box::new(StandardScaler::<f64>::new()), vec![0, 1])
    .add("cat", Box::new(OneHotEncoder::<f64>::new()), vec![2]);

let x_out = ct.fit_transform(&x)?;
```

---

## Dimensionality Reduction

### PCA

**scikit-learn:**

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
X_approx = pca.inverse_transform(X_reduced)
print(pca.explained_variance_ratio_)
```

**Scivex:**

```rust
let mut pca = PCA::<f64>::new(2)?;
let x_reduced = pca.fit_transform(&x)?;       // Transformer trait
let x_approx = pca.inverse_transform(&x_reduced)?;

let ev = pca.explained_variance();             // Option<&[f64]>
let evr = pca.explained_variance_ratio();      // Option<Vec<f64>>
let components = pca.components();             // Option<&[f64]>
```

---

## Metrics

### Classification metrics

**scikit-learn:**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred)
recall_score(y_true, y_pred)
f1_score(y_true, y_pred)
confusion_matrix(y_true, y_pred)
```

**Scivex:**

```rust
use scivex_ml::metrics::classification::{accuracy, precision, recall, f1_score, confusion_matrix};

let acc = accuracy(y_true, y_pred)?;
let prec = precision(y_true, y_pred)?;
let rec = recall(y_true, y_pred)?;
let f1 = f1_score(y_true, y_pred)?;
let cm = confusion_matrix(y_true, y_pred)?;
```

All metric functions take `&[T]` slices and return `Result<T>`.

### Regression metrics

**scikit-learn:**

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mean_squared_error(y_true, y_pred)
mean_absolute_error(y_true, y_pred)
r2_score(y_true, y_pred)
```

**Scivex:**

```rust
use scivex_ml::metrics::regression::{mse, rmse, mae, r2_score};

let m = mse(y_true, y_pred)?;
let rm = rmse(y_true, y_pred)?;
let ma = mae(y_true, y_pred)?;
let r2 = r2_score(y_true, y_pred)?;
```

---

## Model Selection and Hyperparameter Tuning

### Train/test split

**scikit-learn:**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Scivex:**

```rust
use scivex_core::random::Rng;
use scivex_ml::model_selection::train_test_split;

let mut rng = Rng::new(42);
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, &mut rng)?;
```

The return type is `(Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>)`.

### K-Fold cross-validation

**scikit-learn:**

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
```

**Scivex:**

```rust
use scivex_ml::model_selection::cross_val_score;
use scivex_ml::metrics::regression::r2_score;

let model = LinearRegression::<f64>::new();
let mut rng = Rng::new(42);
let scores: Vec<f64> = cross_val_score(&model, &x, &y, 5, r2_score, &mut rng)?;
```

The model must implement `Predictor<T> + Clone`. The scorer is any
`Fn(&[T], &[T]) -> Result<T>`, which all metric functions satisfy.

You can also iterate over folds manually with `KFold`:

```rust
use scivex_ml::model_selection::KFold;

let kfold = KFold::new(5, n_samples, &mut rng)?;
for (train_indices, test_indices) in &kfold {
    // train_indices: Vec<usize>, test_indices: Vec<usize>
}
```

### Grid search

**scikit-learn:**

```python
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
gs = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
gs.fit(X, y)
print(gs.best_params_, gs.best_score_)
```

**Scivex:**

```rust
use scivex_ml::search::{grid_search_cv, SearchResult};

let candidates = vec![
    Ridge::<f64>::new(0.01)?,
    Ridge::<f64>::new(0.1)?,
    Ridge::<f64>::new(1.0)?,
    Ridge::<f64>::new(10.0)?,
];
let result: SearchResult<f64> = grid_search_cv(
    &candidates, &x, &y,
    5,          // n_folds
    r2_score,   // scorer
    &mut rng,
)?;

println!("Best index: {}", result.best_index);
println!("Best score: {}", result.best_score);
// result.mean_scores: Vec<f64>  -- mean CV score per candidate
// result.all_scores: Vec<Vec<f64>>  -- per-fold scores per candidate
```

Instead of a parameter grid dict, you pre-build the list of candidate models.

### Random search

**scikit-learn:**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
rs = RandomizedSearchCV(Ridge(), {'alpha': uniform(0.001, 10)}, n_iter=20, cv=5)
rs.fit(X, y)
```

**Scivex:**

```rust
use scivex_ml::search::random_search_cv;

let result = random_search_cv(
    |rng| {
        let alpha = rng.next_f64() * 10.0 + 0.001;
        Ridge::<f64>::new(alpha)
    },
    20,         // n_iter
    &x, &y,
    5,          // n_folds
    r2_score,   // scorer
    &mut rng,
)?;
```

The builder closure receives an `&mut Rng` and returns a `Result<M>` for each
random candidate.

---

## Model Persistence

**scikit-learn:**

```python
import joblib
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')
```

**Scivex:**

```rust
use scivex_ml::persist::Persistable;

model.save("model.svex")?;
let loaded = LinearRegression::<f64>::load("model.svex")?;
```

The binary format uses a `"SVEX"` magic header with versioning.
Models can also be serialised with serde when the `serde-support` feature is enabled.

---

## Key Differences Summary

| Aspect | scikit-learn | Scivex |
|---|---|---|
| Language | Python | Rust |
| Type safety | Runtime duck typing | Compile-time traits (`Predictor`, `Transformer`, `Classifier`) |
| Numeric type | Always `float64` | Generic: `f32` or `f64` via `Float` trait |
| Error handling | Exceptions | `Result<T, MlError>` |
| Data container | NumPy `ndarray` | `scivex_core::Tensor<T>` |
| Parallelism | joblib / built-in | `parallel` feature flag, `par_fit` methods |
| Serialisation | pickle / joblib | `Persistable` trait (binary), optional serde |
| Pipeline steps | Tuples in a list | Builder pattern: `add_step` / `set_predictor` |
| Hyperparameter search | Dict-based param grid | Pre-built candidate list or builder closure |
| Random state | `random_state=int` | `Rng::new(seed)` passed explicitly |
| Memory model | GC + NumPy C arrays | Ownership, zero-copy slices, stack-first allocation |

---

## Import Cheat Sheet

To get all common ML types in scope:

```rust
use scivex_ml::prelude::*;
use scivex_core::{Tensor, random::Rng};
```

Or, through the umbrella crate:

```rust
use scivex::prelude::*;
```

---

## Advanced Algorithms (Phases 26-30)

### CatBoost-style Gradient Boosting

```python
# Python (CatBoost)
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6)
model.fit(X_train, y_train, cat_features=[0, 2])
preds = model.predict(X_test)
```

```rust
// Scivex
use scivex_ml::ensemble::CatBoostClassifier;

let mut model = CatBoostClassifier::new()
    .iterations(100)
    .learning_rate(0.1)
    .depth(6)
    .cat_features(vec![0, 2]);
model.fit(&x_train, &y_train)?;
let preds = model.predict(&x_test)?;
```

### Stacking Ensemble

```python
# scikit-learn
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
stack = StackingClassifier(
    estimators=[('rf', RandomForestClassifier()), ('svm', SVC())],
    final_estimator=LogisticRegression()
)
stack.fit(X_train, y_train)
```

```rust
// Scivex
use scivex_ml::ensemble::StackingClassifier;

let mut stack = StackingClassifier::new(vec![
    Box::new(RandomForest::new(100)),
    Box::new(Svm::new()),
], Box::new(LogisticRegression::new()));
stack.fit(&x_train, &y_train)?;
let preds = stack.predict(&x_test)?;
```

### Feature Selection (RFE & SelectKBest)

```python
# scikit-learn
from sklearn.feature_selection import SelectKBest, chi2, RFE
selector = SelectKBest(chi2, k=10)
X_new = selector.fit_transform(X, y)

rfe = RFE(estimator=rf, n_features_to_select=5)
rfe.fit(X, y)
```

```rust
// Scivex
use scivex_ml::feature_selection::{SelectKBest, ScoringFunction, RFE};

let mut selector = SelectKBest::new(ScoringFunction::Chi2, 10);
selector.fit(&x, &y)?;
let x_new = selector.transform(&x)?;

let mut rfe = RFE::new(Box::new(RandomForest::new(50)), 5);
rfe.fit(&x, &y)?;
let x_selected = rfe.transform(&x)?;
let ranking = rfe.ranking();
```

### Target & Ordinal Encoding

```python
# category_encoders
from category_encoders import TargetEncoder, OrdinalEncoder
te = TargetEncoder(cols=['category']).fit_transform(X, y)
oe = OrdinalEncoder(cols=['grade']).fit_transform(X)
```

```rust
// Scivex
use scivex_ml::preprocessing::{TargetEncoder, OrdinalEncoder};

let mut te = TargetEncoder::new();
te.fit(&x, &y)?;
let x_encoded = te.transform(&x)?;

let mut oe = OrdinalEncoder::new();
oe.fit(&x)?;
let x_ordinal = oe.transform(&x)?;
```

### Explainable Boosting Machine (EBM)

```python
# InterpretML
from interpret.glassbox import ExplainableBoostingClassifier
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)
```

```rust
// Scivex
use scivex_ml::ensemble::EBMClassifier;

let mut ebm = EBMClassifier::new()
    .n_bins(256)
    .learning_rate(0.01)
    .max_rounds(5000);
ebm.fit(&x_train, &y_train)?;
let preds = ebm.predict(&x_test)?;
let importances = ebm.feature_importances();
```
