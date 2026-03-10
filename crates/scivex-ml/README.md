# scivex-ml

Classical machine learning for Scivex. Supervised and unsupervised algorithms
with a scikit-learn-inspired trait-based API.

## Highlights

- **Trait-based design** — `Transformer`, `Predictor`, `Classifier` traits
- **Linear models** — LinearRegression, Ridge, LogisticRegression
- **Trees** — DecisionTreeClassifier, DecisionTreeRegressor (CART)
- **Ensembles** — RandomForestClassifier, RandomForestRegressor
- **Neighbors** — KNNClassifier, KNNRegressor
- **Clustering** — KMeans (Lloyd's algorithm)
- **Naive Bayes** — GaussianNB
- **Preprocessing** — StandardScaler, MinMaxScaler, LabelEncoder
- **Metrics** — accuracy, precision, recall, F1, confusion matrix, MSE, RMSE, MAE, R²
- **Model selection** — train_test_split, KFold, cross_val_score

## Usage

```rust
use scivex_ml::prelude::*;

// Preprocessing
let mut scaler = StandardScaler::new();
let x_scaled = scaler.fit_transform(&x).unwrap();

// Train a model
let mut model = RandomForestClassifier::new(100, 5);
model.fit(&x_train, &y_train).unwrap();
let preds = model.predict(&x_test).unwrap();

// Evaluate
println!("Accuracy: {}", accuracy(&y_test, &preds));
println!("F1 Score: {}", f1_score(&y_test, &preds));

// Cross-validation
let scores = cross_val_score(&model, &x, &y, 5);
```

## License

MIT
