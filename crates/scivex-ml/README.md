# scivex-ml

Classical machine learning for Scivex. Supervised and unsupervised algorithms
with a scikit-learn-inspired trait-based API.

## Highlights

- **Trait-based design** — `Transformer`, `Predictor`, `Classifier` traits
- **Linear models** — LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
- **Trees** — DecisionTreeClassifier, DecisionTreeRegressor (CART)
- **Ensembles** — RandomForest, GradientBoosting, HistGradientBoosting, AdaBoost, Stacking, CatBoost, EBM
- **SVM** — SVC, SVR with RBF, linear, and polynomial kernels
- **Neighbors** — KNNClassifier, KNNRegressor, HNSW approximate search
- **Clustering** — KMeans, DBSCAN, Agglomerative, Spectral, GMM
- **Naive Bayes** — GaussianNB, MultinomialNB
- **Preprocessing** — StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
- **Pipelines** — Pipeline builder with fit/predict/transform chaining
- **Model selection** — train_test_split, KFold, cross_val_score, GridSearchCV, RandomSearchCV
- **Metrics** — accuracy, precision, recall, F1, confusion matrix, MSE, RMSE, MAE, R², AUC-ROC
- **Explainability** — TreeSHAP for feature importance

## Usage

```rust
use scivex_ml::prelude::*;

let mut scaler = StandardScaler::new();
let x_scaled = scaler.fit_transform(&x).unwrap();

let mut model = RandomForestClassifier::new(100, 5);
model.fit(&x_train, &y_train).unwrap();
let preds = model.predict(&x_test).unwrap();

println!("Accuracy: {}", accuracy(&y_test, &preds));
println!("F1 Score: {}", f1_score(&y_test, &preds));
```

## License

MIT
