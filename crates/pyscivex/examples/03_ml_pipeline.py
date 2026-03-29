"""ML pipeline — train/test split, scaling, model training, evaluation."""
import pyscivex as sv

# Generate sample data: y = 2*x1 + 3*x2 + noise
x = sv.Tensor([
    [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
    [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0],
    [9.0, 10.0], [10.0, 11.0],
])
y = sv.Tensor([8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0])

# Train/test split
x_train, x_test, y_train, y_test = sv.ml.train_test_split(x, y, test_size=0.3, seed=42)
print(f"Train: {x_train.shape()}, Test: {x_test.shape()}")

# Preprocessing
scaler = sv.ml.StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train model
model = sv.ml.LinearRegression()
model.fit(x_train_scaled, y_train)

# Predict
predictions = model.predict(x_test_scaled)
print("Predictions:", predictions)

# Evaluate
print(f"MSE:  {sv.ml.mse(y_test, predictions):.4f}")
print(f"RMSE: {sv.ml.rmse(y_test, predictions):.4f}")
print(f"MAE:  {sv.ml.mae(y_test, predictions):.4f}")
print(f"R2:   {sv.ml.r2_score(y_test, predictions):.4f}")

# Classification example
clf = sv.ml.RandomForestClassifier(n_trees=10, max_depth=5, seed=42)
x_clf = sv.Tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y_clf = sv.Tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
clf.fit(x_clf, y_clf)
pred_clf = clf.predict(sv.Tensor([[2.5], [4.5]]))
print("Classification:", pred_clf)
