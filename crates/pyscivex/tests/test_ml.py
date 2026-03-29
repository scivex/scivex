"""Tests for pyscivex.ml — Classical ML bindings."""

import pyscivex as sv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_classification_data():
    """Simple 2-class dataset: class 0 around (-2,-2), class 1 around (2,2)."""
    x_data = [
        -2.0, -2.0,
        -1.5, -1.8,
        -1.8, -1.5,
        -2.2, -2.1,
         2.0,  2.0,
         1.5,  1.8,
         1.8,  1.5,
         2.2,  2.1,
    ]
    y_data = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    x = sv.Tensor(x_data, [8, 2])
    y = sv.Tensor(y_data, [8])
    return x, y


def make_regression_data():
    """Simple y ≈ 2*x + 1 dataset."""
    x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y_data = [3.1, 5.0, 6.9, 9.1, 10.8, 13.2, 14.9, 17.1]
    x = sv.Tensor(x_data, [8, 1])
    y = sv.Tensor(y_data, [8])
    return x, y


# ===========================================================================
# LINEAR MODELS
# ===========================================================================


class TestLinearRegression:
    def test_fit_predict(self):
        x, y = make_regression_data()
        model = sv.ml.LinearRegression()
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape() == [8]

    def test_weights_and_bias(self):
        x, y = make_regression_data()
        model = sv.ml.LinearRegression()
        assert model.weights() is None
        assert model.bias() is None
        model.fit(x, y)
        w = model.weights()
        assert w is not None
        assert len(w) == 1
        assert abs(w[0] - 2.0) < 0.3  # slope ≈ 2
        b = model.bias()
        assert b is not None
        assert abs(b - 1.0) < 1.0  # intercept ≈ 1


class TestRidge:
    def test_fit_predict(self):
        x, y = make_regression_data()
        model = sv.ml.Ridge(alpha=1.0)
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape() == [8]


class TestLogisticRegression:
    def test_fit_predict(self):
        x, y = make_classification_data()
        model = sv.ml.LogisticRegression(learning_rate=0.5, max_iter=1000, tol=1e-8)
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape() == [8]
        data = preds.tolist()
        # First 4 should be class 0, last 4 class 1
        for v in data[:4]:
            assert v == 0.0
        for v in data[4:]:
            assert v == 1.0

    def test_predict_proba(self):
        x, y = make_classification_data()
        model = sv.ml.LogisticRegression(0.5, 1000, 1e-8)
        model.fit(x, y)
        proba = model.predict_proba(x)
        data = proba.tolist()
        assert len(data) > 0


# ===========================================================================
# TREE MODELS
# ===========================================================================


class TestDecisionTree:
    def test_classifier(self):
        x, y = make_classification_data()
        model = sv.ml.DecisionTreeClassifier(max_depth=3, min_samples_split=2)
        model.fit(x, y)
        preds = model.predict(x)
        data = preds.tolist()
        # Should perfectly classify training data
        for v in data[:4]:
            assert v == 0.0
        for v in data[4:]:
            assert v == 1.0

    def test_regressor(self):
        x, y = make_regression_data()
        model = sv.ml.DecisionTreeRegressor(max_depth=5, min_samples_split=2)
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape() == [8]


# ===========================================================================
# ENSEMBLE MODELS
# ===========================================================================


class TestRandomForest:
    def test_classifier(self):
        x, y = make_classification_data()
        model = sv.ml.RandomForestClassifier(
            n_trees=10, max_depth=3, max_features=None, seed=42
        )
        model.fit(x, y)
        preds = model.predict(x)
        data = preds.tolist()
        correct = sum(1 for a, b in zip(data, [0,0,0,0,1,1,1,1]) if a == b)
        assert correct >= 6

    def test_regressor(self):
        x, y = make_regression_data()
        model = sv.ml.RandomForestRegressor(
            n_trees=10, max_depth=5, max_features=None, seed=42
        )
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape() == [8]


class TestGradientBoosting:
    def test_classifier(self):
        x, y = make_classification_data()
        model = sv.ml.GradientBoostingClassifier(
            n_estimators=20, learning_rate=0.1, max_depth=3
        )
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape() == [8]

    def test_regressor(self):
        x, y = make_regression_data()
        model = sv.ml.GradientBoostingRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=3
        )
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape() == [8]

    def test_regressor_loss_param(self):
        x, y = make_regression_data()
        model = sv.ml.GradientBoostingRegressor(
            n_estimators=20, learning_rate=0.1, loss="mae"
        )
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape() == [8]


# ===========================================================================
# SVM
# ===========================================================================


class TestSVM:
    def test_svc_linear(self):
        x, y = make_classification_data()
        model = sv.ml.SVC(kernel="linear", c=1.0)
        model.fit(x, y)
        preds = model.predict(x)
        data = preds.tolist()
        correct = sum(1 for a, b in zip(data, [0,0,0,0,1,1,1,1]) if a == b)
        assert correct >= 6

    def test_svc_rbf(self):
        x, y = make_classification_data()
        model = sv.ml.SVC(kernel="rbf", c=1.0, gamma=0.5)
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape() == [8]

    def test_svr(self):
        x, y = make_regression_data()
        model = sv.ml.SVR(kernel="rbf", c=10.0, epsilon=0.1, gamma=0.1)
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape() == [8]


# ===========================================================================
# KNN
# ===========================================================================


class TestKNN:
    def test_classifier(self):
        x, y = make_classification_data()
        model = sv.ml.KNNClassifier(k=3)
        model.fit(x, y)
        preds = model.predict(x)
        data = preds.tolist()
        correct = sum(1 for a, b in zip(data, [0,0,0,0,1,1,1,1]) if a == b)
        assert correct >= 6

    def test_regressor(self):
        x, y = make_regression_data()
        model = sv.ml.KNNRegressor(k=3)
        model.fit(x, y)
        preds = model.predict(x)
        assert preds.shape() == [8]


# ===========================================================================
# NAIVE BAYES
# ===========================================================================


class TestGaussianNB:
    def test_fit_predict(self):
        x, y = make_classification_data()
        model = sv.ml.GaussianNB()
        model.fit(x, y)
        preds = model.predict(x)
        data = preds.tolist()
        correct = sum(1 for a, b in zip(data, [0,0,0,0,1,1,1,1]) if a == b)
        assert correct >= 6

    def test_predict_proba(self):
        x, y = make_classification_data()
        model = sv.ml.GaussianNB()
        model.fit(x, y)
        proba = model.predict_proba(x)
        data = proba.tolist()
        assert len(data) > 0


# ===========================================================================
# CLUSTERING
# ===========================================================================


class TestKMeans:
    def test_fit_predict(self):
        x, _ = make_classification_data()
        model = sv.ml.KMeans(n_clusters=2, max_iter=100, seed=42)
        model.fit(x)
        preds = model.predict(x)
        assert preds.shape() == [8]

    def test_inertia(self):
        x, _ = make_classification_data()
        model = sv.ml.KMeans(n_clusters=2)
        model.fit(x)
        inertia = model.inertia()
        assert inertia is not None
        assert inertia >= 0.0


class TestDBSCAN:
    def test_fit_labels(self):
        x, _ = make_classification_data()
        model = sv.ml.DBSCAN(eps=1.5, min_samples=2)
        model.fit(x)
        labels = model.labels()
        assert len(labels) == 8

    def test_fit_predict(self):
        x, _ = make_classification_data()
        model = sv.ml.DBSCAN(eps=1.5, min_samples=2)
        result = model.fit_predict(x)
        assert result.shape() == [8]


class TestAgglomerativeClustering:
    def test_fit_labels(self):
        x, _ = make_classification_data()
        model = sv.ml.AgglomerativeClustering(n_clusters=2, linkage="ward")
        model.fit(x)
        labels = model.labels()
        assert len(labels) == 8

    def test_linkage_types(self):
        x, _ = make_classification_data()
        for linkage in ["single", "complete", "average", "ward"]:
            model = sv.ml.AgglomerativeClustering(n_clusters=2, linkage=linkage)
            model.fit(x)
            labels = model.labels()
            assert len(labels) == 8


# ===========================================================================
# DECOMPOSITION
# ===========================================================================


class TestPCA:
    def test_fit_transform(self):
        x, _ = make_classification_data()
        model = sv.ml.PCA(n_components=1)
        result = model.fit_transform(x)
        assert result.shape() == [8, 1]

    def test_explained_variance(self):
        x, _ = make_classification_data()
        model = sv.ml.PCA(n_components=2)
        model.fit(x)
        ev = model.explained_variance()
        assert ev is not None
        assert len(ev) == 2
        evr = model.explained_variance_ratio()
        assert evr is not None
        assert abs(sum(evr) - 1.0) < 0.01


class TestTruncatedSVD:
    def test_fit_transform(self):
        x, _ = make_classification_data()
        model = sv.ml.TruncatedSVD(n_components=1)
        result = model.fit_transform(x)
        assert result.shape() == [8, 1]


class TestTSNE:
    def test_fit_transform(self):
        x, _ = make_classification_data()
        model = sv.ml.TSNE(n_components=2, perplexity=3.0)
        result = model.fit_transform(x)
        assert result.shape() == [8, 2]


# ===========================================================================
# PREPROCESSING
# ===========================================================================


class TestStandardScaler:
    def test_fit_transform(self):
        x, _ = make_classification_data()
        scaler = sv.ml.StandardScaler()
        result = scaler.fit_transform(x)
        assert result.shape() == [8, 2]

    def test_transform_after_fit(self):
        x, _ = make_classification_data()
        scaler = sv.ml.StandardScaler()
        scaler.fit(x)
        result = scaler.transform(x)
        assert result.shape() == [8, 2]


class TestMinMaxScaler:
    def test_fit_transform(self):
        x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        x = sv.Tensor(x_data, [6, 1])
        scaler = sv.ml.MinMaxScaler()
        result = scaler.fit_transform(x)
        data = result.tolist()
        # All values should be in [0, 1] (tolist returns nested for 2D)
        for row in data:
            for v in row:
                assert -0.01 <= v <= 1.01


class TestOneHotEncoder:
    def test_fit_transform(self):
        data = [0.0, 1.0, 2.0, 1.0, 0.0, 2.0]
        x = sv.Tensor(data, [6, 1])
        enc = sv.ml.OneHotEncoder()
        result = enc.fit_transform(x)
        assert result.shape()[0] == 6


class TestLabelEncoder:
    def test_fit_transform(self):
        enc = sv.ml.LabelEncoder()
        enc.fit([3.0, 1.0, 2.0, 1.0, 3.0])
        encoded = enc.transform([3.0, 1.0, 2.0, 1.0, 3.0])
        # Sorted classes: [1.0, 2.0, 3.0] → indices [2, 0, 1, 0, 2]
        assert encoded == [2, 0, 1, 0, 2]

    def test_inverse_transform(self):
        enc = sv.ml.LabelEncoder()
        enc.fit([3.0, 1.0, 2.0])
        encoded = enc.transform([1.0, 2.0, 3.0])
        decoded = enc.inverse_transform(encoded)
        assert decoded == [1.0, 2.0, 3.0]

    def test_n_classes(self):
        enc = sv.ml.LabelEncoder()
        assert enc.n_classes() is None
        enc.fit([1.0, 2.0, 3.0, 1.0])
        assert enc.n_classes() == 3


# ===========================================================================
# METRICS
# ===========================================================================


class TestMetrics:
    def test_accuracy(self):
        y_true = [0.0, 0.0, 1.0, 1.0]
        y_pred = [0.0, 1.0, 1.0, 1.0]
        assert abs(sv.ml.accuracy(y_true, y_pred) - 0.75) < 1e-10

    def test_precision(self):
        y_true = [0.0, 0.0, 1.0, 1.0]
        y_pred = [0.0, 1.0, 1.0, 1.0]
        p = sv.ml.precision(y_true, y_pred)
        # TP=2, FP=1 → precision = 2/3
        assert abs(p - 2.0 / 3.0) < 1e-10

    def test_recall(self):
        y_true = [0.0, 0.0, 1.0, 1.0]
        y_pred = [0.0, 1.0, 1.0, 1.0]
        r = sv.ml.recall(y_true, y_pred)
        # TP=2, FN=0 → recall = 1.0
        assert abs(r - 1.0) < 1e-10

    def test_f1_score(self):
        y_true = [0.0, 0.0, 1.0, 1.0]
        y_pred = [0.0, 1.0, 1.0, 1.0]
        f1 = sv.ml.f1_score(y_true, y_pred)
        assert 0.0 < f1 < 1.0

    def test_confusion_matrix(self):
        y_true = [0.0, 0.0, 1.0, 1.0]
        y_pred = [0.0, 1.0, 1.0, 1.0]
        cm = sv.ml.confusion_matrix(y_true, y_pred, 2)
        assert cm == [[1, 1], [0, 2]]

    def test_mse(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.1, 2.9]
        val = sv.ml.mse(y_true, y_pred)
        assert val < 0.02

    def test_rmse(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        val = sv.ml.rmse(y_true, y_pred)
        assert abs(val) < 1e-10

    def test_mae(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 3.5]
        val = sv.ml.mae(y_true, y_pred)
        assert abs(val - 0.5) < 1e-10

    def test_r2_score(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.1, 3.9]
        r2 = sv.ml.r2_score(y_true, y_pred)
        assert r2 > 0.9


# ===========================================================================
# MODEL SELECTION
# ===========================================================================


class TestTrainTestSplit:
    def test_basic_split(self):
        x, y = make_regression_data()
        result = sv.ml.train_test_split(x, y, test_ratio=0.25, seed=42)
        assert "x_train" in result
        assert "x_test" in result
        assert "y_train" in result
        assert "y_test" in result
        x_train = result["x_train"]
        x_test = result["x_test"]
        # 8 samples, 25% test → 2 test, 6 train
        assert x_train.shape()[0] + x_test.shape()[0] == 8

    def test_reproducible(self):
        x, y = make_regression_data()
        r1 = sv.ml.train_test_split(x, y, test_ratio=0.25, seed=42)
        r2 = sv.ml.train_test_split(x, y, test_ratio=0.25, seed=42)
        assert r1["x_train"].tolist() == r2["x_train"].tolist()


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================


class TestIntegration:
    def test_pipeline_style(self):
        """Test a typical ML workflow: split, scale, train, evaluate."""
        x, y = make_regression_data()
        split = sv.ml.train_test_split(x, y, test_ratio=0.25, seed=42)
        x_train = split["x_train"]
        x_test = split["x_test"]
        y_train = split["y_train"]
        y_test = split["y_test"]

        model = sv.ml.LinearRegression()
        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        y_test_list = y_test.tolist()
        preds_list = preds.tolist()
        r2 = sv.ml.r2_score(y_test_list, preds_list)
        assert r2 > 0.5

    def test_top_level_aliases(self):
        """LinearRegression and KMeans should be accessible from top-level."""
        model = sv.LinearRegression()
        assert model is not None
        km = sv.KMeans(n_clusters=2)
        assert km is not None

    def test_all_submodule_classes_accessible(self):
        """All ml submodule classes should be importable."""
        classes = [
            sv.ml.LinearRegression,
            sv.ml.Ridge,
            sv.ml.LogisticRegression,
            sv.ml.DecisionTreeClassifier,
            sv.ml.DecisionTreeRegressor,
            sv.ml.RandomForestClassifier,
            sv.ml.RandomForestRegressor,
            sv.ml.GradientBoostingClassifier,
            sv.ml.GradientBoostingRegressor,
            sv.ml.SVC,
            sv.ml.SVR,
            sv.ml.KNNClassifier,
            sv.ml.KNNRegressor,
            sv.ml.GaussianNB,
            sv.ml.KMeans,
            sv.ml.DBSCAN,
            sv.ml.AgglomerativeClustering,
            sv.ml.PCA,
            sv.ml.TruncatedSVD,
            sv.ml.TSNE,
            sv.ml.StandardScaler,
            sv.ml.MinMaxScaler,
            sv.ml.OneHotEncoder,
            sv.ml.LabelEncoder,
        ]
        for cls in classes:
            assert cls is not None

    def test_all_metric_functions_accessible(self):
        """All metric functions should be importable."""
        fns = [
            sv.ml.accuracy,
            sv.ml.precision,
            sv.ml.recall,
            sv.ml.f1_score,
            sv.ml.confusion_matrix,
            sv.ml.mse,
            sv.ml.rmse,
            sv.ml.mae,
            sv.ml.r2_score,
            sv.ml.train_test_split,
        ]
        for fn in fns:
            assert fn is not None
