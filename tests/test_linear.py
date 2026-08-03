import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from microml.linear import LinearRegression


def test_linear_regression_oracle():
    """Verify predictions match Scikit-Learn on synthetic regression data."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

    # Train custom model
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)

    # Oracle model
    oracle = SklearnLinearRegression()
    oracle.fit(X, y)
    oracle_preds = oracle.predict(X)

    # 1. Check predictions match
    np.testing.assert_allclose(preds, oracle_preds, rtol=1e-3, atol=1e-3)

    # 2. Check weights and bias match scikit-learn parameters exactly
    np.testing.assert_allclose(model.weights, oracle.coef_, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(model.bias, oracle.intercept_, rtol=1e-3, atol=1e-3)


def test_linear_regression_exact_math():
    """Verify exact algebraic solution on a deterministic dataset with independent features: y = 3*x1 + 2*x2 + 5."""
    # Linearly independent feature matrix (full rank)
    X = np.array([
        [1.0, 2.0],
        [2.0, 5.0],
        [3.0, 3.0],
        [4.0, 8.0],
        [5.0, 1.0]
    ])
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 5.0

    model = LinearRegression()
    model.fit(X, y)

    # Verify parameters recover ground truth
    np.testing.assert_allclose(model.weights, [3.0, 2.0], atol=1e-5)
    np.testing.assert_allclose(model.bias, 5.0, atol=1e-5)

    # Verify exact predictions
    predictions = model.predict(X)
    np.testing.assert_allclose(predictions, y, atol=1e-5)


def test_single_feature_dataset():
    """Verify model handles single-column (1D feature) matrices correctly."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])  # y = 2x + 0

    model = LinearRegression()
    model.fit(X, y)

    np.testing.assert_allclose(model.weights, [2.0], atol=1e-5)
    np.testing.assert_allclose(model.bias, 0.0, atol=1e-5)


def test_unfitted_model_raises_error():
    """Verify that calling predict before fit raises a RuntimeError."""
    model = LinearRegression()
    X = np.array([[1.0, 2.0]])

    with pytest.raises(RuntimeError):
        model.predict(X)