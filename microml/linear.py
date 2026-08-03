import numpy as np


class LinearRegression:
    """
    Ordinary Least Squares (OLS) Linear Regression using NumPy.
    """

    def __init__(self):
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the model using the closed-form Normal Equation.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, _ = X.shape

        # Add bias column (vector of 1s) to feature matrix X
        X_b = np.c_[np.ones((n_samples, 1)), X]

        # Solve for parameters theta using pseudo-inverse for numerical stability
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = float(theta[0])
        self.weights = theta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for input feature matrix X.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")

        X = np.asarray(X, dtype=np.float64)
        return (X @ self.weights) + self.bias