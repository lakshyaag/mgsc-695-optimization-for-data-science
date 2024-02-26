import numpy as np  # Using numpy for linear algebra
import pandas as pd
from rich import print
from sklearn.linear_model import LinearRegression

data = pd.read_csv("advertising.csv").rename(columns={"Unnamed: 0": "id"})

X = data[["TV", "Radio", "Newspaper"]].to_numpy()
Y = data["Sales"].to_numpy()


# Using matrix algebra
# --------------------
def fit_linear_regression(
    X: np.ndarray, Y: np.ndarray, fit_intercept: bool = True
) -> np.ndarray:
    """
    Fits a linear regression model to the data using matrix algebra.

    Parameters
    ----------
    X : np.ndarray
        The independent variables.
    Y : np.ndarray
        The dependent variable.
    fit_intercept : bool
        Whether to fit an intercept or not, defaults to True.

    Returns
    -------
    :math:`beta`: np.ndarray
        The fitted parameters.
    """

    if fit_intercept:
        X = np.hstack((np.ones((X.shape[0], 1)), X))

    print(f"Shape of X: {X.shape}", f"Shape of Y: {Y.shape}", sep="\n")

    return np.linalg.inv(X.T @ X) @ (X.T @ Y)


coefficients_matrix_algebra = fit_linear_regression(X, Y, fit_intercept=True)

# Using scikit-learn
# ------------------
lr = LinearRegression(fit_intercept=True).fit(X, Y)

coefficients_sklearn = np.hstack((lr.intercept_, lr.coef_))

# Comparing the results
# ---------------------
print(f"Matrix algebra: {coefficients_matrix_algebra}")
print(f"Scikit-learn: {coefficients_sklearn}")
print(
    f"Are they equal? {np.allclose(coefficients_matrix_algebra, coefficients_sklearn)}"
)

assert np.allclose(coefficients_matrix_algebra, coefficients_sklearn)
