
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
yachts = pd.read_csv(
    "yacht_hydrodynamics.data",
    sep=r"\s+",
    header=None
)

# Select input and target
X = yachts[[5]]
y = yachts[6]

# Linear Regression
linear_model = LinearRegression()

# Quadratic Regression
quadratic_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression())
])

# Five-fold cross-validation
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

linear_rmse = -cross_val_score(
    linear_model, X, y, cv=folds, scoring=scoring
)

quadratic_rmse = -cross_val_score(
    quadratic_model, X, y, cv=folds, scoring=scoring
)

# Print results
print("Linear fold RMSEs:", linear_rmse.round(2))
print("Linear mean RMSE:", linear_rmse.mean().round(2))

print("Quadratic fold RMSEs:", quadratic_rmse.round(2))
print("Quadratic mean RMSE:", quadratic_rmse.mean().round(2))

if quadratic_rmse.mean() < linear_rmse.mean():
    print("Better model: Quadratic Regression")
else:
    print("Better model: Linear Regression")

# Fit both models on all available data for visualization
linear_model.fit(X, y)
quadratic_model.fit(X, y)

# Generate smooth input values
X_plot = pd.DataFrame({
    5: np.linspace(X[5].min(), X[5].max(), 300)
})

# Plot observations and regression lines
plt.scatter(X[5], y, label="Actual Data", alpha=0.6)

plt.plot(
    X_plot[5],
    linear_model.predict(X_plot),
    label="Linear Regression"
)

plt.plot(
    X_plot[5],
    quadratic_model.predict(X_plot),
    label="Quadratic Regression"
)

plt.xlabel("Froude Number")
plt.ylabel("Residuary Resistance")
plt.title("Linear vs Quadratic Regression - Yacht")
plt.legend()
plt.tight_layout()

plt.savefig("task2_yacht.png")
plt.show()
