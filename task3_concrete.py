import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load the supplied UCI Concrete Compressive Strength dataset
concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()

# Select input features and target
target = "Concrete compressive strength"
X = concrete.drop(columns=target)
y = concrete[target]

# Reserve 20% of the rows for final testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

# Use the same five shuffled folds for all validation comparisons
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

# (a) Linear Regression baseline
linear_model = LinearRegression()

linear_rmse = -cross_val_score(
    linear_model, X_train, y_train, cv=folds, scoring=scoring
)

print("Linear fold RMSEs:", linear_rmse.round(2))
print("Linear mean CV RMSE:", linear_rmse.mean().round(2), "MPa")

# (b) Polynomial + Scaling + Ridge pipeline
ridge_model = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge())
])

# Search different polynomial degrees and alpha values
grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

search = GridSearchCV(
    ridge_model, grid, cv=folds, scoring=scoring, refit=True
)

search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print("Best mean CV RMSE:", round(-search.best_score_, 2), "MPa")

# Print the three highest-ranked parameter settings
results = pd.DataFrame(search.cv_results_)
top_three = results.sort_values("rank_test_score").head(3)

print("\nTop 3 parameter settings:")
for index, row in top_three.iterrows():
    print(
        "Degree:", row["param_poly__degree"],
        "Alpha:", row["param_ridge__alpha"],
        "Mean CV RMSE:", round(-row["mean_test_score"], 2), "MPa"
    )

# (c) Evaluate the refitted best model on the held-out test set
test_predictions = search.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_mae = mean_absolute_error(y_test, test_predictions)

print("Test RMSE:", round(test_rmse, 2), "MPa")
print("Test MAE:", round(test_mae, 2), "MPa")

# Plot actual values against predicted values
plt.scatter(y_test, test_predictions, label="Predictions")

minimum = min(y_test.min(), test_predictions.min())
maximum = max(y_test.max(), test_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    label="Ideal prediction (y = x)"
)

plt.xlabel("Actual Concrete Strength (MPa)")
plt.ylabel("Predicted Concrete Strength (MPa)")
plt.title("Actual vs Predicted Concrete Strength")
plt.legend()
plt.tight_layout()
plt.savefig("task3_concrete.png")
plt.show()

# (d) Compare baseline and tuned model
print("\nComparison:")
print("Linear baseline mean CV RMSE:", round(linear_rmse.mean(), 2), "MPa")
print("Tuned Ridge mean CV RMSE:", round(-search.best_score_, 2), "MPa")

print(
    "Example degree-2 interaction: Cement * Age. "
    "PolynomialFeatures creates interaction terms between input features."
)
print(
    "Scaling is kept inside the pipeline so it is fitted separately "
    "within each training fold. This prevents validation data from "
    "influencing the scaling step and puts features on comparable scales "
    "before Ridge regularization."
)
