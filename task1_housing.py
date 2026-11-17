import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the supplied UCI Real Estate dataset
homes = pd.read_csv("real_estate.csv")

# Convert distance from metres to units of 100 metres
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100

# Select input and target
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# Fit Linear Regression
model = LinearRegression()
model.fit(X, y)

# Print slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict price per unit area at 500 metres
new_home = pd.DataFrame({"distance_100m": [5]})
prediction = model.predict(new_home)
print("Predicted price at 500 m:", prediction[0])

# Explain the fitted price change
print("Interpretation:")
print(
    "For every additional 100 metres from the nearest MRT station, "
    "the predicted price per unit area changes by",
    round(model.coef_[0], 2),
    "price units."
)

# Plot observations and fitted regression line
plt.scatter(X["distance_100m"], y, label="Observed data")
plt.plot(
    X["distance_100m"],
    model.predict(X),
    color="red",
    label="Linear Regression line"
)
plt.xlabel("Distance to nearest MRT station (100 metres)")
plt.ylabel("House price per unit area (10,000 New Taiwan dollars)")
plt.title("Housing Price vs Distance to MRT Station")
plt.legend()
plt.tight_layout()
plt.savefig("task1_housing.png")
plt.show()
