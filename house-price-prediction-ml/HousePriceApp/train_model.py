# Step 1: Train and Save the Model
import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Example training data (replace with your Week 2 dataset)
# Features: [GrLivArea, OverallQual, TotalBsmtSF, GarageCars, YearBuilt]
X_train = np.array([
    [2000, 5, 800, 2, 2000],
    [1500, 6, 700, 1, 1990],
    [2500, 7, 900, 3, 2010],
    [1800, 4, 600, 1, 1985]
])

y_train = np.array([300000, 250000, 400000, 220000])  # Prices

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as 'house_price_model.pkl'")