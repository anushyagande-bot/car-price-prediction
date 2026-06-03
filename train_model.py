import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("data/car_price_dataset.csv")
# Preprocessing
df['Car_Age'] = 2026 - df['Year']
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
joblib.dump(model, "models/car_price_model.pkl")

print("Model saved successfully!")