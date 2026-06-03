import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import pickle

# Load dataset
df = pd.read_csv("data/car_price_dataset.csv")

# Remove extra spaces
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

# Drop Car_Name
df = df.drop(['Car_Name'], axis=1)

# Encode categorical columns
le = LabelEncoder()

df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Selling_type'] = le.fit_transform(df['Selling_type'])
df['Transmission'] = le.fit_transform(df['Transmission'])

# Features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)
print("Model Accuracy:", score)

# Save model
with open("model/car_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")