import pickle
import numpy as np

# Load model
with open("model/car_model.pkl", "rb") as file:
    model = pickle.load(file)

# Example input:
# Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner

input_data = np.array([[2018, 5.5, 30000, 2, 0, 1, 0]])

prediction = model.predict(input_data)

print("Predicted Selling Price:", prediction[0])