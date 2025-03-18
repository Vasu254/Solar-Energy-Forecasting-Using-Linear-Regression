import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
file_path = "dataset.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Define features and target variable
X = df.drop(columns=["generated_power_kw"])
y = df["generated_power_kw"]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Save the trained model and scaler
joblib.dump(model, "final_trained_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Print evaluation results
print(f"Model Training Complete!")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")
print("Model and Scaler Saved Successfully!")
