import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load data
data = pd.read_csv('data/processed/merged_data.csv')

# Define features and target
X = data[['NewCases', 'Humidity_x', 'PopulationDensity', 'Temperature', 'Rainfall']]
y = data['OutbreakRisk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
os.makedirs('models', exist_ok=True)  # Create folder if missing
joblib.dump(model, 'models/outbreak_model.pkl')
print("Model saved successfully!")