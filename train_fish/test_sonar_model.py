# test_sonar_model.py
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# === 1. Load model ===
model_filename = "sonar_model_pr.joblib"
model = joblib.load(model_filename)
print(f"Loaded trained model: {model_filename}")

# === 2. Define test data ===
# Replace with your actual measured fish counts and average scores
# (Can reuse from training phase if you have only 3 samples)
X_test = np.array([
    [44350],  # from fish_150
    [54311],  # from fish_300
    [5915]    # from fishingpond
])
y_true = np.array([150, 300, 0])

# === 3. Predict ===
y_pred = model.predict(X_test)

# === 4. Evaluate ===
mae = mean_absolute_error(y_true, y_pred)
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# === 5. Print detailed comparison ===
print("\nPredicted vs Actual:")
for actual, pred in zip(y_true, y_pred):
    print(f"  Actual: {actual:>4} | Predicted: {pred:>7.2f}")

# === 6. Visualize ===
plt.figure(figsize=(6, 5))
plt.scatter(y_true, y_pred, color="blue", label="Predictions")
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
plt.xlabel("Actual Fish Count")
plt.ylabel("Predicted Fish Count")
plt.title("Polynomial Regression Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pred_vs_actual.png", dpi=300)
plt.show()

print("\nPlot saved as 'pred_vs_actual.png'")
