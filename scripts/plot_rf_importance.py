import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Project directory
ROOT = Path(__file__).resolve().parents[1]

# Load saved model
model_path = ROOT / "artifacts" / "models" / "rf.joblib"
saved_data = joblib.load(model_path)

# Extract classifier and feature names
model = saved_data["model"]
features = saved_data["features"]

# Extract feature importance
importances = model.feature_importances_

# Create DataFrame
df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
})

# Sort by importance
df = df.sort_values(
    by="Importance",
    ascending=True
)

# Create graph
plt.figure(figsize=(10, 8))

plt.barh(
    df["Feature"],
    df["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest - Feature Importance")

plt.tight_layout()

# Save graph
output = ROOT / "results" / "figures" / "rf_feature_importance.png"
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=300)

print(f"Graph saved successfully: {output}")

plt.close()
