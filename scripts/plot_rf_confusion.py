import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find project directory
ROOT = Path(__file__).resolve().parents[1]

# Load confusion matrix data
file_path = ROOT / "results" / "confusion" / "rf.csv"
cm = pd.read_csv(file_path, index_col=0)

# Create graph
fig, ax = plt.subplots(figsize=(7, 5))

im = ax.imshow(cm.values, cmap="Blues")

# Display values inside the graph
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, cm.iloc[i, j],
            ha="center",
            va="center",
            fontsize=14
        )

ax.set_xticks(range(len(cm.columns)))
ax.set_yticks(range(len(cm.index)))

ax.set_xticklabels(cm.columns)
ax.set_yticklabels(cm.index)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("Random Forest - Confusion Matrix")

plt.colorbar(im, ax=ax)
plt.tight_layout()

# Save graph
output = ROOT / "results" / "figures" / "rf_confusion_matrix.png"
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=300)

print(f"Graph saved successfully: {output}")

plt.show()