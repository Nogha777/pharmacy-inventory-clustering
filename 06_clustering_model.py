import pandas as pd
import os
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Show all columns and rows in console output
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# ============================================================
# STEP 1: Load Data
# ============================================================

files_path = "/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/currently_working"
overstock_data = pd.read_csv(os.path.join(files_path, "overstock_data1.csv"))

# Remove unwanted index column from previous save (forgot index=False)
overstock_data = overstock_data.drop(columns=["Unnamed: 0"])

# ============================================================
# STEP 2: Scale Features
# ============================================================

"""
Common scaling techniques:
- Normalization (Min-Max): rescales to [0,1] → sensitive to outliers
- Standardization (Z-score): mean=0, std=1 → still affected by outliers
- Robust Scaling: uses median and IQR → resistant to outliers ✅

My dataset has outliers (confirmed from describe()) so Robust Scaling is used.
"""

scaler = RobustScaler()
overstock_scaled = pd.DataFrame(
    scaler.fit_transform(overstock_data),
    columns=overstock_data.columns
)

overstock_scaled.to_csv(os.path.join(files_path, "overstock_data_scaled.csv"), index=False)

# ============================================================
# STEP 3: Find Optimal K
# ============================================================

"""
Elbow Method + Silhouette Score — used together to find best k

Inertia (Elbow Method):
- Measures total distance between each point and its cluster center
- Low inertia  = tight clusters ✅
- High inertia = loose clusters ❌
- Pick k where the curve stops dropping sharply (the "elbow")

Silhouette Score:
- Measures how well separated clusters are
- Range: -1 to 1 → closer to 1 is better
- Pick k with the highest score

For both methods k=4 is appropriate:
→ Healthy / Overstocked / Critical / Slow Mover
"""

inertia = []
scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(overstock_scaled)
    inertia.append(kmeans.inertia_)
    scores.append(silhouette_score(overstock_scaled, labels))

# --- Elbow Plot ---
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method — Finding Optimal k')
plt.xticks(k_range)
plt.tight_layout()
plt.savefig(os.path.join(files_path, "elbow_plot.png"))
plt.show()

# --- Silhouette Plot ---
plt.figure(figsize=(8, 5))
plt.plot(k_range, scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score — Finding Optimal k')
plt.xticks(k_range)
plt.tight_layout()
plt.savefig(os.path.join(files_path, "silhouette_plot.png"))
plt.show()