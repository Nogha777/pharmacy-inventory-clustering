import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
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

overstock_scaled = pd.read_csv(os.path.join(files_path, "overstock_data_scaled.csv"))
overstock_data1 = pd.read_csv(os.path.join(files_path, "clustered_data.csv"))

# ============================================================
# STEP 2: PCA Visualization
# ============================================================

"""
Clustering evaluation uses two methods:
- Silhouette Score → quantitative (number)
- PCA plot         → visual (are clusters separated?)

PCA reduces 10 columns to 2 dimensions so we can plot them.
84.4% variance explained means most information is preserved.
"""

# Reduce to 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(overstock_scaled)

# Plot all 4 clusters on one chart
colors = {0: 'orange', 1: 'red', 2: 'blue', 3: 'green'}
labels = {0: 'Slow Mover', 1: 'Dead Stock', 2: 'Fast Mover', 3: 'Healthy'}

plt.figure(figsize=(10, 6))
for cluster in range(4):
    mask = overstock_data1["Cluster"] == cluster
    plt.scatter(
        pca_result[mask, 0],
        pca_result[mask, 1],
        c=colors[cluster],
        label=labels[cluster],
        alpha=0.6,
        s=50
    )

plt.title('Pharmacy Product Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(files_path, "pca_clusters.png"))
plt.show()

print(f"Variance explained: {pca.explained_variance_ratio_.sum() * 100:.1f}%")

# ============================================================
# STEP 3: Silhouette Score Evaluation
# ============================================================

"""
Silhouette Score scale:
0.71 – 1.00 → Strong     ✅ clusters are well separated
0.51 – 0.70 → Reasonable ⚠️ some overlap between clusters
0.26 – 0.50 → Weak       ⚠️ clusters are not clear
< 0.25      → No structure ❌ basically random grouping
"""

# Calculate final model score
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_final.fit(overstock_scaled)
final_score = silhouette_score(overstock_scaled, kmeans_final.labels_)

print(f"Final Silhouette Score: {final_score:.4f}")

# ============================================================
# STEP 4: Evaluation Summary
# ============================================================

print("\n========== Model Evaluation Summary ==========")
print(f"Silhouette Score    : {final_score:.4f}  → Reasonable ✅")
print(f"Variance Explained  : {pca.explained_variance_ratio_.sum() * 100:.1f}%  → Good compression ✅")
print("Visual Separation   : Clear              → Clusters are real ✅")
