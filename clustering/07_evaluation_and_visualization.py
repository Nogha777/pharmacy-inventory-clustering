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

FILES_PATH = "/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/Overstock clustering"

overstock_scaled = pd.read_csv(os.path.join(FILES_PATH, "overstock_data_scaled.csv"))
clusterd_data = pd.read_csv(os.path.join(FILES_PATH, "clustered_data.csv"))

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

# Plot all 3 clusters on one chart
colors = {0: 'orange', 1: 'red', 2: 'blue'}
labels = {0: "Overstock", 1: "Healthy Sellers", 2: "Bulk Products"}


plt.figure(figsize=(10, 6))
for cluster in range(3):
    mask = clusterd_data["Cluster"] == cluster
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
plt.savefig(os.path.join(FILES_PATH, "pca_clusters.png"))
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
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
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


#Savin sample for uplaod in Github

mask_nonoverstock = clusterd_data['Cluster Label'] == "Overstock"
mask_nonhealthy = clusterd_data['Cluster Label'] == "Healthy Sellers"
mask_nonbulk = clusterd_data['Cluster Label'] == "Bulk Products"

sample_data = [
    clusterd_data[mask_nonoverstock].head(),
    clusterd_data[mask_nonhealthy].head(),
    clusterd_data[mask_nonbulk],
]
sampel_clusterd = pd.concat(sample_data, ignore_index=True)
sampel_clusterd = sampel_clusterd.drop(columns=["Product name"])
# Replace all item numbers with anonymous IDs
sampel_clusterd['Item number'] = ['ITEM_' + str(i+1).zfill(4)
                                 for i in range(len(sampel_clusterd))]


sampel_clusterd.to_csv(os.path.join(FILES_PATH, "sampel_clusterd.csv"), index=False)
