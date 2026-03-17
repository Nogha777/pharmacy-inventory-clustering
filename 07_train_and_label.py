import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
overstock_data1 = pd.read_csv(os.path.join(files_path, "overstock_files2.csv"))

# ============================================================
# STEP 2: Train KMeans Model
# ============================================================

# k=4 was chosen based on Elbow Method + Silhouette Score results
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_final.fit(overstock_scaled)

# Evaluate final model
final_score = silhouette_score(overstock_scaled, kmeans_final.labels_)
print(f"Final Silhouette Score: {final_score:.4f}")

# ============================================================
# STEP 3: Add Cluster Labels to Original Dataframe
# ============================================================

# Labels added to original (not scaled) dataframe for readable analysis
overstock_data1['Cluster'] = kmeans_final.labels_

# ============================================================
# STEP 4: Analyze Clusters
# ============================================================

cluster_analysis = overstock_data1.groupby('Cluster')[[
    'Sum of Total available', 'Price per Item', 'Total Sold Q 180d',
    'Sell Through Rate', 'Overstock Ratio', 'Days of Supply', 'Return Rate 180d'
]].mean().round(2)

print("\nCluster Analysis:")
print(cluster_analysis)

# ============================================================
# STEP 5: Map Cluster Numbers to Business Labels
# ============================================================

"""
Cluster labels based on analysis:
0 = Slow Mover   → low sales, slightly overstocked
1 = Dead Stock   → zero sales, extreme overstock, 100% return rate
2 = Fast Mover   → high sales, bestsellers
3 = Healthy      → balanced inventory, good sell through rate

.map() replaces every cluster number using a dictionary as a translation table
"""

cluster_labels = {
    0: "Slow Mover",
    1: "Dead Stock",
    2: "Fast Mover",
    3: "Healthy"
}

overstock_data1['Cluster Label'] = overstock_data1['Cluster'].map(cluster_labels)

# ============================================================
# STEP 6: Final Distribution Check
# ============================================================

print("\nCluster Distribution:")
print(overstock_data1['Cluster Label'].value_counts())

# ============================================================
# STEP 7: Save
# ============================================================

overstock_data1.to_csv(os.path.join(files_path, "clustered_data.csv"), index=False)
print("\n File saved")
