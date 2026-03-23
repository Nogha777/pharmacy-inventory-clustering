import pandas as pd
import os
from sklearn.cluster import KMeans



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
overstock_files = pd.read_csv(os.path.join(FILES_PATH, "overstock_files.csv"))

# ============================================================
# STEP 2: Creat original file
# ============================================================
#After traning we will add the clusterd data in the original file not the scaled data
n_columns = [
    'Item number', 'Product name', 'Sum of Total available',
    'Total Sold Q 180d', 'Sell Through Rate', 'Overstock Ratio',
    'Days of Supply', 'Type_ACT', 'Type_CHR', 'Type_OTC'
]
overstock_clusterd = overstock_files[n_columns]

# ============================================================
# STEP 3: Train KMeans Model
# ============================================================

# k=3 was chosen based on Elbow Method + Silhouette Score results
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_final.fit(overstock_scaled)

# ============================================================
# STEP 4: Add Cluster Labels to Original Dataframe
# ============================================================

# Labels added to original (not scaled) dataframe for readable analysis
overstock_clusterd['Cluster'] = kmeans_final.labels_

# ============================================================
# STEP 4: Analyze Clusters
# ============================================================

cluster_analysis = overstock_clusterd.groupby('Cluster')[[
    'Sum of Total available','Total Sold Q 180d',
    'Sell Through Rate', 'Overstock Ratio',
    'Days of Supply', 'Type_ACT', 'Type_CHR', 'Type_OTC'
]].mean().round(2)

print("\nCluster Analysis:")
print(cluster_analysis)

# ============================================================
# STEP 5: Map Cluster Numbers to Business Labels
# ============================================================

"""
Cluster labels based on analysis:
1 = Dead Stock   → zero sales, extreme overstock, 100% return rate
2 = Fast Mover   → high sales, bestsellers
3 = Healthy      → balanced inventory, good sell through rate

.map() replaces every cluster number using a dictionary as a translation table
"""
"""
Overstock 
Products that are just sitting on the shelf and barely selling. At this rate, it will take 5 years to sell them all.

Good Sellers 
Products that are selling faster than we can stock them. Healthy and no action needed.

Bulk Products 
Products that sell a lot but we also buy a lot of. They're fine but need to be watched so we don't over-order.
"""
cluster_labels = {
    0: "Overstock",
    1: "Healthy Sellers",
    2: "Bulk Products"
}

overstock_clusterd['Cluster Label'] = overstock_clusterd['Cluster'].map(cluster_labels)

# ============================================================
# STEP 6: Final Distribution Check
# ============================================================

print("\nCluster Distribution:")
print(overstock_clusterd['Cluster Label'].value_counts())

# ============================================================
# STEP 7: Save
# ============================================================

overstock_clusterd.to_csv(os.path.join(FILES_PATH, "clustered_data.csv"), index=False)
print("\n File saved")
