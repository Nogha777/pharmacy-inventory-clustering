"""
Your Roadmap
Step 1 → Load clustered_data.csv
Step 2 → Select features and target
Step 3 → Split data (train/test)
Step 4 → Train Logistic Regression
Step 5 → Train Random Forest
Step 6 → Compare results
Step 7 → Evaluate with classification report
"""
import pandas as pd
import os
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ============================================================
# STEP 1: Load Data
# ============================================================
FILES_PATH = "/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/Overstock clustering"
clusterd_file = pd.read_csv(os.path.join(FILES_PATH, "clustered_data.csv"))
overstock_file = pd.read_csv(os.path.join(FILES_PATH, "overstock_files.csv"))

# ============================================================
# STEP 2: Select Features
# ============================================================
"""
Final Recommended Feature List
pythonfeature_cols = [
    # Raw inventory
    'Sum of Total available',
    'Sum of MAX',
    'Total Sold Q 180d',

    # Engineered
    'Price per Item',       # calculate from raw
    'Return Rate 180d',     # calculate from raw

    # Encoded product type
    'Type_ACT',
    'Type_CHR',
    'Type_OTC'
]
"""

#-------Merging Features from original df
merged_features = clusterd_file.merge(overstock_file[['Item number', 'Sum of MAX',  'Quantity 180d', 'Price 180d',
                 'Returned Q 180d']], on='Item number', how='inner')

#-------Adding price per Item
merged_features["Price per Item"] = ((merged_features["Price 180d"]/merged_features["Quantity 180d"]).round(2)).fillna(0)

#-------Adding Return Rate 180d
merged_features["Return Rate 180d"] = (((merged_features["Returned Q 180d"]/merged_features["Total Sold Q 180d"])).round(2)).fillna(0)
#-------Dealing with inf
mask_noninf = np.isinf(merged_features["Return Rate 180d"])
merged_features.loc[mask_noninf, "Return Rate 180d" ] = 1

print(merged_features.columns.tolist())

#-------Now Select the columns
n_columns = [
    'Item number', 'Product name', 'Sum of Total available', 'Sum of MAX', 'Total Sold Q 180d', 'Price per Item',
    'Return Rate 180d', 'Type_ACT', 'Type_CHR', 'Type_OTC','Cluster', 'Cluster Label'
]
merged_features = merged_features[n_columns]

# ============================================================
# STEP 3: Save
# ============================================================
merged_features.to_csv("/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/Overstock Classification/Classification Overstock.csv", index=False)









