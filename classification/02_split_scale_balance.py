import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ============================================================
# STEP 1: Load Data
# ============================================================
FILES_PATH = "/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/Overstock Classification"
classification_data = pd.read_csv(os.path.join(FILES_PATH, "Classification Overstock.csv"))

# ============================================================
# STEP 2: Define X(features) and y(target)
# ============================================================
features_columns = [
       'Sum of Total available', 'Sum of MAX', 'Total Sold Q 180d', 'Price per Item',
       'Return Rate 180d', 'Type_ACT', 'Type_CHR', 'Type_OTC'
]

X = classification_data[features_columns] # capital X → matrix (2D, many columns)
y = classification_data['Cluster']        # lowercase y → vector (1D, one column)

#-------check
# print(f"X shape: {X.shape}")
# print(f"y shape: {y.shape}")
# print(f"\nClass distribution:")
# print(y.value_counts())

# ============================================================
# STEP 3: Splitting the Data
# ============================================================
"""
Step 1 → Split data FIRST
Step 2 → Apply SMOTE on TRAINING set only ← important!
Step 3 → Never apply SMOTE on validation or test set
"""

#-------First split, 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y # stratify=y  keeps class balance in each split
)

#-------Second split, split temp into 50/50 → 20% validation, 20% test
"""
here I cannot add stratify=y because Cluster
0    1153
1      32
2       3
in cluster 2 I only have 3 items it will not split evenly so I hvae to choose
so I chose that only test values contain cluster 2 and validation no
"""
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

#-------check
# print(f"y_train\n{y_train.value_counts()}")
# print(f"y_val\n{y_val.value_counts()}")
# print(f"y_test\n{y_test.value_counts()}")
#
# print(f"Train:      {len(X_train)} rows")
# print(f"Validation: {len(X_val)}   rows")
# print(f"Test:       {len(X_test)}  rows")
# print(f"Total:      {len(X_train) + len(X_val) + len(X_test)} rows")

# ============================================================
# STEP 4: Scale Features
# ============================================================
# Fit ONLY on training data to avoid Data leakage
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train) # remember these are not df these are arrays
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 5: Apply SMOTE on Training Data Only
# ============================================================
#-------Note using SMOTE is considered limitation because it creats synthatic samples
#_Cluster 2 has 2 real + 689 synthetic almost entirely fake data
#_model will try but results for Cluster 2 unreliable
smote = SMOTE(random_state=42,  k_neighbors=1)
# k_neighbors=1 because Cluster 2 has only 2 training samples
# default k=5 would crash with only 2 samples!
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

#-------check
# print("\nBefore SMOTE:")
# print(y_train.value_counts())
#
# print("\nAfter SMOTE:")
# print(pd.DataFrame(y_train_balanced).value_counts())

# print(X_train.columns.tolist())