import pandas as pd
import os
import numpy as np

# Show all columns and rows in console output
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ============================================================
# STEP 1: Load Data
# ============================================================

files_path = "/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/currently_working"
overstock_data = pd.read_csv(os.path.join(files_path, "returner_data.csv"))

# Remove unwanted columns
overstock_data = overstock_data.drop(columns=["Unnamed: 0", "High Return"])

# ============================================================
# STEP 2: Handle Negative Values in Overstock Ratio
# ============================================================

# Investigated rows 76, 126, 559 — confirmed bad data, removed
overstock_data = overstock_data.drop(index=[76, 126, 559]).reset_index(drop=True)

# ============================================================
# STEP 3: Handle INF Values
# ============================================================

"""
INF in Overstock Ratio and Days of Supply means:
products have stock but were never sold in the past 180 days.
We penalize them with max*2 so the ML model understands
they are extreme overstock problems.
"""

max_or = overstock_data[overstock_data["Overstock Ratio"] != np.inf]["Overstock Ratio"].max()
max_dos = overstock_data[overstock_data["Days of Supply"] != np.inf]["Days of Supply"].max()

overstock_data["Overstock Ratio"] = overstock_data["Overstock Ratio"].replace(np.inf, max_or * 2)
overstock_data["Days of Supply"] = overstock_data["Days of Supply"].replace(np.inf, max_dos * 2)

# ============================================================
# STEP 4: Encode Product Type
# ============================================================

"""
Is there a natural order between categories?
       YES → Label Encoding
       NO  → One Hot Encoding
OTC, ACT, CHR have no natural order → One Hot Encoding
"""

product_dummies = pd.get_dummies(overstock_data['Product Type'], prefix='Type').astype(int)
overstock_data = pd.concat([overstock_data.drop('Product Type', axis=1), product_dummies], axis=1)

# ============================================================
# STEP 5: Final Validation Check
# ============================================================

print("Shape:", overstock_data.shape)
print("\nNull values:\n", overstock_data.isnull().sum())
print("\nInf values:")
print("  Days of Supply inf:    ", (overstock_data['Days of Supply'] == np.inf).sum())
print("  Overstock Ratio inf:   ", (overstock_data['Overstock Ratio'] == np.inf).sum())
print("\nNegative values:")
print("  Overstock Ratio negative:", (overstock_data['Overstock Ratio'] < 0).sum())
print("  Days of Supply negative: ", (overstock_data['Days of Supply'] < 0).sum())
print("\nData types:\n", overstock_data.dtypes)

# ============================================================
# STEP 6: Save
# ============================================================

overstock_data.to_csv(os.path.join(files_path, "overstock_data1.csv"), index=False)
print("\nFile saved ")
