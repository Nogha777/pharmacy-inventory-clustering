from operator import index

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

FILES_PATH = "/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/Overstock clustering"
overstock_files = pd.read_csv(os.path.join(FILES_PATH, "overstock_files.csv"))

print(overstock_files.shape)

# ============================================================
# STEP 2: Handle INF  and NaN and (-)ve Values
# ============================================================

"""
INF in Overstock Ratio and Days of Supply means:
products have stock but were never sold in the past 180 days.
We penalize them with max*2 so the ML model understands
they are extreme overstock problems.
"""

max_or = overstock_files[overstock_files["Overstock Ratio"] != np.inf]["Overstock Ratio"].max()
max_dos = overstock_files[overstock_files["Days of Supply"] != np.inf]["Days of Supply"].max()

overstock_files["Overstock Ratio"] = overstock_files["Overstock Ratio"].replace(np.inf, max_or * 2)
overstock_files["Days of Supply"] = overstock_files["Days of Supply"].replace(np.inf, max_dos * 2)

#---------now for Nan means that there is no stock and no sales
#which means inactive product so we are going to remove all nan
overstock_files = overstock_files.dropna()

#----------now for negative values comes from  Sum of Total available negative
#it happenned when we have zero stock but costomer wants to buy it now but we can afford it next time

mask_non0 = (overstock_files["Total Sold Q 180d"] != 0) & (overstock_files["Sum of Total available"] < 0 )
overstock_files.loc[mask_non0, "Sum of Total available"] = overstock_files.loc[mask_non0,"Total Sold Q 180d"]

# There are nigative in the stock and no sales so this is inactive item
overstock_files = overstock_files.drop(index= [297, 1396])

# still negative valuse in Days of Supply, Overstock Ratio, Sell Through Rate
#----------- so lets calculat them again since we fininsh cleaning Sum of Total available
#mask to only calculate the negative once
mask_nonnegative = overstock_files["Days of Supply"] < 0

#------
overstock_files.loc[mask_nonnegative, "Sell Through Rate"] = (
    overstock_files.loc[mask_nonnegative, "Total Sold Q 180d"] /
    overstock_files.loc[mask_nonnegative, "Sum of Total available"]
)

#------
overstock_files.loc[mask_nonnegative, "Overstock Ratio"] = (
    overstock_files.loc[mask_nonnegative, "Sum of Total available"] /
    overstock_files.loc[mask_nonnegative, "Total Sold Q 180d"]
)

#------
overstock_files.loc[mask_nonnegative, "Days of Supply"] = (
    overstock_files.loc[mask_nonnegative, "Sum of Total available"] /
    (overstock_files.loc[mask_nonnegative, "Total Sold Q 180d"] / 180)
)

# ============================================================
# STEP 3: Encode Product Type
# ============================================================

"""
Is there a natural order between categories?
       YES → Label Encoding
       NO  → One Hot Encoding
OTC, ACT, CHR have no natural order → One Hot Encoding
"""

product_dummies = pd.get_dummies(overstock_files['Product Type'], prefix='Type').astype(int)
overstock_files = pd.concat([overstock_files.drop('Product Type', axis=1), product_dummies], axis=1)
# ============================================================
# STEP 4: Final Validation Check
# ============================================================

print("Shape:", overstock_files.shape)
print("\nNull values:\n", overstock_files.isnull().sum())
print("\nInf values:")
print("  Days of Supply inf:    ", (overstock_files['Days of Supply'] == np.inf).sum())
print("  Overstock Ratio inf:   ", (overstock_files['Overstock Ratio'] == np.inf).sum())
print("\nNegative values:")
print("  Overstock Ratio negative:", (overstock_files['Overstock Ratio'] < 0).sum())
print("  Days of Supply negative: ", (overstock_files['Days of Supply'] < 0).sum())
print("\nData types:\n", overstock_files.dtypes)

# ============================================================
# STEP 5: Save
# ============================================================

overstock_files.to_csv(os.path.join(FILES_PATH, "overstock_files.csv"), index=False)
print("\nFile saved ")
print(overstock_files.shape)


