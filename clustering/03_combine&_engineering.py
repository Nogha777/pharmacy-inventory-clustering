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
m_otc = pd.read_csv(os.path.join(FILES_PATH, "merged_otc.csv"))
m_acute = pd.read_csv(os.path.join(FILES_PATH, "merged_acute.csv"))
m_chronic = pd.read_csv(os.path.join(FILES_PATH, "merged_chronic.csv"))

overstock_files = pd.concat([m_otc, m_acute, m_chronic], ignore_index=True)

# ============================================================
# STEP 2: Feature Engineering
# ============================================================

# --- Feature 1: Total Sold Q 180d ---
#First I need to calculate Returned Q 180d"
#Before adding "Returned Q 180d" I need to fix "Price 180d"
#-because most of it zero it will give inf and it is wrong
mask = (overstock_files["Price 180d"] == 0.0) & (overstock_files["Quantity 30d"] != 0)
'df.loc[mask, "Price 180d"]   # give me the "Price 180d" column where mask is True'
overstock_files.loc[mask, "Price 180d"] = (overstock_files.loc[mask,"Price 30d"]/overstock_files.loc[mask,"Quantity 30d"]) * overstock_files.loc[mask,"Quantity 180d"]

#Thre row 748  there is mistake in data collection so need to be deleted
overstock_files = overstock_files.drop(index=748).reset_index(drop=True)

#Thre row 128 and 321  there is missing  Price 180d  in data collection so need to be fixed
overstock_files.at[128, "Price 180d"] = 19.8
overstock_files.at[321, "Price 180d"] = 10.1


price_item180d = overstock_files["Price 180d"]/overstock_files["Quantity 180d"]
overstock_files["Returned Q 180d"] = ((overstock_files["Returned Price In 180d"]/price_item180d).fillna(0))
#Now lets add Total Sold Q 180d column
overstock_files["Total Sold Q 180d"]= (overstock_files["Quantity 180d"] - overstock_files["Returned Q 180d"])

# --- Feature 2: Sell Through Rate ---
# What percentage of available stock was actually sold
# Formula: Total Sold / Total Available
#
# Edge case fix: some rows have Sum of Total available == 0
# but still show sales — data was likely updated later.
# Fix: replace 0 available with actual sold quantity for those rows.
masked_non0 = (
        (overstock_files["Sum of Total available"] == 0) &
        (overstock_files["Total Sold Q 180d"] != 0)
)

# Convert to float FIRST to accept decimal values because "Sum of Total available" contains only integer
overstock_files["Sum of Total available"] = overstock_files["Sum of Total available"].astype(float)

#.loc[...] — a way to select rows and columns by label
overstock_files.loc[masked_non0, "Sum of Total available"] = overstock_files.loc[masked_non0, "Total Sold Q 180d"]

overstock_files["Sell Through Rate"] = (
        overstock_files["Total Sold Q 180d"] / overstock_files["Sum of Total available"]
)
# Result is between 0 and 1 → e.g. 0.8 means 80% of stock was sold
# NaN = 0 / 0 → inactive/ghost item — handled later

# --- Feature 3: Overstock Ratio ---
# How much stock is left compared to what was sold
# Formula: Total Available / Total Sold
overstock_files["Overstock Ratio"] = (
        overstock_files["Sum of Total available"] / overstock_files["Total Sold Q 180d"]
)
# Result > 1 means more stock than sold → e.g. 3.0 means 3x more stock than sold
# inf → 0 sold quantity (extreme overstock)
# NaN → inactive item (0/0)

# --- Feature 4: Days of Supply ---
# How many days current stock will last at the current sales rate
# Formula: Total Available / Daily Sales Rate
daily_sales = overstock_files["Total Sold Q 180d"] / 180
overstock_files["Days of Supply"] = overstock_files["Sum of Total available"] / daily_sales
# e.g. 360 means current stock will last 360 days at current sales pace
# inf and NaN interpretation is the same as above

# ============================================================
# STEP 3: Round Engineered Features
# ============================================================

round_col = ["Sell Through Rate", "Overstock Ratio", "Days of Supply"]
overstock_files[round_col] = overstock_files[round_col].round(0)

# ============================================================
# STEP 4: fix inf
# ============================================================

# Most Price 180d values are zero which would cause inf in calculations
# Fix: estimate Price 180d using the price per item from 30d window
# Only applied where Price 180d == 0 and Quantity 30d != 0
mask = (overstock_files["Price 180d"] == 0.0) & (overstock_files["Quantity 30d"] != 0)
overstock_files.loc[mask, "Price 180d"] = (
        (overstock_files.loc[mask, "Price 30d"] / overstock_files.loc[mask, "Quantity 30d"])
        * overstock_files.loc[mask, "Quantity 180d"]
)

# Row 748 — data collection mistake, remove it
overstock_files = overstock_files.drop(index=748).reset_index(drop=True)

# Rows 128 and 321 — missing Price 180d, fixed manually
overstock_files.at[128, "Price 180d"] = 19.8
overstock_files.at[321, "Price 180d"] = 10.1


# ============================================================
# STEP 5: Reorder Columns
# ============================================================

# Reordering all at once is cleaner than popping one column at a time
new_order = [
    'Item number', 'Product name', 'Product Type',
    'Sum of Total available', 'Sum of MAX',
    'Quantity 30d', 'Price 30d', 'Returned Price In 30d','Quantity 180d',
    'Price 180d', 'Returned Q 180d', 'Returned Price In 180d', 'Total Sold Q 180d',
    'Sell Through Rate', 'Overstock Ratio', 'Days of Supply'
]

overstock_files = overstock_files[new_order]

# ============================================================
# STEP 6: Round Calculated Columns
# ============================================================

round_columns = [
    'Price 30d', 'Price 180d', 'Returned Q 180d', 'Total Sold Q 180d'
]
overstock_files[round_columns] = overstock_files[round_columns].round(0)
# ============================================================
# STEP 7: Save
# ============================================================

overstock_files.to_csv(os.path.join(FILES_PATH, "overstock_files.csv"), index=False)
print("File saved")
print("\nNull value check:")
print(overstock_files.isnull().sum())
print("\nINF value check:")
print(np.isinf(overstock_files.select_dtypes(include=np.number)).sum())

print(overstock_files.shape)

