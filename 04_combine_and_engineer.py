import pandas as pd
import os


# Show all columns and rows in console output
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ============================================================
# STEP 1: Load Cleaned Files
# ============================================================

files_path = "/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/currently_working"

m2_otc = pd.read_csv(os.path.join(files_path, "merged_otc.csv"))
m2_acute = pd.read_csv(os.path.join(files_path, "merged_acute.csv"))
m2_chronic = pd.read_csv(os.path.join(files_path, "merged_chronic.csv"))

# ============================================================
# STEP 2: Combine All Three Files
# ============================================================

# Concatenation causes repeated indices (0,1,0,1) instead of
# continuing (0,1,2,3) — reset_index fixes this
overstock_files = pd.concat([m2_otc, m2_acute, m2_chronic]).reset_index(drop=True)

# Remove unwanted index column created during previous save
overstock_files = overstock_files.drop(columns=["Unnamed: 0"])

# ============================================================
# STEP 3: Fix Price 180d Before Feature Engineering
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
# STEP 4: Feature Engineering — Add Return & Sales Columns
# ============================================================

"""
Before labelling I need total quantity sold in both 30d and 180d windows.
To get that I need:
  - Returned Q 30d and Returned Q 180d  (returns quantity)
  - Total Sold Q 30d and Total Sold Q 180d (net sales after returns)
"""

# Price per item in each time window
price_item30d = overstock_files["Price 30d"] / overstock_files["Quantity 30d"]
price_item180d = overstock_files["Price 180d"] / overstock_files["Quantity 180d"]

# Calculate returned quantities from returned price amounts
overstock_files["Returned Q 30d"] = (overstock_files["Returned Price In 30d"] / price_item30d).fillna(0)
overstock_files["Returned Q 180d"] = (overstock_files["Returned Price In 180d"] / price_item180d).fillna(0)

# Net sold quantity = total sold minus returned
overstock_files["Total Sold Q 30d"] = overstock_files["Quantity 30d"] - overstock_files["Returned Q 30d"]
overstock_files["Total Sold Q 180d"] = overstock_files["Quantity 180d"] - overstock_files["Returned Q 180d"]

# ============================================================
# STEP 5: Reorder Columns
# ============================================================

# Reordering all at once is cleaner than popping one column at a time
new_order = [
    'Item number', 'Product name', 'Product Type',
    'Sum of Total available', 'Sum of MAX',
    'Quantity 30d', 'Price 30d', 'Returned Q 30d', 'Returned Price In 30d', 'Total Sold Q 30d',
    'Quantity 180d', 'Price 180d', 'Returned Q 180d', 'Returned Price In 180d', 'Total Sold Q 180d'
]

overstock_files = overstock_files[new_order]

# ============================================================
# STEP 6: Round Calculated Columns
# ============================================================

round_columns = [
    'Price 30d', 'Price 180d',
    'Returned Q 30d', 'Returned Q 180d',
    'Total Sold Q 30d', 'Total Sold Q 180d'
]
overstock_files[round_columns] = overstock_files[round_columns].round(0)

# ============================================================
# STEP 7: Save
# ============================================================

overstock_files.to_csv(os.path.join(files_path, "overstock_files.csv"), index=False)

print("File saved")
print("\nNull value check:")
print(overstock_files.isnull().sum())
