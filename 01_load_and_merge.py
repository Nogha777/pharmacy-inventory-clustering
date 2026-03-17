import pandas as pd
import os

# ============================================================
# STEP 1: Load Raw Data Files
# ============================================================

# The path is unified so it will be stored in one variable
FILES_PATH = "/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/currently_working"

# First Group: inventory files
otc_data = pd.read_csv(os.path.join(FILES_PATH, "data OTC.csv"))
acute_data = pd.read_csv(os.path.join(FILES_PATH, "data Acute Treatment.csv"))
chronic_data = pd.read_csv(os.path.join(FILES_PATH, "data Chronic treatment.csv"))

# Add product type label to each group
otc_data["Product Type"] = "OTC"
acute_data["Product Type"] = "ACT"
chronic_data["Product Type"] = "CHR"

# ============================================================
# STEP 2: Load Sales & Returns Files
# ============================================================

c_otc = pd.read_csv(os.path.join(FILES_PATH, "c_data OTC.csv"))
c_acute = pd.read_csv(os.path.join(FILES_PATH, "c_data Acute Treatment.csv"))
c_chronic = pd.read_csv(os.path.join(FILES_PATH, "c_data Chronic treatment.csv"))

# Fix inconsistent column name in c_otc before renaming
c_otc = c_otc.rename(columns={'Quantaty 365d': 'Price'})

# ============================================================
# STEP 3: Fix Column Names (spelling + consistency)
# ============================================================

"""
--------df.columns vs df.columns.tolist()--------------
df.columns
Returns an Index object — pandas' own special format:
Index(['Item Number', 'Product Name', 'Price'], dtype='object')

df.columns.tolist()
Converts it to a plain Python list:
['Item Number', 'Product Name', 'Price']

When to use which:
comparing with another list  → tolist() safer
passing into a function      → tolist()
using zip()                  → tolist() cleaner
"""

n_columns = {
    'Quantaty 30 d': 'Quantity 30d',
    'price': 'Price 30d',
    'returnrd price in 30d': 'Returned Price In 30d',
    'Quantaty 180d': 'Quantity 180d',
    'returnrd price in 180d': 'Returned Price In 180d',
    'Price': 'Price 180d'
}

for df in [c_otc, c_acute, c_chronic]:
    df.rename(columns=n_columns, inplace=True)

# ============================================================
# STEP 4: Select Relevant Columns Before Merging
# ============================================================

# Note: selecting columns changes the original dataframes
selected_columns = [
    'Item number',
    'Quantity 30d',
    'Price 30d',
    'Returned Price In 30d',
    'Quantity 180d',
    'Returned Price In 180d',
    'Price 180d'
]

c_otc = c_otc[selected_columns]
c_acute = c_acute[selected_columns]
c_chronic = c_chronic[selected_columns]

# ============================================================
# STEP 5: Merge Inventory with Sales & Returns
# ============================================================

merged_otc = otc_data.merge(c_otc, on=['Item number'], how='outer')
merged_acute = acute_data.merge(c_acute, on=['Item number'], how='outer')
merged_chronic = chronic_data.merge(c_chronic, on=['Item number'], how='outer')

# ============================================================
# STEP 6: Save Merged Files
# ============================================================

merged_otc.to_csv(os.path.join(FILES_PATH, "merged_otc.csv"), index=False)
merged_acute.to_csv(os.path.join(FILES_PATH, "merged_acute.csv"), index=False)
merged_chronic.to_csv(os.path.join(FILES_PATH, "merged_chronic.csv"), index=False)

# Quick summary check
for name, df in [("merged_otc", merged_otc), ("merged_acute", merged_acute), ("merged_chronic", merged_chronic)]:
    print(f"\n--- {name} ---")
    print(df.describe())
