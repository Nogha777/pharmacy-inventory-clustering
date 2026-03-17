import pandas as pd
import os

# Show all columns and rows in console output
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ============================================================
# STEP 1: Load Data
# ============================================================

files_path = "/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/currently_working"
overstock_files = pd.read_csv(os.path.join(files_path, "overstock_files.csv"))

# ============================================================
# STEP 2: Feature Engineering
# ============================================================

# --- Feature 1: Sell Through Rate ---
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
overstock_files.loc[masked_non0, "Sum of Total available"] = overstock_files.loc[masked_non0, "Total Sold Q 180d"]

overstock_files["Sell Through Rate"] = (
        overstock_files["Total Sold Q 180d"] / overstock_files["Sum of Total available"]
)
# Result is between 0 and 1 → e.g. 0.8 means 80% of stock was sold
# NaN = 0 / 0 → inactive/ghost item — handled later

# --- Feature 2: Overstock Ratio ---
# How much stock is left compared to what was sold
# Formula: Total Available / Total Sold
overstock_files["Overstock Ratio"] = (
        overstock_files["Sum of Total available"] / overstock_files["Total Sold Q 180d"]
)
# Result > 1 means more stock than sold → e.g. 3.0 means 3x more stock than sold
# inf → 0 sold quantity (extreme overstock)
# NaN → inactive item (0/0)

# --- Feature 3: Days of Supply ---
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
# STEP 4: Save
# ============================================================

overstock_files.to_csv(os.path.join(files_path, "overstock_files.csv"), index=False)
print("Feature engineering complete and overstock_files.csv saved")
