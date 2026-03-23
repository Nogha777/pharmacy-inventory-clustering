import pandas as pd
import os


# ============================================================
# STEP 1: Load Merged Files
# ============================================================

# The path is unified so it will be stored in one variable
FILES_PATH ="/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/Overstock clustering"

m_otc = pd.read_csv(os.path.join(FILES_PATH, "merged_otc.csv"))
m_acute = pd.read_csv(os.path.join(FILES_PATH, "merged_acute.csv"))
m_chronic = pd.read_csv(os.path.join(FILES_PATH, "merged_chronic.csv"))

# ============================================================
# STEP 2: Reorder Columns
# ============================================================

# Move Product Type to index 2 so it stays organized
# — the rest of the columns are numeric
for df in [m_otc, m_acute, m_chronic]:
    type_col = df.pop("Product Type")
    df.insert(2, "Product Type", type_col)

# ============================================================
# STEP 3: Fix Data Types
# ============================================================

# Some price columns were loaded as strings due to comma formatting
# Fix Price 30d in acute
m_acute["Price 30d"] = m_acute["Price 30d"].str.replace(",", "").astype(float)

# Fix Price 30d and Price 180d in chronic
# Note: .astype() does not work directly on dataframe — apply() needed
col = ["Price 30d", "Price 180d"]
m_chronic[col] = m_chronic[col].apply(lambda x: x.str.replace(",", "")).astype(float)

# ============================================================
# STEP 4: Handle Missing Values
# ============================================================

# Replace all NaN with 0 — no data is truly missing,
# zeros mean no sales/returns for that period
m_otc = m_otc.fillna(0)
m_acute = m_acute.fillna(0)
m_chronic = m_chronic.fillna(0)

# Verify no nulls remain
for name, df in [("m_otc", m_otc), ("m_acute", m_acute), ("m_chronic", m_chronic)]:
    print(f"\n-----{name}-----")
    print(df.isna().sum())

# ============================================================
# STEP 5: Save Cleaned Files
# ============================================================

m_otc.to_csv(os.path.join(FILES_PATH, "merged_otc.csv"), index=False)
m_acute.to_csv(os.path.join(FILES_PATH, "merged_acute.csv"), index=False)
m_chronic.to_csv(os.path.join(FILES_PATH, "merged_chronic.csv"), index=False)
