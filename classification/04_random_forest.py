import importlib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib

# ============================================================
# STEP 1: Import results from 02_split_scale_balance file
# ============================================================
# Load your file "02_split_scale_balance.py" as a module
split = importlib.import_module("02_split_scale_balance")
# Grab variables that were created in that file
X_train_balanced = split.X_train_balanced
y_train_balanced = split.y_train_balanced
X_val_scaled     = split.X_val_scaled
y_val            = split.y_val
X_test_scaled    = split.X_test_scaled
y_test           = split.y_test

# ============================================================
# STEP 2: Train Random Forest
# ============================================================

rf_model = RandomForestClassifier(
n_estimators=100,         # number of trees
    max_depth=5,              # prevents overfitting
    min_samples_leaf=5,       # minimum samples per leaf
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train_balanced, y_train_balanced)
print("Random Forest trained!")

# ============================================================
# STEP 3: Check Overfitting
# ============================================================

y_train_pred = rf_model.predict(X_train_balanced)
train_acc = accuracy_score(y_train_balanced, y_train_pred)

y_val_pred = rf_model.predict(X_val_scaled)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"Training accuracy: {train_acc:.3f}")
print(f"Validation accuracy: {val_acc:.3f}")
print(f"Difference: {train_acc - val_acc}")

# ============================================================
# STEP 4: Cross Validation
# ============================================================

cv_scores = cross_val_score(
    rf_model, X_train_balanced, y_train_balanced, cv=5, scoring='f1_macro'
)
print(f"\nCV scores: {cv_scores}")
print(f"Mean:      {cv_scores.mean():.4f}")
print(f"Std:       {cv_scores.std():.4f}")

# ============================================================
# STEP 5: Validation Results
# ============================================================
print("\n---------Validation Report---------")
print(classification_report(y_val, y_val_pred,  target_names=['Overstock', 'Healthy']))
print("---------Validation Confusion Matrics---------")
print(confusion_matrix(y_val, y_val_pred))

# ============================================================
# STEP 6: Test Results
# ============================================================
y_test_pred = rf_model.predict(X_test_scaled)

print("\n---------Test Report---------")
print(classification_report(y_test, y_test_pred, target_names=['Overstock', 'Healthy', 'Bulk']))
print("---------Test Confusion Matrics---------")
print(confusion_matrix(y_test, y_test_pred))

# ============================================================
# STEP 7: Feature Importance
# ============================================================

feature_cols = [
    'Sum of Total available', 'Sum of MAX', 'Total Sold Q 180d',
    'Price per Item', 'Return Rate 180d',
    'Type_ACT', 'Type_CHR', 'Type_OTC'
]

importance = pd.DataFrame({
    'Feature'   : feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--------Feature Importance------------")
print(importance)

# ============================================================
# STEP 7: Save Model
# ============================================================
FILES_PATH = "/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/Overstock Classification"

joblib.dump(rf_model, os.path.join(FILES_PATH,'RandomForestClassifier.pkl'))
