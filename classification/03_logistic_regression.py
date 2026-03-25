import os.path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import importlib # helps importing results from other files
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import joblib # to save model

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
# STEP 2:Train Logistic Regression
# ============================================================
"""
random_state=42 Reproducibility(the number stays the same after run)
max_iter=1000  # default is 100
- How many times the model tries to **converge** (find the best weights)
- Default 100 is sometimes not enough → you get a warning:
ConvergenceWarning: Logistic Regression failed to converge

class_weight='balanced'
class_weight='balanced'
Handle class imbalance
"""
"""
# C parameter controls regularization C controls how strict the model is:
# Small C = more regularization = less overfitting
High C (e.g. C=10)  → model is flexible
                    → fits training data very closely
                    → risk of memorizing = overfitting 

Low C (e.g. C=0.1)  → model is strict/penalized
                    → can't fit training too closely
                    → forces learning general patterns 
"""
lr_model = LogisticRegression(C=0.1, random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_balanced, y_train_balanced)
print("Logistic Regression trained!")

# ============================================================
# STEP 3:Evaluate on Validation Set
# ============================================================
"""
lr_model:  Your trained Logistic Regression model 
.predict(): Uses learned weights to classify each row 
X_val_scaled: Validation features (scaled, unseen during training) 
y_val_pred: Predicted labels output (0 or 1) 
"""
y_val_pred = lr_model.predict(X_val_scaled)

#-------check results of validation

print("--------Validation results------------")
print(classification_report(y_val, y_val_pred,
      target_names=['Overstock', 'Healthy']))
print("\n-------Confusion matrix------------")
print(confusion_matrix(y_val, y_val_pred))

# ============================================================
# STEP 4:Evaluate on Test Set
# ============================================================
y_test_pred = lr_model.predict(X_test_scaled)

#-------check results of validation
print("\n--------Test results------------")
print(classification_report(y_test, y_test_pred,
      target_names=['Overstock', 'Healthy', 'Bulk']))
print("\n-------Confusion matrix------------")
print(confusion_matrix(y_test, y_test_pred))

# ============================================================
# STEP 5: Compare training vs validation accuracy
# ============================================================
# Training accuracy
y_train_pred = lr_model.predict(X_train_balanced)
train_acc = accuracy_score(y_train_balanced, y_train_pred)

# Validation accuracy
y_val_pred = lr_model.predict(X_val_scaled)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"Training accuracy:   {train_acc:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Difference:          {train_acc - val_acc:.4f}")


"""
## How to Interpret
Train=1.00, Val=1.00 → difference=0.00 → NO overfitting 
Train=1.00, Val=0.70 → difference=0.30 → OVERFITTING 
Train=0.75, Val=0.73 → difference=0.02 → healthy 

in my case Difference = 0.0028 → almost zero → no overfitting 
"""
# ============================================================
# STEP 6: Cross Validation to Confirm
# ============================================================
cv_scores = cross_val_score(
    lr_model,
    X_train_balanced,
    y_train_balanced,
    cv=5,
    scoring='f1_macro'
)

print(f"CV scores: {cv_scores}")
print(f"Mean:      {cv_scores.mean():.4f}")
print(f"Std:       {cv_scores.std():.4f}")

#Model performs consistently well across ALL splits — not just lucky on one!
"""
ow std → model is STABLE
        → not sensitive to which data it sees
        → genuinely learned patterns ✅

High std → model is UNSTABLE
         → results vary wildly
         → overfitting or data issues ❌
"""
# ============================================================
# STEP 7: Save Model
# ============================================================
FILES_PATH = "/Users/norahnasser/Desktop/AI/ElevateHer/Over stock prediction/Overstock Classification"

joblib.dump(lr_model, os.path.join(FILES_PATH, 'LogisticRegression.pkl'))

"""
Pickle = converts any Python object into bytes
         and saves it to a file

Like taking a snapshot of your trained model
and freezing it for later use

Without pickle:
→ close PyCharm
→ model is gone from memory ❌
→ must retrain from scratch every time!

With pickle:
→ save model once after training
→ load it anytime in seconds ✅
→ no retraining needed
"""