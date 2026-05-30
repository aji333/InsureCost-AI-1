# =============================================================================
# train.py  --  InsureCost-AI POC Training Script
# =============================================================================
# Run this ONCE on your local machine before deploying to Vercel.
#
# What it does:
#   1. Reads the 2,000-record CSV training file
#   2. Selects the 7 most important input features
#   3. Encodes gender (Male=1, Female=0)
#   4. Applies log1p() to the cost target (normalises the right-skewed
#      distribution -- converts $785-$192K range to a tighter ~6.7-12.2 range)
#   5. Scales all 7 features using StandardScaler
#   6. Trains an MLP neural network (80/20 train/test split)
#      Architecture: Input(7) -> Dense(64,ReLU) -> Dense(32,ReLU)
#                    -> Dense(16,ReLU) -> Output(1,Linear)
#   7. Evaluates the model (MAE and R2) on the test set
#      Note: predictions are converted back via expm1() before evaluation
#   8. Saves model.pkl and scaler.pkl to the project root
#
# Usage:
#   python train.py
#
# Output:
#   model.pkl   -- trained MLP model (predicts log1p(cost))
#   scaler.pkl  -- fitted StandardScaler (must be kept with the model)
#
# At inference time (api/predict.py, local_app.py):
#   cost_dollars = np.expm1(model.predict(scaled_features))
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -- Configuration -------------------------------------------------------------
CSV_PATH    = "health_insurance_training_data.csv"
MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"
TARGET_COL  = "annual_cost_usd"
RANDOM_SEED = 42

# The 7 features selected for the POC (order must match api/predict.py)
FEATURE_COLS = [
    "age",
    "gender",
    "chronic_condition_count",
    "prior_12m_cost",
    "er_visit_count_12m",
    "inpatient_days_12m",
    "comorbidity_index",
]

# -- Step 1: Load data ---------------------------------------------------------
print("=" * 60)
print("  InsureCost-AI -- Training Script")
print("=" * 60)
print(f"\n[1/9] Loading data from: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"\n  ERROR: '{CSV_PATH}' not found.\n"
        "  Please place the training CSV in the same folder as train.py."
    )

df = pd.read_csv(CSV_PATH)
print(f"       Loaded {len(df):,} records with {len(df.columns)} columns.")

# -- Step 2: Select features + target ------------------------------------------
print(f"\n[2/9] Selecting 7 features + target column...")
required_cols = FEATURE_COLS + [TARGET_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"  ERROR: Missing columns in CSV: {missing}")

df = df[required_cols].copy()
print(f"       Columns kept: {FEATURE_COLS}")

# -- Step 3: Encode gender -----------------------------------------------------
print(f"\n[3/9] Encoding gender  (Male -> 1, Female -> 0)...")
df["gender"] = (
    df["gender"].str.strip().str.upper()
    .map({"M": 1, "F": 0, "MALE": 1, "FEMALE": 0})
)
if df["gender"].isnull().any():
    raise ValueError(
        "  ERROR: Unexpected gender values found. Expected 'M'/'F' or 'Male'/'Female'."
    )
print(
    f"       Done. Male count: {int(df['gender'].sum()):,}  |  "
    f"Female count: {int((df['gender'] == 0).sum()):,}"
)

# -- Step 4: Apply log1p transform to target -----------------------------------
# Design spec: use log1p() to normalise the right-skewed cost distribution.
# This compresses the $785-$192K range into a ~6.7-12.2 range, allowing
# the MLP to learn on a stable, near-Gaussian target.
# At inference: predicted_log_cost -> np.expm1() -> dollars.
print(f"\n[4/9] Applying log1p() transform to target '{TARGET_COL}'...")
y_raw = df[TARGET_COL].values
y_log = np.log1p(y_raw)
print(
    f"       Raw cost   -- min: ${y_raw.min():,.0f}  |  "
    f"max: ${y_raw.max():,.0f}  |  mean: ${y_raw.mean():,.0f}"
)
print(
    f"       log1p(cost)-- min: {y_log.min():.3f}  |  "
    f"max: {y_log.max():.3f}  |  mean: {y_log.mean():.3f}"
)
print(f"       Inverse at inference: np.expm1(prediction) -> dollars")

# -- Step 5: Build feature matrix ----------------------------------------------
print(f"\n[5/9] Building feature matrix...")
X = df[FEATURE_COLS].values
print(f"       Shape: {X.shape}  ({X.shape[0]} rows x {X.shape[1]} features)")

# -- Step 6: Train / test split ------------------------------------------------
print(f"\n[6/9] Splitting 80% train / 20% test  (seed={RANDOM_SEED})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=RANDOM_SEED
)
print(f"       Training set : {len(X_train):,} records")
print(f"       Test set     : {len(X_test):,} records")

# -- Step 7: Scale features ----------------------------------------------------
print(f"\n[7/9] Fitting StandardScaler on training data only...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"       Scaler fitted. Each feature now has mean~0 and std~1.")

# -- Step 8: Train the MLP -----------------------------------------------------
# Design spec architecture: Input(7) -> 64(ReLU) -> 32(ReLU) -> 16(ReLU) -> 1
# Optimizer: Adam, batch_size=32, max_iter=100, early stopping (patience=10)
# alpha=0.01 (L2 regularisation) -- increased from 0.001 to prevent overfitting
# on this 2K-record dataset when training on log1p-transformed costs.
print(f"\n[8/9] Training MLP neural network...")
print(f"       Architecture : 7 -> 64 -> 32 -> 16 -> 1  (ReLU activations)")
print(f"       Optimizer    : Adam  |  Batch size: 32  |  Max iterations: 300")
print(f"       Early stopping: ON (patience=30 epochs)")
print(f"       Regularisation: alpha=0.001 (L2)\n")

model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),     # Design spec: 3 hidden layers
    activation="relu",                    # Design spec: ReLU
    solver="adam",                        # Design spec: Adam
    alpha=0.001,                          # L2 regularisation (design spec)
    batch_size=32,                        # Design spec: mini-batch size
    max_iter=300,                         # Increased from design's 100; Adam needs
                                          #   ~120 epochs on log1p targets to converge
                                          #   (early stopping still guards against
                                          #   unnecessary extra epochs)
    learning_rate_init=0.001,
    early_stopping=True,                  # Design spec: early stopping ON
    validation_fraction=0.1,
    n_iter_no_change=30,                  # Increased patience (design spec says 10)
                                          #   log1p loss curves are flatter -- need
                                          #   more patience before declaring convergence
    random_state=RANDOM_SEED,
    verbose=False,
)

model.fit(X_train_scaled, y_train)
print(f"       Training complete. Iterations run: {model.n_iter_}")

# -- Step 9: Evaluate ----------------------------------------------------------
# Predictions are in log1p space -- apply expm1() to convert back to dollars.
#
# Metric notes:
#   R2 (log space) -- primary metric. The model was trained on log1p(cost), so
#     R2 in log space correctly measures the model's predictive power.
#     Target: R2 > 0.70 is a good fit for a 2K-record POC dataset.
#   MAE (dollar space) -- secondary metric. Computed by back-transforming with
#     expm1(). Note: the dollar-space R2 is deliberately NOT reported because
#     expm1() amplifies high-cost outlier errors, making dollar-space R2
#     a misleading metric for log-transformed regression models.
print(f"\n[9/9] Evaluating on held-out test set...")
y_pred_log = model.predict(X_test_scaled)
y_pred_raw = np.maximum(np.expm1(y_pred_log), 0.0)  # back to dollars, no negatives
y_test_raw = np.expm1(y_test)                         # back to dollars

r2_log = r2_score(y_test, y_pred_log)                # primary: log-space R2
mae    = mean_absolute_error(y_test_raw, y_pred_raw)  # dollar MAE
mape   = np.mean(np.abs((y_test_raw - y_pred_raw) / (y_test_raw + 1))) * 100

print()
print("  +--------------------------------------------------+")
print(f"  |  R2 (log space) : {r2_log:>8.4f}  (primary metric)  |")
print(f"  |  MAE (dollars)  : ${mae:>10,.0f} per member/year  |")
print(f"  |  MAPE           : {mape:>8.1f}%                      |")
print("  +--------------------------------------------------+")
print("  Note: R2 is measured in log1p space (model's training space).")
print("        MAE is measured in dollars after expm1() back-transform.")

if r2_log >= 0.70:
    print("\n  [OK] R2 >= 0.70 -- good fit for a 2,000-record POC dataset.")
elif r2_log >= 0.50:
    print("\n  [NOTE] R2 is moderate. Acceptable for POC demonstration.")
else:
    print("\n  [!!] R2 is low. Consider adding more training data or features.")

# -- Save artifacts ------------------------------------------------------------
print(f"\n  Saving model  ->  {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)
model_size = os.path.getsize(MODEL_PATH) / 1024
print(f"  Saved. File size: {model_size:.1f} KB")

print(f"\n  Saving scaler ->  {SCALER_PATH}")
joblib.dump(scaler, SCALER_PATH)
scaler_size = os.path.getsize(SCALER_PATH) / 1024
print(f"  Saved. File size: {scaler_size:.1f} KB")

print()
print("=" * 60)
print("  Training complete!")
print()
print("  Model details:")
print("    Input features   : 7")
print("    Architecture     : 64 -> 32 -> 16 (ReLU) -> 1")
print("    Target transform : log1p(cost) during training")
print("    Inference step   : np.expm1(model.predict(...)) -> dollars")
print()
print("  Next steps:")
print("  1. Commit model.pkl and scaler.pkl to GitHub")
print("  2. Vercel will auto-deploy on your next git push")
print("  3. Or run:  python local_app.py  to test locally first")
print("=" * 60)
print()
