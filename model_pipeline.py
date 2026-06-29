import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

# Professional logging template setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =====================================================================
# STEP 1: LOAD CLEANED DATASET & DEFINE TARGET LABELS
# =====================================================================
CLEANED_DATA_PATH = "cleaned_data.csv"
RAW_DATA_PATH = "california_raw_data.csv"

if not os.path.exists(CLEANED_DATA_PATH):
    if os.path.exists(RAW_DATA_PATH):
        logger.info("Cleaned dataset not found, regenerating from raw data source...")
        df = pd.read_csv(RAW_DATA_PATH)

        # Normalize column naming across pipeline expectations
        if 'MedHouseVal' in df.columns and 'MedHouseValue' not in df.columns:
            df.rename(columns={'MedHouseVal': 'MedHouseValue'}, inplace=True)

        # Convert the dirty string object values to numeric and impute errors
        if 'Inferred_Numeric_Object' in df.columns:
            df['Inferred_Numeric_Object'] = pd.to_numeric(df['Inferred_Numeric_Object'], errors='coerce')
            df['Inferred_Numeric_Object'].fillna(df['Inferred_Numeric_Object'].median(), inplace=True)

        # Impute numeric columns with medians for any sparse nulls
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        df.to_csv(CLEANED_DATA_PATH, index=False)
        logger.info(f"Regenerated and saved cleaned dataset to '{CLEANED_DATA_PATH}'")
    else:
        raise FileNotFoundError(
            f"Cleaned dataset missing at '{CLEANED_DATA_PATH}', and raw dataset missing at '{RAW_DATA_PATH}'. Please execute main.py first!"
        )
else:
    logger.info("Ingesting clean output dataset into pipeline...")
    df = pd.read_csv(CLEANED_DATA_PATH)

# Normalize final label naming if needed
if 'MedHouseVal' in df.columns and 'MedHouseValue' not in df.columns:
    df.rename(columns={'MedHouseVal': 'MedHouseValue'}, inplace=True)

# Define label arrays as per criteria instructions
y_reg = df['MedHouseValue']
# Binarize continuous response exactly at its global median point
y_clf = (y_reg > y_reg.median()).astype(int)

# Extract core features grid matrix (Dropping targets and un-encoded variables)
X_raw = df.drop(columns=['MedHouseValue'], errors='ignore')

# =====================================================================
# STEP 2: CATEGORICAL ENCODING (Ordinal & False-Ordinal Mitigation)
# =====================================================================
logger.info("Applying custom feature engineering encoding structures...")

# A. Ordinal Label Encoding for 'Account_Segment_Str' (Basic < Standard < Premium)
order_mapping = {'Basic_Tier': 0, 'Standard_Tier': 1, 'Premium_Tier': 2}
# If col doesn't exist, we fall back safely using sample indices
if 'Account_Segment_Str' in X_raw.columns:
    X_raw['Account_Segment_Str'] = X_raw['Account_Segment_Str'].map(order_mapping)
    # Fill defaults if any parsing cells slip out
    X_raw['Account_Segment_Str'].fillna(1, inplace=True)

# B. One-Hot Encoding for non-ordinal strings using drop_first=True
X_encoded = pd.get_dummies(X_raw, drop_first=True)

# Convert boolean flag matrices into clear int data patterns for standard scalar conversions
for col in X_encoded.columns:
    if X_encoded[col].dtype == bool:
        X_encoded[col] = X_encoded[col].astype(int)

print(f"\nFinal feature engineering shape after column mutations: {X_encoded.shape}")

# =====================================================================
# STEP 3: LEAK-FREE SPLIT AND SCALING SCHEMES
# =====================================================================
logger.info("Executing leak-free feature matrix array scaling blocks...")

# Splitting arrays evenly before running any transformations to keep test matrix safe
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_encoded, y_reg, y_clf, test_size=0.2, random_state=42
)

# Build scaler transform vector engine
scaler = StandardScaler()
# Strict Rule: Fit scaler properties ONLY across training arrays to secure data boundaries
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# STEP 4: REGRESSION PIPELINE MODULE (OLS VS RIDGE COEFFICIENTS)
# =====================================================================
logger.info("Executing regression metrics suite...")

# Standard Ordinary Least Squares Execution
ols_model = LinearRegression().fit(X_train_scaled, y_reg_train)
y_pred_ols = ols_model.predict(X_test_scaled)

mse_ols = mean_squared_error(y_reg_test, y_pred_ols)
r2_ols = r2_score(y_reg_test, y_pred_ols)

print("\n--- OLS LINEAR MODEL PARAMETERS & INTERPRETATIONS ---")
coef_report = pd.DataFrame({
    'Feature Matrix Header Name': X_encoded.columns,
    'OLS Learned Weight Parameter': ols_model.coef_
})
coef_report['Absolute Magnitude'] = coef_report['OLS Learned Weight Parameter'].abs()
# Sort variables by size weight to locate top 3 features
coef_report = coef_report.sort_values(by='Absolute Magnitude', ascending=False)
print(coef_report[['Feature Matrix Header Name', 'OLS Learned Weight Parameter']])

# Ridge Regularization Model Implementation
ridge_model = Ridge(alpha=1.0).fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)

print("\n--- LINEAR REGRESSION VS RIDGE COMPARISON REPORT ---")
regression_summary_table = pd.DataFrame({
    "Framework Model Profile": ["Ordinary Least Squares (OLS)", "Ridge Regularization (Alpha=1.0)"],
    "Mean Squared Error (MSE)": [mse_ols, mse_ridge],
    "R-Squared Coefficient (R²)": [r2_ols, r2_ridge]
})
print(regression_summary_table.to_string(index=False))

# =====================================================================
# STEP 5: CLASSIFICATION PIPELINE MODULE & BALANCING CHECKS
# =====================================================================
logger.info("Executing binary classification engines...")

# Verify training distributions before compiling models
print("\nClass target count matrix distributions sitting inside raw training array:")
print(y_clf_train.value_counts())

# Constructing Logistic Regression using class_weight='balanced' to handle any skew risks
model_c1 = LogisticRegression(C=1.0, max_iter=1500, class_weight='balanced', random_state=42)
model_c1.fit(X_train_scaled, y_clf_train)

probs_c1 = model_c1.predict_proba(X_test_scaled)[:, 1]
preds_c1 = model_c1.predict(X_test_scaled)

print("\n--- CONFUSION MATRIX DATA ARRAY (C=1.0) ---")
print(confusion_matrix(y_clf_test, preds_c1))

print("\n--- CLASSIFICATION METRICS PROFILE REPORT ---")
print(classification_report(y_clf_test, preds_c1))

# Generate ROC validation plot structures
fpr_c1, tpr_c1, _ = roc_curve(y_clf_test, probs_c1)
auc_value_c1 = roc_auc_score(y_clf_test, probs_c1)

plt.figure(figsize=(7, 6))
plt.plot(fpr_c1, tpr_c1, color='darkorange', lw=2, label=f'Model C=1.0 (AUC = {auc_value_c1:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Layout Matrix')
plt.legend(loc="lower right")
plt.text(0.4, 0.2, f"Annotated Area Size: {auc_value_c1:.4f}", bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig("classification_roc_curve.png", dpi=110)
plt.close()
logger.info("ROC Verification layout completely drawn and saved to active folder.")

# =====================================================================
# STEP 6: DECISION THRESHOLD SENSITIVITY GRID EVALUATION
# =====================================================================
logger.info("Compiling dynamic threshold metrics array grid...")
target_threshold_splits = [0.30, 0.40, 0.50, 0.60, 0.70]
threshold_matrix_records = []

for threshold in target_threshold_splits:
    custom_predictions = (probs_c1 >= threshold).astype(int)
    p = precision_score(y_clf_test, custom_predictions, zero_division=0)
    r = recall_score(y_clf_test, custom_predictions, zero_division=0)
    f1 = f1_score(y_clf_test, custom_predictions, zero_division=0)
    
    threshold_matrix_records.append({
        'Threshold': f"{threshold:.2f}",
        'Precision': f"{p:.4f}",
        'Recall': f"{r:.4f}",
        'F1-Score': f"{f1:.4f}"
    })

print("\n--- DECISION-THRESHOLD TUNING TRACKING TABLE ---")
print(pd.DataFrame(threshold_matrix_records).to_string(index=False))

# =====================================================================
# STEP 7: REGULARIZATION STRUCTURAL COMPLEXITY EXP
# =====================================================================
logger.info("Triggering high-strength regularized alternate networks...")
model_c01 = LogisticRegression(C=0.01, max_iter=1500, class_weight='balanced', random_state=42)
model_c01.fit(X_train_scaled, y_clf_train)

probs_c01 = model_c01.predict_proba(X_test_scaled)[:, 1]
preds_c01 = model_c01.predict(X_test_scaled)

p_c01 = precision_score(y_clf_test, preds_c01, zero_division=0)
r_c01 = recall_score(y_clf_test, preds_c01, zero_division=0)
auc_c01 = roc_auc_score(y_clf_test, probs_c01)

p_c1_base = precision_score(y_clf_test, preds_c1, zero_division=0)
r_c1_base = recall_score(y_clf_test, preds_c1, zero_division=0)

print("\n--- L2 PENALTY STRENGTH PARAMETER EXPERIMENT RESULTS ---")
experiment_summary_table = pd.DataFrame({
    "Hyperparameter Space Setting": ["Baseline Model (C=1.0, Regularization=Standard)", "Regularized Model (C=0.01, Penalty=High)"],
    "Precision Score": [p_c1_base, p_c01],
    "Recall Score": [r_c1_base, r_c01],
    "Area Under Curve (AUC)": [auc_value_c1, auc_c01]
})
print(experiment_summary_table.to_string(index=False))

# =====================================================================
# STEP 8: STRUCTURAL GUIDED BOOTSTRAP CONFIDENCE INTERVAL (n=500)
# =====================================================================
logger.info("Initializing statistical bootstrap evaluation loop engine...")
n_rounds = 500
auc_difference_vectors = []
np.random.seed(42)

# Convert true labels to clean index-mapped numpy structures for fast resampling
y_clf_test_array = y_clf_test.values

for loop_round in range(n_rounds):
    # Sample row indices with replacement exactly matching guided prompt conditions
    bootstrap_sampled_indices = np.random.choice(len(y_clf_test_array), size=len(y_clf_test_array), replace=True)
    bootstrap_X = X_test_scaled[bootstrap_sampled_indices]
    bootstrap_y = y_clf_test_array[bootstrap_sampled_indices]

    boot_preds = model_c1.predict(bootstrap_X)
    boot_probs = model_c1.predict_proba(bootstrap_X)[:, 1]
    auc_difference_vectors.append(roc_auc_score(bootstrap_y, boot_probs))

# Report bootstrap confidence interval for the AUC metric
lower_bound = np.percentile(auc_difference_vectors, 2.5)
upper_bound = np.percentile(auc_difference_vectors, 97.5)
print("\n--- BOOTSTRAPPED AUC CONFIDENCE INTERVAL ---")
print(f"AUC 95% confidence interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
