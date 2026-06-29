# =====================================================================
# IMPORTS FOR PART 3
# =====================================================================
import os
import logging
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =====================================================================
# DATA INGESTION & PREPARATION FOR ADVANCED MODELING
# =====================================================================
CLEANED_DATA_PATH = "cleaned_data.csv"
RAW_DATA_PATH = "california_raw_data.csv"

if os.path.exists(CLEANED_DATA_PATH):
    df = pd.read_csv(CLEANED_DATA_PATH)
else:
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Cleaned dataset missing at '{CLEANED_DATA_PATH}' and raw dataset missing at '{RAW_DATA_PATH}'."
        )
    df = pd.read_csv(RAW_DATA_PATH)

    if 'MedHouseVal' in df.columns and 'MedHouseValue' not in df.columns:
        df.rename(columns={'MedHouseVal': 'MedHouseValue'}, inplace=True)

    if 'Inferred_Numeric_Object' in df.columns:
        df['Inferred_Numeric_Object'] = pd.to_numeric(df['Inferred_Numeric_Object'], errors='coerce')
        df['Inferred_Numeric_Object'].fillna(df['Inferred_Numeric_Object'].median(), inplace=True)

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    df.to_csv(CLEANED_DATA_PATH, index=False)
    logger.info(f"Regenerated cleaned dataset as '{CLEANED_DATA_PATH}'")

if 'MedHouseVal' in df.columns and 'MedHouseValue' not in df.columns:
    df.rename(columns={'MedHouseVal': 'MedHouseValue'}, inplace=True)

if 'MedHouseValue' not in df.columns:
    raise ValueError("Required target column 'MedHouseValue' is missing from the dataset.")

# Prepare classification target and feature matrix
y_clf = (df['MedHouseValue'] > df['MedHouseValue'].median()).astype(int)
X_raw = df.drop(columns=['MedHouseValue'], errors='ignore')

# Encode categorical features
order_mapping = {'Basic_Tier': 0, 'Standard_Tier': 1, 'Premium_Tier': 2}
if 'Account_Segment_Str' in X_raw.columns:
    X_raw['Account_Segment_Str'] = X_raw['Account_Segment_Str'].map(order_mapping)
    X_raw['Account_Segment_Str'].fillna(1, inplace=True)

X_encoded = pd.get_dummies(X_raw, drop_first=True)
for col in X_encoded.columns:
    if X_encoded[col].dtype == bool:
        X_encoded[col] = X_encoded[col].astype(int)

X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X_encoded, y_clf, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_c1 = LogisticRegression(C=1.0, max_iter=1500, class_weight='balanced', random_state=42)
model_c1.fit(X_train_scaled, y_clf_train)

# Ensure logger is inherited from previous steps
logger.info("Initializing Part 3: Advanced Modeling, Ensembles & Tuning Pipeline...")

# =====================================================================
# TASK 1 & 2: DECISION TREE COMPARISONS (UNCONSTRAINED VS CONTROLLED)
# =====================================================================
logger.info("Task 1 & 2: Training Decision Trees...")

# Task 1: Unconstrained Tree
dt_unconstrained = DecisionTreeClassifier(random_state=42)
dt_unconstrained.fit(X_train_scaled, y_clf_train)

acc_train_un = dt_unconstrained.score(X_train_scaled, y_clf_train)
acc_test_un = dt_unconstrained.score(X_test_scaled, y_clf_test)

print("\n--- TASK 1: UNCONSTRAINED DECISION TREE CORES ---")
print(f"Training Accuracy: {acc_train_un:.4f}")
print(f"Test Accuracy    : {acc_test_un:.4f}")

# Task 2: Controlled Tree
dt_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_controlled.fit(X_train_scaled, y_clf_train)

acc_train_ctrl = dt_controlled.score(X_train_scaled, y_clf_train)
acc_test_ctrl = dt_controlled.score(X_test_scaled, y_clf_test)

print("\n--- TASK 2: CONTROLLED DECISION TREE CORES ---")
print(f"Training Accuracy: {acc_train_ctrl:.4f}")
print(f"Test Accuracy    : {acc_test_ctrl:.4f}")


# =====================================================================
# TASK 3: CRITERION COMPARISON (GINI VS ENTROPY)
# =====================================================================
logger.info("Task 3: Running Split Criterion Performance Splitting...")

dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42).fit(X_train_scaled, y_clf_train)
dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42).fit(X_train_scaled, y_clf_train)

print("\n--- TASK 3: CRITERION EVALUATION ---")
print(f"Gini Criterion Test Accuracy   : {dt_gini.score(X_test_scaled, y_clf_test):.4f}")
print(f"Entropy Criterion Test Accuracy: {dt_entropy.score(X_test_scaled, y_clf_test):.4f}")


# =====================================================================
# TASK 4: RANDOM FOREST & FEATURE IMPORTANCE
# =====================================================================
logger.info("Task 4: Compiling Random Forest Grid Architecture...")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_clf_train)

rf_train_acc = rf_model.score(X_train_scaled, y_clf_train)
rf_test_acc = rf_model.score(X_test_scaled, y_clf_test)
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_auc = roc_auc_score(y_clf_test, rf_probs)

print("\n--- TASK 4: RANDOM FOREST METRICS ---")
print(f"Training Accuracy: {rf_train_acc:.4f}")
print(f"Test Accuracy    : {rf_test_acc:.4f}")
print(f"Test ROC-AUC     : {rf_auc:.4f}")

# Extract and Sort Feature Importances
feat_importances = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 5 Features by Importance (Random Forest):")
print(feat_importances.head(5).to_string(index=False))


# =====================================================================
# TASK 4a & 4b: GRADIENT BOOSTING & FEATURE ABLATION STUDY
# =====================================================================
logger.info("Task 4a & 4b: Executing Boosting and Feature Ablation...")

# 4a. Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_scaled, y_clf_train)

gb_train_acc = gb_model.score(X_train_scaled, y_clf_train)
gb_test_acc = gb_model.score(X_test_scaled, y_clf_test)
gb_probs = gb_model.predict_proba(X_test_scaled)[:, 1]
gb_auc = roc_auc_score(y_clf_test, gb_probs)

print("\n--- TASK 4a: GRADIENT BOOSTING METRICS ---")
print(f"Training Accuracy: {gb_train_acc:.4f}")
print(f"Test Accuracy    : {gb_test_acc:.4f}")
print(f"Test ROC-AUC     : {gb_auc:.4f}")

# 4b. Feature Ablation Study
# Identify 5 lowest importance features
lowest_5_features = feat_importances.tail(5)['Feature'].tolist()
print("\n--- TASK 4b: FEATURE ABLATION ---")
print(f"Removing 5 lowest-importance features: {lowest_5_features}")

# Create safe boolean masks or column drop listings
ablation_cols = [c for c in X_encoded.columns if c not in lowest_5_features]
ablation_indices = [X_encoded.columns.get_loc(c) for c in ablation_cols]

# Subset the pre-scaled arrays cleanly across feature dimensions
X_train_scaled_reduced = X_train_scaled[:, ablation_indices]
X_test_scaled_reduced = X_test_scaled[:, ablation_indices]

# Retrain RF with reduced features
rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_scaled_reduced, y_clf_train)
rf_reduced_probs = rf_reduced.predict_proba(X_test_scaled_reduced)[:, 1]
rf_reduced_auc = roc_auc_score(y_clf_test, rf_reduced_probs)

print(f"Full Model Test ROC-AUC (All Features)  : {rf_auc:.4f}")
print(f"Reduced Model Test ROC-AUC (Dropped 5)  : {rf_reduced_auc:.4f}")


# =====================================================================
# TASK 5: CROSS-VALIDATED COMPARISON SUITE
# =====================================================================
logger.info("Task 5: Evaluating models across uniform Stratified K-Fold CV blocks...")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_to_cv = {
    "Logistic Regression (Part 2)": model_c1,
    "Controlled Decision Tree": dt_controlled,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}

cv_summary = {}

print("\n--- TASK 5: 5-FOLD STRATIFIED CV SCORES (ROC-AUC) ---")
for name, model in models_to_cv.items():
    # cv_score fits inside scaled metrics safely
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
    cv_summary[name] = (scores.mean(), scores.std())
    print(f"{name:30} -> Mean AUC: {scores.mean():.4f} | Std Dev: {scores.std():.4f}")


# =====================================================================
# TASK 6: HYPERPARAMETER TUNING VIA GRIDSEARCHCV PIPELINE
# =====================================================================
logger.info("Task 6: Deploying full scikit-learn optimization grid search workflow...")

# Build standalone target search pipeline matching rules (Input handles unscaled X_train)
tuning_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}

grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1
)

# Crucial: Fit on raw/unscaled training arrays to ensure zero leak rules in cross-validation blocks
grid_search.fit(X_train, y_clf_train)

best_pipeline = grid_search.best_estimator_

print("\n--- TASK 6: GRIDSEARCHCV PROFILE ---")
print(f"Best Hyperparameter Node Configs Found: {grid_search.best_params_}")
print(f"Best Validation Cross-Validation AUC   : {grid_search.best_score_:.4f}")


# =====================================================================
# TASK 7: MANUAL LEARNING CURVE CALCULATION
# =====================================================================
logger.info("Task 7: Extracting training size complexity vectors...")

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_curve_records = []

print("\n--- TASK 7: MANUAL LEARNING CURVE DATA ARRAY ---")
for f in fractions:
    subset_size = int(f * len(X_train))
    X_train_sub = X_train.iloc[:subset_size]
    y_clf_train_sub = y_clf_train.iloc[:subset_size]
    
    # Fit custom pipeline across slice
    best_pipeline.fit(X_train_sub, y_clf_train_sub)
    
    # Evaluate probabilities
    train_probs = best_pipeline.predict_proba(X_train_sub)[:, 1]
    test_probs = best_pipeline.predict_proba(X_test)[:, 1] # Pipeline handles test raw transformations safely
    
    train_auc = roc_auc_score(y_clf_train_sub, train_probs)
    test_auc = roc_auc_score(y_clf_test, test_probs)
    
    learning_curve_records.append({
        'Training Fraction': f"{f * 100:.0f}%",
        'Training AUC': f"{train_auc:.4f}",
        'Test AUC': f"{test_auc:.4f}"
    })

print(pd.DataFrame(learning_curve_records).to_string(index=False))


# =====================================================================
# TASK 8: BEST MODEL SERIALIZATION & VALIDATION RELOAD
# =====================================================================
logger.info("Task 8: Archiving and validating model configuration outputs...")

# Serialize the optimized operational pipeline object
model_filename = 'best_model.pkl'
joblib.dump(best_pipeline, model_filename)
logger.info(f"Optimized model pipeline safely preserved to file: '{model_filename}'")

print("\n--- TASK 8: RELOAD & PREDICT CHECK REPLICATION ---")
# Reload testing array validation 
reloaded_model = joblib.load(model_filename)

# Mock two test rows using actual structures from the raw input footprint
mock_test_rows = X_test.iloc[:2].copy()

# Call prediction engine
mock_predictions = reloaded_model.predict(mock_test_rows)
mock_probabilities = reloaded_model.predict_proba(mock_test_rows)[:, 1]

print(f"Successfully processed runtime validation array. Target Outputs: {mock_predictions}")
print(f"Assigned class output risk probabilities               : {mock_probabilities}")

# Compute holdout test performance for the summary matrix
final_test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
