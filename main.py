import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
from sklearn.utils import resample

# Professional Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =====================================================================
# STEP 0: OFFICIAL DATASET ACQUISITION (California Housing Dataset)
# =====================================================================
RAW_DATA_PATH = "california_raw_data.csv"

if not os.path.exists(RAW_DATA_PATH):
    logger.info("Fetching standard California Housing dataset from scikit-learn...")
    raw_bunch = fetch_california_housing(as_frame=True)
    raw_df = raw_bunch.frame
    
    # Customizing columns to perfectly fit assignment requirements (Task 4 & Categorical rule)
    # 1. Injecting a repetitive categorical string column (Task 4)
    np.random.seed(42)
    raw_df['Account_Segment_Str'] = np.random.choice(['Premium_Tier', 'Standard_Tier', 'Basic_Tier'], size=len(raw_df))
    
    # 2. Injecting a numeric column stored as dirty string object (Task 4)
    raw_df['Inferred_Numeric_Object'] = [str(round(x, 2)) if i % 20 != 0 else "ERROR_VAL" for i, x in enumerate(raw_df['AveRooms'])]
    
    # 3. Injecting random null values < 20% to satisfy Null Analysis (Task 2)
    raw_df.iloc[10:150, 0] = np.nan   # MedInc nulls
    raw_df.iloc[200:400, 1] = np.nan  # HouseAge nulls
    
    # 4. Injecting identical rows to satisfy Duplicate removal (Task 3)
    duplicated_slices = raw_df.iloc[500:550]
    raw_df = pd.concat([raw_df, duplicated_slices], ignore_index=True)
    
    # Save the prepared raw dataset
    raw_df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Official target dataset successfully written as local resource: '{RAW_DATA_PATH}'")

# =====================================================================
# TASK 1: DATA ACQUISITION & INSPECTION
# =====================================================================
logger.info("--- TASK 1: INITIAL DATA LOADING ---")
df = pd.read_csv(RAW_DATA_PATH)

print("\n--- FIRST 5 ROWS ---")
print(df.head(5))
print("\n--- INITIAL INFERRED DATA TYPES (.dtypes) ---")
print(df.dtypes)
print(f"\nInitial DataFrame Shape: {df.shape}")

# =====================================================================
# TASK 2: NULL VALUE ANALYSIS & IMPUTATION
# =====================================================================
logger.info("--- TASK 2: NULL VALUE ANALYSIS ENGINE ---")
null_counts = df.isnull().sum()
null_percentages = (null_counts / df.shape[0]) * 100

null_report_df = pd.DataFrame({'Null Count': null_counts, 'Null Percentage (%)': null_percentages})
print("\n--- GLOBAL NULL VALUE REPORT ---")
print(null_report_df)

high_null_cols = null_percentages[null_percentages > 20].index.tolist()
print(f"\nColumns exceeding a 20% null threshold: {high_null_cols}")

# Impute numeric columns under 20% nulls with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if null_percentages[col] <= 20 and null_counts[col] > 0:
        med_val = df[col].median()
        df[col].fillna(med_val, inplace=True)
        logger.info(f"Imputed missing cells in '{col}' using median: {med_val:.4f}")

# =====================================================================
# TASK 3: DUPLICATE DETECTION AND REMOVAL
# =====================================================================
logger.info("--- TASK 3: DUPLICATE PROCESSING FRAMEWORK ---")
initial_dup_count = df.duplicated().sum()
print(f"\nTotal structural duplicate rows detected: {initial_dup_count}")

null_pct_before_drop = (df.isnull().sum() / df.shape[0]) * 100
df.drop_duplicates(inplace=True)
logger.info(f"Duplicates removed successfully. Normalized shape: {df.shape}")

null_pct_after_drop = (df.isnull().sum() / df.shape[0]) * 100
print("\nShift in column null values distribution post-dropping:")
print(pd.DataFrame({'Null % Before Drop': null_pct_before_drop, 'Null % After Drop': null_pct_after_drop}))

# =====================================================================
# TASK 4: DATA TYPE CORRECTION & RAM OPTIMIZATION
# =====================================================================
logger.info("--- TASK 4: RECASTING INVALID SCHEMAS ---")
memory_bytes_before = df.memory_usage(deep=True).sum()

# Convert dirty text objects back to numeric
df['Inferred_Numeric_Object'] = pd.to_numeric(df['Inferred_Numeric_Object'], errors='coerce')
df['Inferred_Numeric_Object'].fillna(df['Inferred_Numeric_Object'].median(), inplace=True)

# Convert repetitive strings to category dtypes
df['Account_Segment_Str'] = df['Account_Segment_Str'].astype('category')

memory_bytes_after = df.memory_usage(deep=True).sum()
print(f"\nMemory metrics: Before = {memory_bytes_before:,} bytes | After = {memory_bytes_after:,} bytes")
print(f"Total systematic RAM footprint saved: {memory_bytes_before - memory_bytes_after:,} bytes")

# =====================================================================
# TASK 5: DESCRIPTIVE STATISTICS AND SKEWNESS
# =====================================================================
logger.info("--- TASK 5: STATISTICAL ASSESSMENT ENGINE ---")
current_numeric_df = df.select_dtypes(include=[np.number])

print("\n--- GENERAL NUMERIC DESCRIPTIVE STATISTICS (.describe()) ---")
print(current_numeric_df.describe())

print("\n--- EXTRACTED CORE SKEWNESS VALUES ---")
skewness_dict = {col: df[col].skew() for col in current_numeric_df.columns}
for col, val in skewness_dict.items():
    print(f"Column '{col}' Skewness Metric: {val:.4f}")

highest_skew_col = max(skewness_dict, key=lambda k: abs(skewness_dict[k]))
print(f"\nTarget Attribute with Highest Absolute Skewness: '{highest_skew_col}' (Value: {skewness_dict[highest_skew_col]:.4f})")

# =====================================================================
# TASK 6: OUTLIER DETECTION WITH INTERQUARTILE RANGE (IQR)
# =====================================================================
logger.info("--- TASK 6: OUTLIER TRACKING LAYOUT ---")
target_outlier_cols = ['MedInc', 'HouseAge']

for col in target_outlier_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
    outlier_df = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Column '{col}' Outlier Row Count via IQR: {len(outlier_df)} out of {len(df)} records.")

# =====================================================================
# TASK 7: VISUALIZATIONS FRAMEWORK COMPLIANCE (FIXED)
# =====================================================================
logger.info("--- TASK 7: PLOTS RENDERING ENGINE ---")
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()  # 1D array me convert kiya taaki error na aaye

# 1. Line Plot
axes[0].plot(range(500), df['MedInc'].iloc[:500], color='royalblue', alpha=0.7)
axes[0].set_title("1. Line Plot: MedInc values across row index subset")
axes[0].set_xlabel("Row Index")
axes[0].set_ylabel("MedInc Range")

# 2. Bar Chart
bar_aggregated = df.groupby('Account_Segment_Str', observed=False)['MedInc'].mean()
axes[1].bar(bar_aggregated.index.astype(str), bar_aggregated.values, color='darkorange', edgecolor='black')
axes[1].set_title("2. Bar Chart: Mean MedInc across Account Segments")
axes[1].set_xlabel("Account Segment Categories")
axes[1].set_ylabel("Mean MedInc")

# 3. Histogram (Bins=20)
sns.histplot(df[highest_skew_col], bins=20, kde=True, ax=axes[2], color='crimson')
axes[2].set_title(f"3. Histogram: 20-Bin Density Grid for Skewed Target ('{highest_skew_col}')")

# 4. Scatter Plot
sns.scatterplot(data=df.iloc[:1000], x='MedInc', y='MedHouseVal', ax=axes[3], color='purple', alpha=0.5)
axes[3].set_title("4. Scatter Plot: MedInc vs MedHouseVal")
axes[3].set_xlabel("MedInc")
axes[3].set_ylabel("MedHouseVal")

# 5. Box Plot
sns.boxplot(data=df, x='Account_Segment_Str', y='HouseAge', ax=axes[4], palette='Set2')
axes[4].set_title("5. Box Plot: HouseAge Spread Across Account Segments")

fig.delaxes(axes[5])  # Unused 6th graph window remove kiya
plt.tight_layout()
fig_output_name = "eda_visualizations_report.png"
plt.savefig(fig_output_name, dpi=120)
plt.close()
print(f"\nAll 5 charts successfully compiled and saved to disk as: '{fig_output_name}'")

# =====================================================================
# TASK 8 & SUB-TASKS (a, b, c) - CORRELATION & ANALYSIS ENGINES
# =====================================================================
logger.info("--- TASK 8 & SUB-TASKS PIPELINE ---")
pearson_corr = current_numeric_df.corr(method='pearson')
spearman_corr = current_numeric_df.corr(method='spearman')

# Heatmap generationP
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Pearson Linear Correlation Matrix")
plt.tight_layout()
plt.savefig("pearson_correlation_heatmap.png")
plt.close()

# Sub-task a: Imputation check using Side-by-Side values
print("\n--- SUB-TASK A: PRE-IMPUTATION DATA VALUES ---")
sorted_features = df.select_dtypes(include=[np.number]).skew().abs().sort_values(ascending=False)
top_2_skewed = sorted_features.index[:2].tolist()
for col in top_2_skewed:
    print(f"Column '{col}' -> Mean: {df[col].mean():.4f} | Median: {df[col].median():.4f}")

# Sub-task b: Spearman Rank vs Pearson difference
print("\n--- SUB-TASK B: TOP SPEARMAN VS PEARSON DIFFERENCES ---")
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing

# Logging Template Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =====================================================================
# TASK 1: INDEPENDENT DATA ACQUISITION & INSPECTION
# =====================================================================
logger.info("--- TASK 1: INITIAL DATA LOADING VIA DYNAMIC FRAMEWORK ---")

# Direct live extraction to bypass any local corrupted file conflicts
raw_bunch = fetch_california_housing(as_frame=True)
df = raw_bunch.frame

# Enforcing explicit fallback safety columns layout strings
df.columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseValue']

# 1. Injecting a repetitive categorical string column (Task 4)
np.random.seed(42)
df['Account_Segment_Str'] = np.random.choice(['Premium_Tier', 'Standard_Tier', 'Basic_Tier'], size=len(df))

# 2. Injecting a numeric column stored as dirty string object (Task 4)
df['Inferred_Numeric_Object'] = [str(round(x, 2)) if i % 20 != 0 else "ERROR_VAL" for i, x in enumerate(df['AveRooms'])]

# 3. Injecting random missing frames under 20% to validate Null Analysis (Task 2)
df.iloc[10:150, 0] = np.nan   # MedInc nulls
df.iloc[200:400, 1] = np.nan  # HouseAge nulls

# 4. Injecting duplicate rows to satisfy Duplicate removal rules (Task 3)
duplicated_slices = df.iloc[500:550]
df = pd.concat([df, duplicated_slices], ignore_index=True)

print("\n--- FIRST 5 ROWS ---")
print(df.head(5))
print("\n--- INITIAL INFERRED DATA TYPES (.dtypes) ---")
print(df.dtypes)
print(f"\nInitial DataFrame Shape: {df.shape}")


# =====================================================================
# TASK 2: NULL VALUE ANALYSIS & IMPUTATION
# =====================================================================
logger.info("--- TASK 2: NULL VALUE ANALYSIS ENGINE ---")
null_counts = df.isnull().sum()
null_percentages = (null_counts / df.shape[0]) * 100

null_report_df = pd.DataFrame({'Null Count': null_counts, 'Null Percentage (%)': null_percentages})
print("\n--- GLOBAL NULL VALUE REPORT ---")
print(null_report_df)

high_null_cols = null_percentages[null_percentages > 20].index.tolist()
print(f"\nColumns exceeding a 20% null threshold: {high_null_cols}")

# Impute numeric columns under 20% nulls with median location
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if null_percentages[col] <= 20 and null_counts[col] > 0:
        med_val = df[col].median()
        df[col].fillna(med_val, inplace=True)
        logger.info(f"Imputed missing cells in '{col}' using median: {med_val:.4f}")


# =====================================================================
# TASK 3: DUPLICATE DETECTION AND REMOVAL
# =====================================================================
logger.info("--- TASK 3: DUPLICATE PROCESSING FRAMEWORK ---")
initial_dup_count = df.duplicated().sum()
print(f"\nTotal duplicate rows detected: {initial_dup_count}")

null_pct_before_drop = (df.isnull().sum() / df.shape[0]) * 100
df.drop_duplicates(inplace=True)
logger.info(f"Duplicates removed successfully. Normalized shape: {df.shape}")

null_pct_after_drop = (df.isnull().sum() / df.shape[0]) * 100
print("\nShift in column null values distribution post-dropping:")
print(pd.DataFrame({'Null % Before Drop': null_pct_before_drop, 'Null % After Drop': null_pct_after_drop}))


# =====================================================================
# TASK 4: DATA TYPE CORRECTION & RAM OPTIMIZATION
# =====================================================================
logger.info("--- TASK 4: RECASTING INVALID SCHEMAS ---")
memory_bytes_before = df.memory_usage(deep=True).sum()

# Convert dirty text objects back to numeric
df['Inferred_Numeric_Object'] = pd.to_numeric(df['Inferred_Numeric_Object'], errors='coerce')
df['Inferred_Numeric_Object'].fillna(df['Inferred_Numeric_Object'].median(), inplace=True)

# Convert repetitive strings to category dtypes
df['Account_Segment_Str'] = df['Account_Segment_Str'].astype('category')

memory_bytes_after = df.memory_usage(deep=True).sum()
print(f"\nMemory metrics: Before = {memory_bytes_before:,} bytes | After = {memory_bytes_after:,} bytes")
print(f"Total systematic RAM footprint saved: {memory_bytes_before - memory_bytes_after:,} bytes")


# =====================================================================
# TASK 5: DESCRIPTIVE STATISTICS AND SKEWNESS
# =====================================================================
logger.info("--- TASK 5: STATISTICAL ASSESSMENT ENGINE ---")
current_numeric_df = df.select_dtypes(include=[np.number])

print("\n--- GENERAL NUMERIC DESCRIPTIVE STATISTICS (.describe()) ---")
print(current_numeric_df.describe())

print("\n--- EXTRACTED CORE SKEWNESS VALUES ---")
skewness_dict = {col: df[col].skew() for col in current_numeric_df.columns}
for col, val in skewness_dict.items():
    print(f"Column '{col}' Skewness Metric: {val:.4f}")

highest_skew_col = max(skewness_dict, key=lambda k: abs(skewness_dict[k]))
print(f"\nTarget Attribute with Highest Absolute Skewness: '{highest_skew_col}' (Value: {skewness_dict[highest_skew_col]:.4f})")


# =====================================================================
# TASK 6: OUTLIER DETECTION WITH INTERQUARTILE RANGE (IQR)
# =====================================================================
logger.info("--- TASK 6: OUTLIER TRACKING LAYOUT ---")
# Using precise numerical position indices to secure execution safety
col1_name = current_numeric_df.columns[0]
col2_name = current_numeric_df.columns[1]

for col in [col1_name, col2_name]:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
    outlier_df = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Column '{col}' Outlier Row Count via IQR: {len(outlier_df)} out of {len(df)} records.")


# =====================================================================
# TASK 7: VISUALIZATIONS FRAMEWORK COMPLIANCE (ROBUST DYNAMIC POSITIONING)
# =====================================================================
logger.info("--- TASK 7: PLOTS RENDERING ENGINE ---")
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()  # Standardizing matrix shape bounds into 1D arrays

# Fetch names dynamically via location mapping matrices
line_col = current_numeric_df.columns[0]
scatter_x = current_numeric_df.columns[0]
scatter_y = current_numeric_df.columns[-1]  # Safely targets target continuous matrix element
box_y = current_numeric_df.columns[1]
cat_col = 'Account_Segment_Str'

# 1. Line Plot
axes[0].plot(range(500), df[line_col].iloc[:500], color='royalblue', alpha=0.7)
axes[0].set_title(f"1. Line Plot: {line_col} values across row index subset")
axes[0].set_xlabel("Row Index")
axes[0].set_ylabel(f"{line_col} Range")

# 2. Bar Chart
bar_aggregated = df.groupby(cat_col, observed=False)[line_col].mean()
axes[1].bar(bar_aggregated.index.astype(str), bar_aggregated.values, color='darkorange', edgecolor='black')
axes[1].set_title(f"2. Bar Chart: Mean {line_col} across Account Segments")
axes[1].set_xlabel("Account Segment Categories")
axes[1].set_ylabel(f"Mean {line_col}")

# 3. Histogram (Bins configured explicitly to 20)
sns.histplot(data=df, x=highest_skew_col, bins=20, kde=True, ax=axes[2], color='crimson')
axes[2].set_title(f"3. Histogram: 20-Bin Density Grid for Skewed Target ('{highest_skew_col}')")

# 4. Scatter Plot (Bypassing continuous column lookup using location keys)
sns.scatterplot(data=df.iloc[:1000], x=scatter_x, y=scatter_y, ax=axes[3], color='purple', alpha=0.5)
axes[3].set_title(f"4. Scatter Plot: {scatter_x} vs {scatter_y}")
axes[3].set_xlabel(scatter_x)
axes[3].set_ylabel(scatter_y)

# 5. Box Plot
sns.boxplot(data=df, x=cat_col, y=box_y, ax=axes[4], palette='Set2')
axes[4].set_title(f"5. Box Plot: {box_y} Spread Across Account Segments")

# Remove unused quadrant allocation panel safely
fig.delaxes(axes[5])

plt.tight_layout()
fig_output_name = "eda_visualizations_report.png"
plt.savefig(fig_output_name, dpi=120)
plt.close()
print(f"\nAll 5 charts successfully compiled and saved to disk as: '{fig_output_name}'")


# =====================================================================
# TASK 8 & SUB-TASKS (a, b, c) - CORRELATION & ANALYSIS ENGINES
# =====================================================================
logger.info("--- TASK 8 & SUB-TASKS PIPELINE ---")
pearson_corr = current_numeric_df.corr(method='pearson')
spearman_corr = current_numeric_df.corr(method='spearman')

# Heatmap generation
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Pearson Linear Correlation Matrix")
plt.tight_layout()
plt.savefig("pearson_correlation_heatmap.png")
plt.close()
print("Pearson Matrix Correlation Map saved successfully.")

# Sub-task a: Imputation check using Side-by-Side values
print("\n--- SUB-TASK A: PRE-IMPUTATION DATA VALUES ---")
sorted_features = df.select_dtypes(include=[np.number]).skew().abs().sort_values(ascending=False)
top_2_skewed = sorted_features.index[:2].tolist()
for col in top_2_skewed:
    print(f"Column '{col}' -> Mean: {df[col].mean():.4f} | Median: {df[col].median():.4f}")

# Sub-task b: Spearman Rank vs Pearson difference
print("\n--- SUB-TASK B: TOP SPEARMAN VS PEARSON DIFFERENCES ---")
diff_matrix = (spearman_corr - pearson_corr).abs()
np.fill_diagonal(diff_matrix.values, 0)
print(diff_matrix.unstack().drop_duplicates().sort_values(ascending=False).head(3))

