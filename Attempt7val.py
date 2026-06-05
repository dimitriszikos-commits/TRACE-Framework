import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from prefixspan import PrefixSpan
import pm4py
import warnings
import sys
import time

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. CONFIGURATION 
# ==========================================
FILE_NAME = "/Users/dimitrioszikos/Dropbox/SeedExperiments/LDS2024_CCSR.csv"
MAPPING_FILE = "/Users/dimitrioszikos/Dropbox/SeedExperiments/PRCCSR_v2026-1.csv"
MAX_FEATURES = 500  # Number of top sequences/variants to pass to Lasso

print("="*85)
print("ABLATION STUDY ENGINE: PREFIXSPAN & PM4PY BASELINES (RAW DATA)")
print("="*85)

# ==========================================
# 2. USER PROMPT: SELECT ENGINE
# ==========================================
print("\nSelect the Baseline Extraction Engine for the Ablation Study:")
print("[1] PrefixSpan (Sequential Pattern Mining - Subsequences)")
print("[2] pm4py (Process Mining - Trace Variants)")
engine_choice = input("Enter 1 or 2: ").strip()
if engine_choice not in ['1', '2']:
    print("Invalid choice. Exiting.")
    sys.exit()

# ==========================================
# 3. LOAD DATA & MAPPINGS
# ==========================================
print(f"\nLoading Mapping Dictionary: {MAPPING_FILE}...")
try:
    map_df = pd.read_csv(MAPPING_FILE, dtype=str)
    map_df.columns = [col.replace("'", "").strip() for col in map_df.columns]
    icd_col = 'ICD-10-PCS CODE' if 'ICD-10-PCS CODE' in map_df.columns else map_df.columns[0]
    icd_to_ccsr = dict(zip(map_df[icd_col].str.replace("'", "").str.replace(".", "", regex=False).str.strip(), map_df['PRCCSR'].str.replace("'", "").str.strip()))
except FileNotFoundError:
    print(f"[!] Error: Could not find {MAPPING_FILE}.")
    sys.exit()

print(f"Loading Raw Cohort Data: {FILE_NAME}...")
try:
    df = pd.read_csv(FILE_NAME, dtype=str)
except FileNotFoundError:
    print(f"[!] Error: Could not find {FILE_NAME}.")
    sys.exit()

sample_size = input("\nEnter a sample size for processing (e.g., 50000) or press Enter for full data: ").strip()
if sample_size.isdigit() and int(sample_size) < len(df):
    df = df.sample(n=int(sample_size), random_state=42)
    print(f"--> Sampled down to {len(df)} total patients.")

target_input = input("\nEnter the Target Outcome (e.g., DIED): ").strip()
if target_input not in df.columns:
    print(f"[!] Critical Error: Column '{target_input}' not found.")
    sys.exit()
TARGET_COL = target_input
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0).astype(int)

# Use the exact same split logic and seed (42) as Attempt 7 to ensure identical test sets
df_train, df_temp = train_test_split(df, test_size=0.40, random_state=42, stratify=df[TARGET_COL])
df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42, stratify=df_temp[TARGET_COL])
print(f"--> Strict Splitting (60/20/20): {len(df_train)} Train | {len(df_val)} Validation | {len(df_test)} Test.")

ID_COL = 'PATIENT_ID' if 'PATIENT_ID' in df.columns else 'ROW_INDEX'
if ID_COL == 'ROW_INDEX':
    for d in [df_train, df_val, df_test]: d['ROW_INDEX'] = d.index.astype(str)

# ==========================================
# 4. CHRONOLOGICAL TIMELINE EXTRACTION (RAW ONLY)
# ==========================================
print("\nExtracting Raw Chronological Timelines (NO Recurrences, NO Absences)...")

def extract_raw_timelines(dataframe):
    patient_seqs = {}
    event_list_for_pm4py = []
    
    for idx, row in dataframe.iterrows():
        pid = row[ID_COL]
        raw_events = []
        for i in range(1, 26):
            icd_val, dt_val = row.get(f'ICD_PRCDR_CD{i}'), row.get(f'PRCDR_DT{i}')
            if pd.notna(icd_val) and pd.notna(dt_val) and str(icd_val).lower() != 'nan' and str(dt_val).lower() != 'nan':
                clean_icd = str(icd_val).replace('.', '').replace("'", "").strip()
                clean_dt = str(dt_val).split('.')[0].strip()
                try:
                    event_date = datetime.strptime(clean_dt, '%Y%m%d')
                except ValueError: continue
                ccsr_code = icd_to_ccsr.get(clean_icd)
                if ccsr_code:
                    raw_events.append({'date': event_date, 'ccsr': f"CCSR_{ccsr_code}"})
        if raw_events:
            raw_events.sort(key=lambda x: x['date'])
            seq = [e['ccsr'] for e in raw_events]
            patient_seqs[pid] = seq
            for e in raw_events:
                event_list_for_pm4py.append({'Patient_ID': pid, 'CCSR': e['ccsr'], 'Date': e['date']})
        else:
            patient_seqs[pid] = []
    return patient_seqs, event_list_for_pm4py

train_seqs, train_events = extract_raw_timelines(df_train)
val_seqs, _ = extract_raw_timelines(df_val)
test_seqs, _ = extract_raw_timelines(df_test)

# ==========================================
# 5. BASELINE MINING ENGINES
# ==========================================
feature_matrix_cols = []
extracted_features = []

# Global Support Floor Configuration
floor_input = input("\nEnter Minimum Prevalence Threshold % (e.g., 1.0, or press Enter for 1%): ").strip()
min_pct = float(floor_input) / 100.0 if floor_input else 0.01

if engine_choice == '1':
    print(f"\n[PrefixSpan] Mining Frequent Sequential Patterns from Train Set (Floor: {min_pct*100}%)...")
    # PrefixSpan requires a list of sequences (list of lists)
    db_train = [seq for seq in train_seqs.values() if seq]
    
    ps = PrefixSpan(db_train)
    min_support = int(len(db_train) * min_pct)
    all_patterns = ps.frequent(min_support)
    
    # Filter for length >= 2 and sort by frequency
    valid_patterns = [p for p in all_patterns if len(p[1]) >= 2]
    valid_patterns.sort(key=lambda x: x[0], reverse=True)
    top_patterns = valid_patterns[:MAX_FEATURES]
    
    extracted_features = [tuple(p[1]) for p in top_patterns]
    print(f"--> Extracted top {len(extracted_features)} sequential patterns.")

    # Helper to check if subsequence exists
    def is_subseq(sub, full_seq):
        it = iter(full_seq)
        return all(c in it for c in sub)

    build_fn = lambda seq, f: 1 if is_subseq(f, seq) else 0

elif engine_choice == '2':
    print(f"\n[pm4py] Extracting Trace Variants from Train Set (Floor: {min_pct*100}%)...")
    event_log = pd.DataFrame(train_events)
    log = pm4py.format_dataframe(event_log, case_id='Patient_ID', activity_key='CCSR', timestamp_key='Date')
    
    # pm4py trace variants natively extracted
    variants_dict = pm4py.get_variants(log)
    
    # Sort variants by count (most frequent trace paths)
    sorted_variants = sorted(variants_dict.items(), key=lambda x: x[1], reverse=True)
    top_variants = sorted_variants[:MAX_FEATURES]
    
    # The variant keys are strings separated by commas in pm4py
    extracted_features = [tuple(v[0]) for v in top_variants]
    print(f"--> Extracted top {len(extracted_features)} process trace variants.")
    
    # Helper to check if full trace exactly matches
    build_fn = lambda seq, f: 1 if tuple(seq) == f else 0

# ==========================================
# 6. BUILD PATIENT-FEATURE MATRIX
# ==========================================
print("\nBuilding binary feature matrices for Lasso Regression...")

def build_matrix(df_split, seq_dict):
    matrix_dict = {}
    for f in extracted_features:
        col_name = "BASE_" + "_TO_".join(f)
        matrix_dict[col_name] = [build_fn(seq_dict.get(pid, []), f) for pid in df_split[ID_COL]]
    return pd.DataFrame(matrix_dict, index=df_split.index)

X_train_base = build_matrix(df_train, train_seqs)
X_val_base = build_matrix(df_val, val_seqs)
X_test_base = build_matrix(df_test, test_seqs)

y_train, y_val, y_test = df_train[TARGET_COL].values, df_val[TARGET_COL].values, df_test[TARGET_COL].values

# Get basic active nodes and STRICTLY apply the global support floor to prevent baseline cheating
base_cols = [c for c in df_train.columns if c.startswith("CCSR_")]
valid_base_cols = [c for c in base_cols if (df_train[c].astype(np.int8).sum() / len(df_train)) >= min_pct]

X_train_nodes = df_train[valid_base_cols].fillna(0).astype(np.int8)
X_test_nodes = df_test[valid_base_cols].fillna(0).astype(np.int8)

# ==========================================
# 7. LASSO CLASSIFICATION & ABLATION COMPARISON
# ==========================================
print("\n" + "="*85)
print("PHASE 4: DOWNSTREAM ABLATION CLASSIFICATION (L1 Regularized LR)")
print("="*85)

print("\n1. Standard Baseline (Nodes Only - No Sequences)")
lasso_nodes = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, class_weight='balanced', random_state=42).fit(X_train_nodes, y_train)
pr_nodes = average_precision_score(y_test, lasso_nodes.predict_proba(X_test_nodes)[:, 1])
print(f"  > PR-AUC: {pr_nodes:.4f}")

print(f"\n2. Baseline Sequence Engine (Engine: {'PrefixSpan' if engine_choice == '1' else 'pm4py'})")
# Combine standard nodes + the newly extracted raw baseline sequences
X_train_hybrid = pd.concat([X_train_nodes, X_train_base], axis=1)
X_test_hybrid = pd.concat([X_test_nodes, X_test_base], axis=1)

from sklearn.model_selection import GridSearchCV

# Allow the model to find the optimal penalty for the specific feature sparsity
param_grid = {'C': [0.001, 0.01, 0.05, 0.1, 0.5]}
base_lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500, tol=0.01, class_weight='balanced', random_state=42)

print("  > Tuning Lasso Penalty (C) via 3-Fold CV...")
lasso_cv = GridSearchCV(base_lasso, param_grid, cv=3, scoring='average_precision', n_jobs=-1)
lasso_cv.fit(X_train_hybrid, y_train)

lasso_hybrid = lasso_cv.best_estimator_
best_c = lasso_cv.best_params_['C']
print(f"  > Optimal C Selected: {best_c}")

# Evaluate surviving features
coef_df = pd.DataFrame({'Feature': X_train_hybrid.columns, 'Coefficient': lasso_hybrid.coef_[0]})
surviving_seqs = coef_df[(coef_df['Feature'].str.startswith('BASE_')) & (coef_df['Coefficient'] != 0)]

# Evaluate surviving features
coef_df = pd.DataFrame({'Feature': X_train_hybrid.columns, 'Coefficient': lasso_hybrid.coef_[0]})
surviving_seqs = coef_df[(coef_df['Feature'].str.startswith('BASE_')) & (coef_df['Coefficient'] != 0)]

final_test_roc = roc_auc_score(y_test, lasso_hybrid.predict_proba(X_test_hybrid)[:, 1])
final_test_pr = average_precision_score(y_test, lasso_hybrid.predict_proba(X_test_hybrid)[:, 1])
test_lift = final_test_pr - pr_nodes

print(f"  > ROC-AUC: {final_test_roc:.4f}")
print(f"  > PR-AUC:  {final_test_pr:.4f} (Lift vs Baseline: {test_lift:+.4f})")
print(f"  > Surviving Baseline Sequences: {len(surviving_seqs)} / {len(extracted_features)}")

print("\n=====================================================================")
print("ABLATION CHECK: Run Attempt7.py and compare the PR-AUC Lift to this output.")
print("If Attempt7 has a higher Lift and higher Surviving features, you have mathematically")
print("proven that Node Engineering (Absences/Recurrences) is superior to standard engines.")
print("=====================================================================")