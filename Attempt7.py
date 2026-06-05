import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import chi2_contingency
import warnings
import sys
import json
import re
import time

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. CONFIGURATION & INTERACTIVE SETUP
# ==========================================
FILE_NAME = "/Users/dimitrioszikos/Dropbox/SeedExperiments/LDS2024_CCSR.csv"
MAPPING_FILE = "/Users/dimitrioszikos/Dropbox/SeedExperiments/PRCCSR_v2026-1.csv"
OUTPUT_FILE = "/Users/dimitrioszikos/Dropbox/SeedExperiments/Divergence_Sequences.csv"

# Hyperparameters & Thresholds
JACCARD_THRESHOLD = 0.90     
MIN_CO_OCCURRENCE = 5         
ASYM_FLOOR = 0.10
MARKOV_FLOOR = 0.05
MAX_PRUNE_SEARCH = 500

# Rigorous Empirical Statistical Thresholds
P_VALUE_ALPHA = 0.05          # Empirical Alpha for Collider Traps
N_PERMUTATIONS = 1000         # Iterations for Empirical Null Distribution
FDR_ALPHA = 0.05              # Benjamini-Hochberg False Discovery Rate

print("="*85)
print("CLINICAL SEQUENCE EXTRACTOR: THE MASTER PIPELINE (EMPIRICAL CAUSAL EDITION)")
print("="*85)

print(f"Loading Mapping Dictionary: {MAPPING_FILE}...")
try:
    map_df = pd.read_csv(MAPPING_FILE, dtype=str)
    map_df.columns = [col.replace("'", "").strip() for col in map_df.columns]
    ccsr_desc_dict = dict(zip(map_df['PRCCSR'].str.replace("'", "").str.strip(), 
                              map_df['PRCCSR DESCRIPTION'].str.replace("'", "").str.strip()))
except FileNotFoundError:
    print(f"[!] Warning: Could not find {MAPPING_FILE}. Narratives will use raw codes.")
    ccsr_desc_dict = {}

print(f"\nLoading Cohort Data: {FILE_NAME}...")
try:
    df = pd.read_csv(FILE_NAME, dtype=str)
except FileNotFoundError:
    print(f"[!] Error: Could not find {FILE_NAME}.")
    sys.exit()

target_input = input("\n[1] Enter the Target Outcome (e.g., DIED, Readmission, or LOS_Flag): ").strip()
if target_input not in df.columns:
    print(f"[!] Critical Error: Column '{target_input}' not found.")
    sys.exit()
TARGET_COL = target_input
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0).astype(int)

drg_filter = input("[2] Enter a specific DRG code to filter by (or press Enter for all patients): ").strip()
if drg_filter:
    if 'DRG_CD' in df.columns: 
        df = df[df['DRG_CD'] == drg_filter]
        print(f"--> Filtered to {len(df)} patients.")
    else:
        print("[!] Warning: 'DRG_CD' column not found. Proceeding without filter.")

sample_size = input("[3] Enter a sample size for processing (e.g., 50000) or press Enter for full data: ").strip()
if sample_size.isdigit() and int(sample_size) < len(df):
    df = df.sample(n=int(sample_size), random_state=42)
    print(f"--> Sampled down to {len(df)} total patients.")

len_input = input("[4] Enter Maximum Sequence Length (or press Enter for purely organic growth): ").strip()
manual_max_len = int(len_input) if len_input.isdigit() else float('inf')

run_mode = input("[5] Run Dynamic Grid Search Optimization (Y/N)? ").strip().upper()
if run_mode == 'N':
    manual_prune = int(input("  -> Enter Sequence Pruning Limit (e.g., 200): "))
    manual_lasso = float(input("  -> Enter final Lasso Strictness (C) (e.g., 0.1): "))
else:
    print("--> Auto-Optimization selected. (Dynamic Early Stopping Enabled).")

run_jaccard = input("[6] Enable Jaccard Bundling for concurrent events (Y/N)? (Skip for disease progression): ").strip().upper()

print("\n[7] Choose Directionality Metric:")
print("    [A] Pure Difference: P(B|A) - P(A|B) (Standard, stable for general population trends)")
print("    [B] Normalized Asymmetry: (P(B|A) - P(A|B)) / (P(B|A) + P(A|B)) (*Recommended for highly unbalanced datasets/rare events*)")
metric_choice = input("    Enter A or B (Default A): ").strip().upper()
use_normalized = (metric_choice == 'B')

floor_input = input("\n[8] Enter Minimum Baseline Support % (e.g., 0.5 for 0.5% prevalence, or press Enter for 0.5%): ").strip()
MIN_TRAIN_PREVALENCE_PCT = float(floor_input) / 100.0 if floor_input else 0.005

neg_floor_input = input("\n[9] Enter Minimum Prevalence % for generating Negative 'Absence' Nodes (e.g., 5 for 5%, or press Enter for default 5%): ").strip()
MIN_PREVALENCE_FOR_NEGATIVE = float(neg_floor_input) / 100.0 if neg_floor_input else 0.05

# ==========================================
# 2. STRICT 3-WAY SPLIT (LEAKAGE FIX)
# ==========================================
print("\nPHASE 1: Strict Train / Validation / Test Splitting")
df_train, df_temp = train_test_split(df, test_size=0.40, random_state=42, stratify=df[TARGET_COL])
df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42, stratify=df_temp[TARGET_COL])
print(f"--> Data split (60/20/20): {len(df_train)} Train | {len(df_val)} Validation | {len(df_test)} Test.")

# ==========================================
# 3. PRE-PROCESSING & INFORMATIVE ABSENCES
# ==========================================
print("\nPHASE 2: Pre-Processing & Informative Absences")
ccsr_cols = [col for col in df_train.columns if col.startswith("CCSR_")]

for col in ccsr_cols:
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0).astype(np.int8)
    df_val[col] = pd.to_numeric(df_val[col], errors='coerce').fillna(0).astype(np.int8)
    df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(0).astype(np.int8)

X_train_raw = df_train[ccsr_cols].values.astype(bool)
col_to_idx = {col: idx for idx, col in enumerate(ccsr_cols)}
bundled_nodes = set()
new_bundles_train, new_bundles_val, new_bundles_test = {}, {}, {}

if run_jaccard == 'Y':
    print("--> Executing Jaccard Bundling...")
    potential_bundles = []
    for node_A, node_B in combinations(ccsr_cols, 2):
        idx_A, idx_B = col_to_idx[node_A], col_to_idx[node_B]
        intersect = np.bitwise_and(X_train_raw[:, idx_A], X_train_raw[:, idx_B]).sum()
        if intersect < MIN_CO_OCCURRENCE: continue
        union = np.bitwise_or(X_train_raw[:, idx_A], X_train_raw[:, idx_B]).sum()
        jaccard = intersect / union
        if jaccard >= JACCARD_THRESHOLD:
            potential_bundles.append((jaccard, node_A, node_B, idx_A, idx_B))
            
    potential_bundles.sort(key=lambda x: x[0], reverse=True)
    
    for jaccard, node_A, node_B, idx_A, idx_B in potential_bundles:
        if node_A in bundled_nodes or node_B in bundled_nodes: continue 
        bundle_name = f"BUNDLE_{node_A}_{node_B}"
        new_bundles_train[bundle_name] = np.bitwise_and(X_train_raw[:, idx_A], X_train_raw[:, idx_B]).astype(np.int8)
        new_bundles_val[bundle_name] = ((df_val[node_A] == 1) & (df_val[node_B] == 1)).astype(np.int8)
        new_bundles_test[bundle_name] = ((df_test[node_A] == 1) & (df_test[node_B] == 1)).astype(np.int8)
        bundled_nodes.update([node_A, node_B])

    if new_bundles_train:
        df_train = pd.concat([df_train, pd.DataFrame(new_bundles_train, index=df_train.index)], axis=1)
        df_val = pd.concat([df_val, pd.DataFrame(new_bundles_val, index=df_val.index)], axis=1)
        df_test = pd.concat([df_test, pd.DataFrame(new_bundles_test, index=df_test.index)], axis=1)

active_positive_nodes = [c for c in ccsr_cols if c not in bundled_nodes] + list(new_bundles_train.keys())
neg_cols_train, neg_cols_val, neg_cols_test = {}, {}, {}
total_train = len(df_train)

for col in active_positive_nodes:
    clean_col = str(col).strip()
    if clean_col.startswith("BUNDLE_"): continue 
    
    if "_R" not in clean_col:
        prevalence = df_train[col].sum() / total_train
        if prevalence >= MIN_PREVALENCE_FOR_NEGATIVE:
            neg_cols_train[f"NOT_{clean_col}"] = (df_train[col] == 0).astype(np.int8)
            neg_cols_val[f"NOT_{clean_col}"] = (df_val[col] == 0).astype(np.int8)
            neg_cols_test[f"NOT_{clean_col}"] = (df_test[col] == 0).astype(np.int8)
    else:
        base_col = col.split("_R")[0]
        if base_col in df_train.columns:
            neg_cols_train[f"NOT_{clean_col}"] = ((df_train[base_col] == 1) & (df_train[col] == 0)).astype(np.int8)
            neg_cols_val[f"NOT_{clean_col}"] = ((df_val[base_col] == 1) & (df_val[col] == 0)).astype(np.int8)
            neg_cols_test[f"NOT_{clean_col}"] = ((df_test[base_col] == 1) & (df_test[col] == 0)).astype(np.int8)

if neg_cols_train:
    df_train = pd.concat([df_train, pd.DataFrame(neg_cols_train, index=df_train.index)], axis=1)
    df_val = pd.concat([df_val, pd.DataFrame(neg_cols_val, index=df_val.index)], axis=1)
    df_test = pd.concat([df_test, pd.DataFrame(neg_cols_test, index=df_test.index)], axis=1)

all_nodes = active_positive_nodes + list(neg_cols_train.keys())
train_counts = df_train[all_nodes].sum()
dynamic_floor = max(3, int(total_train * MIN_TRAIN_PREVALENCE_PCT)) 
valid_nodes = train_counts[train_counts >= dynamic_floor].index.tolist()

if not valid_nodes:
    print(f"\n[!] CRITICAL ERROR: 0 features met the support floor of {dynamic_floor} patients.")
    sys.exit()

P_marginal_train = (df_train[valid_nodes].sum() / total_train).to_dict()
X_np_train = df_train[valid_nodes].values.astype(bool)
X_np_val = df_val[valid_nodes].values.astype(bool)
X_np_test = df_test[valid_nodes].values.astype(bool)
node_to_idx = {n: i for i, n in enumerate(valid_nodes)}
y_train, y_val, y_test = df_train[TARGET_COL].values, df_val[TARGET_COL].values, df_test[TARGET_COL].values

# ==========================================
# 4. SUPERVISED SEQUENCE MINING ENGINE
# ==========================================
print(f"\nPHASE 3: Sequence Mining & Permutation Ranking (Target: {TARGET_COL})")
baseline_lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, class_weight='balanced', random_state=42).fit(df_train[valid_nodes], y_train)
baseline_pr_val = average_precision_score(y_val, baseline_lasso.predict_proba(df_val[valid_nodes])[:, 1])
baseline_pr_test = average_precision_score(y_test, baseline_lasso.predict_proba(df_test[valid_nodes])[:, 1])
print(f"--> Baseline PR-AUC Val: {baseline_pr_val:.4f} | Test: {baseline_pr_test:.4f}")

def mine_top_sequences(max_len, max_prune):
    W_DIV, W_MARK, W_OUT = 1.0, 1.0, 2.0   
    all_discovered = []
    y_tr_bool = y_train.astype(bool)
    P_target_base = max(0.0001, y_tr_bool.sum() / total_train)
    
    def calc_lift(mask):
        count = mask.sum()
        if count == 0: return 1.0
        p_tg_seq = np.bitwise_and(mask, y_tr_bool).sum() / count
        max_lift = 1.0 / P_target_base
        if p_tg_seq == 0: return max_lift
        return min(max_lift, max(p_tg_seq / P_target_base, P_target_base / p_tg_seq))

    def get_base_root(node_str):
        return re.sub(r'_R\d*', '', node_str.replace("NOT_", ""))

    def is_logically_valid(curr_seq, next_node):
        if next_node in curr_seq: return False
        if next_node.startswith("NOT_") and curr_seq[-1].startswith("NOT_"): return False
        if get_base_root(curr_seq[-1]) == get_base_root(next_node):
            if not curr_seq[-1].startswith("NOT_") != (not next_node.startswith("NOT_")):
                return False 
        return True

    def is_collider_trap(curr_seq, next_node):
        if len(curr_seq) < 2: return False
        Z = next_node
        Y = curr_seq[-1] 
        idx_Z, idx_Y = node_to_idx[Z], node_to_idx[Y]
        mask_Z, mask_Y = X_np_train[:, idx_Z], X_np_train[:, idx_Y]
        
        for X in curr_seq[:-1]:
            idx_X = node_to_idx[X]
            mask_X = X_np_train[:, idx_X]
            
            O_11 = np.bitwise_and(mask_X, mask_Z).sum()
            O_10 = np.bitwise_and(mask_X, ~mask_Z).sum()
            O_01 = np.bitwise_and(~mask_X, mask_Z).sum()
            O_00 = total_train - (O_11 + O_10 + O_01)
            if O_11 == 0 or O_10 == 0 or O_01 == 0 or O_00 == 0: continue
            
            _, p_uncond, _, _ = chi2_contingency([[O_00, O_01], [O_10, O_11]], correction=False)
            if p_uncond < P_VALUE_ALPHA: continue 
                
            mask_X_Y, mask_Z_Y = np.bitwise_and(mask_X, mask_Y), np.bitwise_and(mask_Z, mask_Y)
            C_11 = np.bitwise_and(mask_X_Y, mask_Z_Y).sum()
            C_10 = np.bitwise_and(mask_X_Y, ~mask_Z_Y).sum()
            C_01 = np.bitwise_and(~mask_X_Y, mask_Z_Y).sum()
            C_00 = mask_Y.sum() - (C_11 + C_10 + C_01)
            
            if C_11 == 0 or C_10 == 0 or C_01 == 0 or C_00 == 0: continue
                
            _, p_cond, _, _ = chi2_contingency([[C_00, C_01], [C_10, C_11]], correction=False)
            if p_cond < P_VALUE_ALPHA: return True 
        return False

    growth_queue = []
    for node_A, node_B in combinations(valid_nodes, 2):
        if node_A.startswith("NOT_") and node_B.startswith("NOT_"): continue
        if get_base_root(node_A) == get_base_root(node_B):
            if not node_A.startswith("NOT_") != (not node_B.startswith("NOT_")): continue 
        
        idx_A, idx_B = node_to_idx[node_A], node_to_idx[node_B]
        mask = np.bitwise_and(X_np_train[:, idx_A], X_np_train[:, idx_B])
        if mask.sum() < dynamic_floor: continue
            
        growth_queue.extend([{'seq': (node_A, node_B), 'mask': mask}, {'seq': (node_B, node_A), 'mask': mask}])
        
        pa_b = (mask.sum() / total_train) / P_marginal_train[node_B]
        pb_a = (mask.sum() / total_train) / P_marginal_train[node_A]
        if pa_b == 0 or pb_a == 0: continue
            
        diff = (pb_a - pa_b) / (pb_a + pa_b) if use_normalized else pb_a - pa_b
        lift = calc_lift(mask)
        
        if diff >= ASYM_FLOOR:
            markov = (pb_a - P_marginal_train[node_B]) / P_marginal_train[node_B]
            if abs(markov) >= MARKOV_FLOOR:
                all_discovered.append({'seq': (node_A, node_B), 'mask': mask, 'diff': diff, 'markov': markov, 'power': (abs(diff)**W_DIV)*(abs(markov)**W_MARK)*(lift**W_OUT)})
        elif diff <= -ASYM_FLOOR:
            markov = (pa_b - P_marginal_train[node_A]) / P_marginal_train[node_A]
            if abs(markov) >= MARKOV_FLOOR:
                all_discovered.append({'seq': (node_B, node_A), 'mask': mask, 'diff': diff, 'markov': markov, 'power': (abs(diff)**W_DIV)*(abs(markov)**W_MARK)*(lift**W_OUT)})

    current_len = 2
    while growth_queue and current_len < max_len:
        next_q = []
        for seq_dict in growth_queue:
            curr_seq, curr_mask = seq_dict['seq'], seq_dict['mask']
            curr_p = curr_mask.sum() / total_train
            
            for next_node in valid_nodes:
                if not is_logically_valid(curr_seq, next_node) or is_collider_trap(curr_seq, next_node): continue
                
                mask = np.bitwise_and(curr_mask, X_np_train[:, node_to_idx[next_node]])
                if mask.sum() < dynamic_floor: continue
                next_q.append({'seq': curr_seq + (next_node,), 'mask': mask})
                
                pn_s = (mask.sum() / total_train) / curr_p
                ps_n = (mask.sum() / total_train) / P_marginal_train[next_node]
                if pn_s == 0 or ps_n == 0: continue
                    
                diff = (pn_s - ps_n) / (pn_s + ps_n) if use_normalized else pn_s - ps_n
                markov = (pn_s - P_marginal_train[next_node])/P_marginal_train[next_node]
                lift = calc_lift(mask)
                
                if diff >= ASYM_FLOOR and abs(markov) >= MARKOV_FLOOR:
                    all_discovered.append({'seq': curr_seq + (next_node,), 'mask': mask, 'diff': diff, 'markov': markov, 'power': (abs(diff)**W_DIV)*(abs(markov)**W_MARK)*(lift**W_OUT)})
        growth_queue, current_len = next_q, current_len + 1

    if not all_discovered: return []

    # ==========================================
    # -> THE EMPIRICAL PERMUTATION TEST (FDR) <-
    # ==========================================
    print(f"\n--> Structural Mining Complete. Found {len(all_discovered)} potential causal trajectories.")
    print(f"--> Executing Empirical Permutation Test (N={N_PERMUTATIONS}) for FDR Control...")
    start_t = time.time()
    
    pooled_null_powers = []
    y_tr_copy = y_train.copy()
    
    for _ in range(N_PERMUTATIONS):
        np.random.shuffle(y_tr_copy)
        y_bool_perm = y_tr_copy.astype(bool)
        p_base_perm = max(0.0001, y_bool_perm.sum() / total_train)
        max_l = 1.0 / p_base_perm
        
        for d in all_discovered:
            m = d['mask']
            c = m.sum()
            if c == 0: 
                l_perm = 1.0
            else:
                p_tg = np.bitwise_and(m, y_bool_perm).sum() / c
                if p_tg == 0: l_perm = max_l
                else: l_perm = min(max_l, max(p_tg / p_base_perm, p_base_perm / p_tg))
            
            null_p = (abs(d['diff'])**W_DIV) * (abs(d['markov'])**W_MARK) * (l_perm**W_OUT)
            pooled_null_powers.append(null_p)
            
    pooled_null_powers = np.array(pooled_null_powers)
    pooled_null_powers.sort()
    
    n_null = len(pooled_null_powers)
    for d in all_discovered:
        idx = np.searchsorted(pooled_null_powers, d['power'], side='left')
        count_greater = n_null - idx
        d['p_value'] = (count_greater + 1) / (n_null + 1)
        
    all_discovered.sort(key=lambda x: x['p_value'])
    
    max_pass_idx = -1
    total_tests = len(all_discovered)
    for i, d in enumerate(all_discovered):
        bh_critical_value = ((i + 1) / total_tests) * FDR_ALPHA
        if d['p_value'] <= bh_critical_value:
            max_pass_idx = i
            
    if max_pass_idx >= 0:
        fdr_passed = all_discovered[:max_pass_idx + 1]
    else:
        fdr_passed = []
        
    print(f"    * Test completed in {time.time() - start_t:.1f} seconds.")
    print(f"    * {len(fdr_passed)} sequences mathematically proved non-random via FDR Step-Up (\u03B1 = {FDR_ALPHA}).")
    
    # Sort survivors by actual power and prune
    fdr_passed.sort(key=lambda x: x['power'], reverse=True)
    seen, final = set(), []
    for d in fdr_passed:
        if d['seq'] not in seen:
            seen.add(d['seq'])
            final.append(d['seq'])
            if len(final) >= max_prune: break
            
    return final

top_seqs = mine_top_sequences(manual_max_len, MAX_PRUNE_SEARCH)
if not top_seqs:
    print("[!] Zero sequences survived the empirical permutation test. The pipeline cannot proceed without verifiable signals.")
    sys.exit()
    
print(f"--> Extracted and Globally Ranked top {len(top_seqs)} strictly validated sequences.")

seq_mat_tr = np.zeros((total_train, len(top_seqs)), dtype=np.int8)
seq_mat_val = np.zeros((len(df_val), len(top_seqs)), dtype=np.int8)
seq_mat_te = np.zeros((len(df_test), len(top_seqs)), dtype=np.int8)

for i, seq in enumerate(top_seqs):
    m_tr, m_val, m_te = np.ones(total_train, dtype=bool), np.ones(len(df_val), dtype=bool), np.ones(len(df_test), dtype=bool)
    for node in seq:
        m_tr &= X_np_train[:, node_to_idx[node]]
        m_val &= X_np_val[:, node_to_idx[node]]
        m_te &= X_np_test[:, node_to_idx[node]]
    seq_mat_tr[:, i] = m_tr
    seq_mat_val[:, i] = m_val
    seq_mat_te[:, i] = m_te

# ==========================================
# 4.5 PRE-OPTIMIZATION MULTI-DIMENSIONAL COMPRESSION
# ==========================================
print("\nPHASE 3.5: Multi-Dimensional Matrix Compression...")

seq_cols_train, seq_cols_val, seq_cols_test = {}, {}, {}
sequence_feature_names = []

for i, seq_tuple in enumerate(top_seqs):
    seq_name = "SEQ_" + "_TO_".join(seq_tuple)
    seq_cols_train[seq_name] = seq_mat_tr[:, i]
    seq_cols_val[seq_name] = seq_mat_val[:, i]
    seq_cols_test[seq_name] = seq_mat_te[:, i]
    sequence_feature_names.append(seq_name)

df_train = pd.concat([df_train, pd.DataFrame(seq_cols_train, index=df_train.index)], axis=1)
df_val = pd.concat([df_val, pd.DataFrame(seq_cols_val, index=df_val.index)], axis=1)
df_test = pd.concat([df_test, pd.DataFrame(seq_cols_test, index=df_test.index)], axis=1)

families = {}
for col in sequence_feature_names:
    if re.search(r'_R\d+', col):
        base_path = re.sub(r'_R\d+', '', col)
        numbers = [int(n) for n in re.findall(r'_R(\d+)', col)]
        
        if base_path not in families: 
            families[base_path] = []
            
        families[base_path].append((col, numbers))

cols_to_drop = []
new_dimension_cols = []

for base_path, variants in families.items():
    if len(variants) > 1:
        num_dimensions = len(variants[0][1]) 
        
        for dim_idx in range(num_dimensions):
            dim_col_name = f"{base_path}_[NODE_{dim_idx + 1}_DOSE]"
            new_dimension_cols.append(dim_col_name)
            
            for df_target in [df_train, df_val, df_test]:
                if dim_col_name not in df_target.columns:
                    df_target[dim_col_name] = 0
                    
            for col, vector in variants:
                dose_val = vector[dim_idx] if dim_idx < len(vector) else 0
                for df_target in [df_train, df_val, df_test]:
                    mask = (df_target[col] == 1) & (df_target[dim_col_name] < dose_val)
                    df_target.loc[mask, dim_col_name] = dose_val

        for col, _ in variants:
            if col not in cols_to_drop: cols_to_drop.append(col)

df_train.drop(columns=cols_to_drop, inplace=True)
df_val.drop(columns=cols_to_drop, inplace=True)
df_test.drop(columns=cols_to_drop, inplace=True)
sequence_feature_names = [f for f in sequence_feature_names if f not in cols_to_drop] + new_dimension_cols

print(f"--> Compression Complete! Converted {len(cols_to_drop)} redundant flags into {len(new_dimension_cols)} continuous features.")

# ==========================================
# 5. OPTIMIZATION & EVALUATION
# ==========================================
if run_mode == 'Y':
    print("\nExecuting Dynamic Hill-Climbing Optimization (Optimizing for PR-AUC against Validation Set)...")
    prune_opts, c_opts = [50, 100, 150, 200, 300, 400, 500], [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    best_pr, best_p, best_c = 0, 50, 0.1
    macro_patience, macro_fails = 2, 0
    
    for p_limit in prune_opts:
        if p_limit > len(sequence_feature_names): continue
        print(f"\n  > Exploring P = {p_limit} sequences...")
        
        current_seqs = sequence_feature_names[:p_limit]
        xt_h = pd.concat([df_train[valid_nodes], df_train[current_seqs]], axis=1)
        xv_h = pd.concat([df_val[valid_nodes], df_val[current_seqs]], axis=1)
        
        p_improved, micro_fails, last_pr = False, 0, 0
        for c_v in c_opts:
            lso = LogisticRegression(penalty='l1', solver='liblinear', C=c_v, class_weight='balanced', random_state=42).fit(xt_h, y_train)
            pr = average_precision_score(y_val, lso.predict_proba(xv_h)[:, 1])
            lift = pr - baseline_pr_val
            print(f"    - Lasso C: {c_v:.3f} -> Val PR-AUC: {pr:.4f} (Val Lift: {lift:+.4f})")
            
            if pr > best_pr: 
                best_pr, best_p, best_c = pr, p_limit, c_v
                p_improved = True
            if pr < last_pr:
                micro_fails += 1
                if micro_fails >= 2: break
            else: micro_fails = 0
            last_pr = pr
            
        if not p_improved:
            macro_fails += 1
            if macro_fails >= macro_patience: break
        else: macro_fails = 0
    f_p, f_c = best_p, best_c
    print(f"\n--> WINNING HYPERPARAMETERS: Top {f_p} Sequences | Lasso C: {f_c}")
else:
    f_p, f_c = min(MAX_PRUNE_SEARCH, len(sequence_feature_names)), 0.1
    print(f"--> Using Manual Hyperparameters: Top {f_p} Sequences | Lasso C: {f_c}")

# ==========================================
# 6. EXTRACT AND SAVE
# ==========================================
def get_clinical_desc(node_name):
    clean_node = re.sub(r'_\[NODE_\d+_DOSE\]', '', node_name)
    is_neg = clean_node.startswith("NOT_")
    clean_node = clean_node.replace("NOT_", "")
    
    if clean_node.startswith("BUNDLE_"):
        parts = clean_node.replace("BUNDLE_CCSR_", "").split("_CCSR_")
        d1 = ccsr_desc_dict.get(parts[0], parts[0])
        d2 = ccsr_desc_dict.get(parts[1], parts[1])
        return f"Concurrent [{d1} & {d2}]"
    elif clean_node.startswith("CCSR_"):
        base_code = clean_node.replace("CCSR_", "").split("_R")[0]
        base_desc = ccsr_desc_dict.get(base_code, base_code)
        desc = f"{base_desc} (Recurrence)" if "_R" in clean_node else base_desc
        return f"Absence of {desc}" if is_neg else desc
    return node_name

final_features = sequence_feature_names[:f_p]
best_edges = []

if final_features:
    output_data = []
    print("\nExtracting and saving rich sequence data...")
    for seq_name in final_features:
        dose_match = re.search(r'_\[NODE_(\d+)_DOSE\]', seq_name)
        dose_str = f" (Dose: Problem Node {dose_match.group(1)})" if dose_match else ""
        
        clean_seq_name = re.sub(r'_\[NODE_\d+_DOSE\]', '', seq_name)
        parts = clean_seq_name.replace("SEQ_", "").split("_TO_")
        best_edges.append(tuple(parts))
        
        clinical_chain = [get_clinical_desc(p) for p in parts]
        narrative = "  --->  ".join(clinical_chain) + dose_str
        
        output_data.append({
            "Raw_Feature_Name": seq_name,
            "Sequence_Length": len(parts),
            "Clinical_Narrative": narrative,
            "Train_Patients": (df_train[seq_name] > 0).sum(),
            "Val_Patients": (df_val[seq_name] > 0).sum(),
            "Test_Patients": (df_test[seq_name] > 0).sum()
        })
    pd.DataFrame(output_data).to_csv(OUTPUT_FILE, index=False)
    print(f"--> Saved presentation-ready clinical sequences to {OUTPUT_FILE}")

# ==========================================
# 7. FINAL EVALUATION MATRIX
# ==========================================
print("\n" + "="*85)
print("PHASE 4: 3x3 CLINICAL VALIDATION MATRIX (EVALUATED ON UNTOUCHED HOLD-OUT SET)")
print("="*85)

feature_sets = {
    "1. RAW BASELINE (Original Nodes Only)": valid_nodes,
    "2. ENGINEERED SEQS (Sequences Only)": final_features,
    "3. HYBRID (Baseline + Sequences)": valid_nodes + final_features
}

fallback_seq_winners = pd.DataFrame() 

for set_name, features in feature_sets.items():
    if not features: continue
        
    print(f"\n--- Matrix Cell: {set_name} ({len(features)} features) ---")
    
    X_tr_final = df_train[features]
    y_tr_final = y_train
    X_te_final = df_test[features]
    
    std_model = LogisticRegression(penalty='l2', solver='lbfgs', C=100.0, max_iter=1000, class_weight='balanced')
    std_model.fit(X_tr_final, y_tr_final)
    print(f"  > Standard LR ROC-AUC: {roc_auc_score(y_test, std_model.predict_proba(X_te_final)[:, 1]):.4f} | PR-AUC: {average_precision_score(y_test, std_model.predict_proba(X_te_final)[:, 1]):.4f}")
    
    nb_model = BernoulliNB()
    nb_model.fit(X_tr_final, y_tr_final)
    print(f"  > Naive Bayes ROC-AUC: {roc_auc_score(y_test, nb_model.predict_proba(X_te_final)[:, 1]):.4f} | PR-AUC: {average_precision_score(y_test, nb_model.predict_proba(X_te_final)[:, 1]):.4f}")
    
    eval_c = 1.0 if "Sequences Only" in set_name else f_c
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=eval_c, max_iter=200, tol=0.01, class_weight='balanced', random_state=42)
    lasso.fit(X_tr_final, y_tr_final)
    surviving = sum(lasso.coef_[0] != 0)
    
    final_test_roc = roc_auc_score(y_test, lasso.predict_proba(X_te_final)[:, 1])
    final_test_pr = average_precision_score(y_test, lasso.predict_proba(X_te_final)[:, 1])
    test_lift_pr = final_test_pr - baseline_pr_test
    
    print(f"  > Lasso LR ROC-AUC:    {final_test_roc:.4f}")
    print(f"  > Lasso LR PR-AUC:     {final_test_pr:.4f} (True Lift: {test_lift_pr:+.4f}) | Surviving Features: {surviving}/{len(features)}")
    
    if "ENGINEERED" in set_name:
        coef_df = pd.DataFrame({'Feature': features, 'Coefficient': lasso.coef_[0]})
        fallback_seq_winners = coef_df[(coef_df['Feature'].str.startswith('SEQ_')) & (coef_df['Coefficient'] != 0)].copy()
        
    if "HYBRID" in set_name:
        coef_df = pd.DataFrame({'Feature': features, 'Coefficient': lasso.coef_[0]})
        seq_winners = coef_df[(coef_df['Feature'].str.startswith('SEQ_')) & (coef_df['Coefficient'] != 0)].copy()
        
        if seq_winners.empty and not fallback_seq_winners.empty:
            print("\n    [!] NOTE: The Hybrid model penalized all sequences out. Falling back to the 'Engineered Seqs' model to generate visual dashboard.")
            seq_winners = fallback_seq_winners
            source_model = "Engineered Sequences Model"
        elif not seq_winners.empty:
            source_model = "Hybrid Model"
        else:
            source_model = None

if source_model is not None:
    seq_winners['Magnitude'] = seq_winners['Coefficient'].abs()
    seq_winners = seq_winners.sort_values(by='Magnitude', ascending=False)
    print(f"\n    [!] CLINICAL PROOF ({source_model}): {len(seq_winners)} trajectories proved independent value.")
    
    # Process ALL surviving features instead of just the top 5
    all_winning_features = seq_winners['Feature'].tolist()
    
    # --- POST-SELECTION INFERENCE & BOOTSTRAP ---
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.utils import resample
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    print("      * Calculating Unbiased ORs and 95% CIs (Post-Selection Inference) for all surviving pathways...")
    
    # Isolate only the winning features to remove Lasso shrinkage bias
    X_selected = X_tr_final[all_winning_features]
    y_selected = y_tr_final
    
    # 1. Fit an unpenalized model to get the true, unbiased Odds Ratios (using C=1e9 to simulate no penalty)
    unbiased_model = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000, class_weight='balanced')
    unbiased_model.fit(X_selected, y_selected)
    unbiased_ors = np.exp(unbiased_model.coef_[0])
    
    # 2. Bootstrap the unbiased model to get robust CIs
    n_boot = 100
    boot_coefs = {feat: [] for feat in all_winning_features}
    
    for b in range(n_boot):
        X_b, y_b = resample(X_selected, y_selected, random_state=b)
        boot_model = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000, class_weight='balanced')
        boot_model.fit(X_b, y_b)
        
        for i, feat in enumerate(all_winning_features):
            boot_coefs[feat].append(boot_model.coef_[0][i])
    # ------------------------------------

    significant_count = 0
    print(f"\n      [!] STATISTICALLY SIGNIFICANT PATHWAYS (95% CI strictly excludes 1.00):")

    for idx, feat in enumerate(all_winning_features):
        # 95% CI from the Bootstrap distribution
        or_distribution = np.exp(boot_coefs[feat])
        lower_ci = np.percentile(or_distribution, 2.5)
        upper_ci = np.percentile(or_distribution, 97.5)
        
        # Empirical Gate: Check if statistically significant (strictly > 1 OR strictly < 1)
        if (lower_ci > 1.0 and upper_ci > 1.0) or (lower_ci < 1.0 and upper_ci < 1.0):
            significant_count += 1
            
            # Format the clinical string
            dose_match = re.search(r'_\[NODE_(\d+)_DOSE\]', feat)
            dose_str = f" (Node {dose_match.group(1)} Dose Limit)" if dose_match else ""
            clean_feature = re.sub(r'_\[NODE_\d+_DOSE\]', '', feat)
            clinical_chain = [get_clinical_desc(p) for p in clean_feature.replace("SEQ_", "").split("_TO_")]
            
            # Primary Unbiased Odds Ratio
            odds_ratio = unbiased_ors[idx]
            
            print(f"      * Pathway: {'  --->  '.join(clinical_chain)}{dose_str} \n          -> OR: {odds_ratio:.2f} [95% CI: {lower_ci:.2f} - {upper_ci:.2f}]")

    if significant_count == 0:
        print("      * No pathways achieved strict 95% CI significance post-bootstrap.")

    # ==========================================
    # 8. DUAL PATHWAY DASHBOARDS (D3.js)
    # ==========================================
    print("\n" + "="*85)
    print(f"PHASE 8: GENERATING INTERACTIVE PATHWAY EXPLORERS (Source: {source_model})")
    print("="*85)
    html_template = """
    <!DOCTYPE html>
    <meta charset="utf-8">
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f8f9fa; margin: 0; padding: 20px; overflow: hidden; }
      #header-box { position: absolute; top: 20px; left: 20px; z-index: 10; background: rgba(248, 249, 250, 0.9); padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); max-width: 600px;}
      h2 { margin: 0 0 10px 0; font-size: 20px; color: #2c3e50; }
      p { margin: 0; font-size: 14px; color: #7f8c8d; line-height: 1.5; }
      .controls-hint { display: inline-block; margin-top: 10px; font-size: 12px; font-weight: bold; color: #3498db; padding: 4px 8px; background: #e8f4f8; border-radius: 4px; }
      #tree-container { width: 100vw; height: 100vh; position: absolute; top: 0; left: 0; cursor: grab; }
      #tree-container:active { cursor: grabbing; }
      .node { cursor: pointer; }
      .node circle { stroke: #2c3e50; stroke-width: 2px; }
      .node text { font: 12px sans-serif; text-shadow: 0 1px 3px rgba(255,255,255,0.8); }
      .link { fill: none; stroke: #ccc; stroke-width: 2px; }
      .link--loop { fill: none; stroke: #8e44ad; stroke-width: 3px; stroke-dasharray: 6,4; }
      .node--positive circle { fill: #34495e; }
      .node--negative circle { fill: #ffffff; stroke-dasharray: 4,2; stroke: #7f8c8d; }
      .node--outcome circle { fill: #e74c3c; stroke: #c0392b; r: 6;}
      .node--outcome-good circle { fill: #2ecc71; stroke: #27ae60; r: 6;}
      .node--endpoint circle { fill: #f39c12; stroke: #e67e22; r: 5;}
      .node--outcome text { font-weight: bold; fill: #c0392b;}
      .node--outcome-good text { font-weight: bold; fill: #27ae60;}
      .node--endpoint text { font-weight: bold; fill: #d35400;}
      .node--negative text { fill: #7f8c8d; font-style: italic;}
    </style>
    <body>
    <div id="header-box">
        <h2>__TITLE_PLACEHOLDER__</h2>
        <p>__DESC_PLACEHOLDER__<br>Hollow/dashed nodes = <i>absence</i>. Purple arcs = <i>Recurrent Loops</i>.</p>
        <div class="controls-hint">🖱️ Scroll to Zoom | Click & Drag to Pan</div>
    </div>
    <div id="tree-container"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    const treeData = __DATA_PLACEHOLDER__;
    let i = 0; const margin = {top: 150, right: 120, bottom: 20, left: 180};
    const svg = d3.select("#tree-container").append("svg").attr("width", "100%").attr("height", "100%");
    const svgGroup = svg.append("g");
    const zoom = d3.zoom().scaleExtent([0.1, 3]).on("zoom", (event) => { svgGroup.attr("transform", event.transform); });
    svg.call(zoom);
    svg.call(zoom.transform, d3.zoomIdentity.translate(margin.left, margin.top).scale(0.9));
    const root = d3.hierarchy(treeData);
    const treeLayout = d3.tree().nodeSize([40, 250]); 
    if (root.children) { root.children.forEach(collapse); }
    function collapse(d) { if (d.children) { d._children = d.children; d._children.forEach(collapse); d.children = null; } }
    update(root);
    function update(source) {
      const treeData = treeLayout(root);
      const nodes = treeData.descendants(), links = treeData.descendants().slice(1);
      nodes.forEach(d => d.y = d.depth * 280);
      const node = svgGroup.selectAll("g.node").data(nodes, d => d.id || (d.id = ++i)); 
      const nodeEnter = node.enter().append("g")
          .attr("class", d => d.data.is_outcome ? (d.data.coef > 0 ? "node node--outcome" : "node node--outcome-good") : (d.data.is_endpoint ? "node node--endpoint" : (d.data.is_negative ? "node node--negative" : "node node--positive")))
          .attr("transform", d => `translate(${source.y0 || root.y},${source.x0 || root.x})`)
          .on("click", click);
      nodeEnter.append("circle").attr("r", 1e-6);
      nodeEnter.append("text").attr("dy", ".35em").attr("x", d => d.children || d._children ? -13 : 13)
          .attr("text-anchor", d => d.children || d._children ? "end" : "start").text(d => d.data.name);
      const nodeUpdate = nodeEnter.merge(node);
      nodeUpdate.transition().duration(400).attr("transform", d => `translate(${d.y},${d.x})`);
      nodeUpdate.select("circle").attr("r", 5);
      node.exit().transition().duration(400).attr("transform", d => `translate(${source.y},${source.x})`).remove().select("circle").attr("r", 1e-6);
      const link = svgGroup.selectAll("path.link").data(links, d => d.id);
      const linkEnter = link.enter().insert("path", "g")
          .attr("class", d => d.data.name.includes("(Recurrence)") ? "link link--loop" : "link")
          .attr("d", d => { const o = {x: source.x0 || root.x, y: source.y0 || root.y}; return diagonal(o, o); });
      linkEnter.merge(link).attr("class", d => d.data.name.includes("(Recurrence)") ? "link link--loop" : "link")
          .transition().duration(400).attr("d", d => diagonal(d, d.parent));
      link.exit().transition().duration(400).attr("d", d => { const o = {x: source.x, y: source.y}; return diagonal(o, o); }).remove();
      nodes.forEach(d => { d.x0 = d.x; d.y0 = d.y; });
      function diagonal(s, d) { return (d.data && d.data.name && d.data.name.includes("(Recurrence)")) ? `M ${s.y} ${s.x} A 160 160 0 0 1 ${d.y} ${d.x}` : `M ${s.y} ${s.x} C ${(s.y + d.y) / 2} ${s.x}, ${(s.y + d.y) / 2} ${d.x}, ${d.y} ${d.x}`; }
    }
    function click(event, d) { if (d.children) { d._children = d.children; d.children = null; } else { d.children = d._children; d._children = null; } update(d); }
    </script>
    </body>
    </html>
    """
    if source_model is not None and not seq_winners.empty:
        tree_data_lasso = {"name": "Root: Clinical Cohort", "children": []}
        def insert_path_lasso(current_node, path, coef, dose_info):
            if not path:
                or_val = np.exp(coef)
                outcome_str = f"OUTCOME: {'Increased' if coef > 0 else 'Decreased'} Risk (OR: {or_val:.2f}) {dose_info}"
                if "children" not in current_node: current_node["children"] = []
                current_node["children"].append({"name": outcome_str, "is_outcome": True, "coef": coef})
                return
            node_name = get_clinical_desc(path[0])
            if "children" not in current_node: current_node["children"] = []
            target_child = next((c for c in current_node["children"] if c["name"] == node_name), None)
            if not target_child:
                target_child = {"name": node_name, "is_negative": path[0].startswith("NOT_")}
                current_node["children"].append(target_child)
            insert_path_lasso(target_child, path[1:], coef, dose_info)
        for _, row in seq_winners.iterrows():
            dose_match = re.search(r'_\[NODE_(\d+)_DOSE\]', row['Feature'])
            clean_f = re.sub(r'_\[NODE_\d+_DOSE\]', '', row['Feature'])
            insert_path_lasso(tree_data_lasso, clean_f.replace("SEQ_", "").split("_TO_"), row['Coefficient'], f"[Node {dose_match.group(1)} Dose]" if dose_match else "")
        html_lasso = html_template.replace("__DATA_PLACEHOLDER__", json.dumps(tree_data_lasso)).replace("__TITLE_PLACEHOLDER__", "Targeted Intervention Dashboard (Lasso Logic)").replace("__DESC_PLACEHOLDER__", "Displays ONLY chronological pathways possessing mathematically independent predictive power for the target outcome.")
        with open("DASHBOARD_1_Outcome_Drivers.html", "w") as f: f.write(html_lasso)
        print("--> SUCCESS: Saved 'DASHBOARD_1_Outcome_Drivers.html'")

    if best_edges:
        tree_data_markov = {"name": "Root: Clinical Cohort", "children": []}
        def insert_path_markov(current_node, path):
            if not path:
                if "children" not in current_node: current_node["children"] = []
                if not any(c.get("is_endpoint") for c in current_node["children"]): current_node["children"].append({"name": "Validated Pathway End", "is_endpoint": True})
                return
            node_name = get_clinical_desc(path[0])
            if "children" not in current_node: current_node["children"] = []
            target_child = next((c for c in current_node["children"] if c["name"] == node_name), None)
            if not target_child:
                target_child = {"name": node_name, "is_negative": path[0].startswith("NOT_")}
                current_node["children"].append(target_child)
            insert_path_markov(target_child, path[1:])
        for seq_tuple in best_edges: insert_path_markov(tree_data_markov, seq_tuple)
        html_markov = html_template.replace("__DATA_PLACEHOLDER__", json.dumps(tree_data_markov)).replace("__TITLE_PLACEHOLDER__", "General Pathway Explorer (Markov Logic)").replace("__DESC_PLACEHOLDER__", "Displays ALL sequences that survived statistical significance and divergence thresholds.")
        with open("DASHBOARD_2_All_Validated_Pathways.html", "w") as f: f.write(html_markov)
        print("--> SUCCESS: Saved 'DASHBOARD_2_All_Validated_Pathways.html'")

print("\nPipeline Complete.")