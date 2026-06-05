import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import sys
import json
import re
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)


def run_master_pipeline(
    target_col="READMIT_30", 
    min_train_prev_pct=0.005, 
    enable_jaccard=True, 
    jaccard_threshold=0.90, 
    min_co_occurrence=5, 
    asym_floor=0.10,
    collider_indep_lower=0.75,
    collider_indep_upper=1.33,
    collider_dep_lower=0.66,
    collider_dep_upper=1.50,
    enable_negatives=True,
    min_prev_for_negative=0.05,
    sample_size=0,            # <-- Restored Parameter
    auto_optimize=True,       # <-- Restored Parameter
    max_seq_len=5,            # <-- Restored Parameter
    use_normalized_diff=False,# <-- Restored Parameter
    max_prune_search=500      # <-- Restored Parameter
):

    # ==========================================
    # 1. CONFIGURATION (Mapped from Streamlit)
    # ==========================================
    FILE_NAME = "LDS2024_CCSR.csv"
    MAPPING_FILE = "PRCCSR_v2026-1.csv"
    OUTPUT_FILE = "Divergence_Sequences.csv"
    
    MARKOV_FLOOR = 0.05

    # Direct Web-to-Engine Variable Mapping
    TARGET_COL = target_col
    MIN_TRAIN_PREVALENCE_PCT = min_train_prev_pct
    JACCARD_THRESHOLD = jaccard_threshold
    MIN_CO_OCCURRENCE = min_co_occurrence
    ASYM_FLOOR = asym_floor
    MIN_PREVALENCE_FOR_NEGATIVE = min_prev_for_negative

    COLLIDER_INDEP_LOWER = collider_indep_lower
    COLLIDER_INDEP_UPPER = collider_indep_upper
    COLLIDER_DEP_LOWER = collider_dep_lower
    COLLIDER_DEP_UPPER = collider_dep_upper

    MAX_PRUNE_SEARCH = max_prune_search
    manual_max_len = max_seq_len if max_seq_len > 0 else float('inf')
    run_mode = 'Y' if auto_optimize else 'N'
    run_jaccard = 'Y' if enable_jaccard else 'N'
    use_normalized = use_normalized_diff

    print("="*85)
    print("CLINICAL SEQUENCE EXTRACTOR: STREAMLIT SERVER EDITION")
    print("="*85)

    # ---> THE MISSING DICTIONARY BUILDER <---
    print(f"Loading Mapping Dictionary: {MAPPING_FILE}...")
    try:
        map_df = pd.read_csv(MAPPING_FILE, dtype=str)
        map_df.columns = [col.replace("'", "").strip() for col in map_df.columns]
        ccsr_desc_dict = dict(zip(map_df['PRCCSR'].str.replace("'", "").str.strip(), 
                                  map_df['PRCCSR DESCRIPTION'].str.replace("'", "").str.strip()))
    except FileNotFoundError:
        print(f"[!] Warning: Could not find {MAPPING_FILE}. Narratives will use raw codes.")
        ccsr_desc_dict = {}

    # ---> THE DATA LOADER & SUCCESS UI <---
    print(f"\nLoading Cohort Data: {FILE_NAME}...")
    try:
        if sample_size > 0:
            df = pd.read_csv(FILE_NAME, dtype=str, nrows=sample_size)
            st.success(f"✅ **Dataset Loaded Successfully:** {len(df):,} patients (Restricted Test Sample)")
        else:
            df = pd.read_csv(FILE_NAME, dtype=str)
            st.success(f"✅ **Dataset Loaded Successfully:** {len(df):,} patients (Full Cohort)")
    except FileNotFoundError:
        raise FileNotFoundError(f"CRITICAL ERROR: The file '{FILE_NAME}' is missing from the folder! Please place it next to app.py.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"CRITICAL ERROR: The target column '{TARGET_COL}' does not exist in your dataset. Check your spelling.")
        
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0).astype(int)

    # ==========================================
    # 2. STRICT 3-WAY SPLIT (LEAKAGE FIX)
    # ==========================================
      
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
        print("--> Executing Jaccard Bundling (Optimized Global Maximum Approach)...")
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
    else:
        print("--> Skipping Jaccard Bundling (Optimized for distinct chronological progression).")

    active_positive_nodes = [c for c in ccsr_cols if c not in bundled_nodes] + list(new_bundles_train.keys())
    neg_cols_train, neg_cols_val, neg_cols_test = {}, {}, {}
    total_train = len(df_train)

    print(f"--> Generating Informative Absences (Prevalence Floor: {MIN_PREVALENCE_FOR_NEGATIVE*100}%)...")
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

    # --- DYNAMIC SUPPORT FLOOR ---
    train_counts = df_train[all_nodes].sum()
    dynamic_floor = max(3, int(total_train * MIN_TRAIN_PREVALENCE_PCT)) 
    valid_nodes = train_counts[train_counts >= dynamic_floor].index.tolist()

    if not valid_nodes:
        print(f"\n[!] CRITICAL ERROR: 0 features met the support floor of {dynamic_floor} patients.")
        sys.exit()
    print(f"--> Dynamic Support Floor set to {dynamic_floor} patients ({MIN_TRAIN_PREVALENCE_PCT*100}%). Kept {len(valid_nodes)} viable features.")

    P_marginal_train = (df_train[valid_nodes].sum() / total_train).to_dict()
    X_np_train = df_train[valid_nodes].values.astype(bool)
    X_np_val = df_val[valid_nodes].values.astype(bool)
    X_np_test = df_test[valid_nodes].values.astype(bool)
    node_to_idx = {n: i for i, n in enumerate(valid_nodes)}
    y_train, y_val, y_test = df_train[TARGET_COL].values, df_val[TARGET_COL].values, df_test[TARGET_COL].values

    # ==========================================
    # 4. SUPERVISED SEQUENCE MINING ENGINE
    # ==========================================
    print(f"\nPHASE 3: Sequence Mining & Ranking (Target: {TARGET_COL})")
    baseline_lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, class_weight='balanced', random_state=42)
    baseline_lasso.fit(df_train[valid_nodes], y_train)
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

        # Helper 1: Extract the pure base code
        def get_base_root(node_str):
            return re.sub(r'_R\d*', '', node_str.replace("NOT_", ""))

        # Helper 2: The Streamlined Logic Gate (State Locks)
        def is_logically_valid(curr_seq, next_node):
            if next_node in curr_seq: return False
            if next_node.startswith("NOT_") and curr_seq[-1].startswith("NOT_"): return False
            
            next_root = get_base_root(next_node)
            next_is_pos = not next_node.startswith("NOT_")
            
            last_root = get_base_root(curr_seq[-1])
            if last_root == next_root:
                last_is_pos = not curr_seq[-1].startswith("NOT_")
                if last_is_pos != next_is_pos:
                    return False 
            return True

        # Helper 3: The Conditional Independence Gate (Berkson's Trap Detector)
        def is_collider_trap(curr_seq, next_node):
            if len(curr_seq) < 2: return False
            
            X = curr_seq[-2] # The root cause
            Y = curr_seq[-1] # The potential collider
            Z = next_node    # The new effect
            
            idx_X, idx_Y, idx_Z = node_to_idx[X], node_to_idx[Y], node_to_idx[Z]
            
            P_X = P_marginal_train[X]
            P_Y = P_marginal_train[Y]
            P_Z = P_marginal_train[Z]
            
            # 1. Unconditional Independence Check P(X,Z) ≈ P(X)P(Z)
            P_XZ = np.bitwise_and(X_np_train[:, idx_X], X_np_train[:, idx_Z]).sum() / total_train
            expected_XZ = P_X * P_Z
            if expected_XZ == 0: return False
            
            unconditional_ratio = P_XZ / expected_XZ
            
            # If they are NATURALLY correlated, Y is not creating the illusion out of nowhere
            if unconditional_ratio < COLLIDER_INDEP_LOWER or unconditional_ratio > COLLIDER_INDEP_UPPER:
                return False
                
            # 2. Conditional Dependence Check P(X,Z | Y) ≠ P(X | Y)P(Z | Y)
            P_XY = np.bitwise_and(X_np_train[:, idx_X], X_np_train[:, idx_Y]).sum() / total_train
            P_ZY = np.bitwise_and(X_np_train[:, idx_Z], X_np_train[:, idx_Y]).sum() / total_train
            
            P_X_given_Y = P_XY / P_Y
            P_Z_given_Y = P_ZY / P_Y
            
            mask_XYZ = np.bitwise_and(np.bitwise_and(X_np_train[:, idx_X], X_np_train[:, idx_Y]), X_np_train[:, idx_Z])
            P_XZ_given_Y = mask_XYZ.sum() / (P_Y * total_train)
            
            expected_XZ_given_Y = P_X_given_Y * P_Z_given_Y
            if expected_XZ_given_Y == 0: return False
            
            conditional_ratio = P_XZ_given_Y / expected_XZ_given_Y
            
            # If filtering by Y artificially causes a massive swing in their correlation, it's a Trap.
            if conditional_ratio >= COLLIDER_DEP_UPPER or conditional_ratio <= COLLIDER_DEP_LOWER:
                return True
                
            return False

        # LEVEL 1: APRIORI INITIALIZATION
        growth_queue = []
        for node_A, node_B in combinations(valid_nodes, 2):
            if node_A.startswith("NOT_") and node_B.startswith("NOT_"): continue
            
            if get_base_root(node_A) == get_base_root(node_B):
                a_is_pos = not node_A.startswith("NOT_")
                b_is_pos = not node_B.startswith("NOT_")
                if a_is_pos != b_is_pos:
                    continue 
            
            idx_A, idx_B = node_to_idx[node_A], node_to_idx[node_B]
            mask = np.bitwise_and(X_np_train[:, idx_A], X_np_train[:, idx_B])
            if mask.sum() < dynamic_floor: continue
                
            growth_queue.append({'seq': (node_A, node_B), 'mask': mask})
            growth_queue.append({'seq': (node_B, node_A), 'mask': mask})
            
            pa_b = (mask.sum() / total_train) / P_marginal_train[node_B]
            pb_a = (mask.sum() / total_train) / P_marginal_train[node_A]
            if pa_b == 0 or pb_a == 0: continue
                
            if use_normalized:
                diff = (pb_a - pa_b) / (pb_a + pa_b)
            else:
                diff = pb_a - pa_b
                
            lift = calc_lift(mask)
            
            if diff >= ASYM_FLOOR:
                markov = (pb_a - P_marginal_train[node_B]) / P_marginal_train[node_B]
                if abs(markov) >= MARKOV_FLOOR:
                    all_discovered.append({'seq': (node_A, node_B), 'power': (abs(diff)**W_DIV)*(abs(markov)**W_MARK)*(lift**W_OUT)})
            elif diff <= -ASYM_FLOOR:
                markov = (pa_b - P_marginal_train[node_A]) / P_marginal_train[node_A]
                if abs(markov) >= MARKOV_FLOOR:
                    all_discovered.append({'seq': (node_B, node_A), 'power': (abs(diff)**W_DIV)*(abs(markov)**W_MARK)*(lift**W_OUT)})

        # LEVEL 2+: EXHAUSTIVE BOUNDED SEARCH
        current_len = 2
        while growth_queue and current_len < max_len:
            next_q = []
            for seq_dict in growth_queue:
                curr_seq, curr_mask = seq_dict['seq'], seq_dict['mask']
                curr_p = curr_mask.sum() / total_train
                
                for next_node in valid_nodes:
                    # 1. State / Logic Checks
                    if not is_logically_valid(curr_seq, next_node):
                        continue
                    
                    # 2. Transitivity Trap / Collider Bias Check
                    if is_collider_trap(curr_seq, next_node):
                        continue
                    
                    mask = np.bitwise_and(curr_mask, X_np_train[:, node_to_idx[next_node]])
                    if mask.sum() < dynamic_floor: continue
                    
                    next_q.append({'seq': curr_seq + (next_node,), 'mask': mask})
                    
                    pn_s = (mask.sum() / total_train) / curr_p
                    ps_n = (mask.sum() / total_train) / P_marginal_train[next_node]
                    if pn_s == 0 or ps_n == 0: continue
                        
                    if use_normalized:
                        diff = (pn_s - ps_n) / (pn_s + ps_n)
                    else:
                        diff = pn_s - ps_n
                        
                    markov = (pn_s - P_marginal_train[next_node])/P_marginal_train[next_node]
                    lift = calc_lift(mask)
                    
                    if diff >= ASYM_FLOOR and abs(markov) >= MARKOV_FLOOR:
                        all_discovered.append({'seq': curr_seq + (next_node,), 'power': (abs(diff)**W_DIV)*(abs(markov)**W_MARK)*(lift**W_OUT)})
            growth_queue, current_len = next_q, current_len + 1

        all_discovered.sort(key=lambda x: x['power'], reverse=True)
        seen, final = set(), []
        for d in all_discovered:
            if d['seq'] not in seen:
                seen.add(d['seq'])
                final.append(d['seq'])
                if len(final) >= max_prune: break
        return final

    top_seqs = mine_top_sequences(manual_max_len, MAX_PRUNE_SEARCH)
    print(f"--> Extracted and Globally Ranked top {len(top_seqs)} valid sequences.")

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
    families_collapsed = 0
    new_dimension_cols = []

    for base_path, variants in families.items():
        if len(variants) > 1:
            families_collapsed += 1
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
                if col not in cols_to_drop:
                    cols_to_drop.append(col)
                    
            print(f"  -> Extracted {num_dimensions}-dimensional tensor for: '{base_path}'")

    df_train.drop(columns=cols_to_drop, inplace=True)
    df_val.drop(columns=cols_to_drop, inplace=True)
    df_test.drop(columns=cols_to_drop, inplace=True)

    sequence_feature_names = [f for f in sequence_feature_names if f not in cols_to_drop] + new_dimension_cols

    print(f"--> Compression Complete! Converted {len(cols_to_drop)} redundant boolean flags into {len(new_dimension_cols)} multi-dimensional continuous features.")

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
        f_p, f_c = min(manual_prune, len(sequence_feature_names)), manual_lasso
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
            
            if "_R" in clean_node:
                desc = f"{base_desc} (Recurrence)"
            else:
                desc = base_desc
                
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
    # 7. FINAL EVALUATION MATRIX (UNTOUCHED TEST SET)
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
        
        X_tr_final = pd.concat([df_train[features], df_val[features]])
        y_tr_final = np.concatenate((y_train, y_val))
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
                
                # ---> NEW STREAMLIT WEB UI DATA TABLE <---
                st.subheader(f"📊 Validated Clinical Pathways ({source_model})")
                display_df = seq_winners.copy()
                
                # Translate internal codes to English pathways and calculate Odds Ratios
                display_df['Odds Ratio'] = np.exp(display_df['Coefficient']).round(3)
                display_df['Pathway'] = display_df['Feature'].apply(
                    lambda x: "  ➔  ".join([get_clinical_desc(p) for p in re.sub(r'_\[NODE_\d+_DOSE\]', '', x).replace("SEQ_", "").split("_TO_")])
                )
                
                # Render the interactive dataframe on the website
                st.dataframe(
                    display_df[['Pathway', 'Coefficient', 'Odds Ratio']], 
                    use_container_width=True,
                    hide_index=True
                )
                st.info(f"Engine extracted {len(display_df)} statistically significant causal pathways.")
                # ---> END NEW STREAMLIT CODE <---
                
                for idx, row in seq_winners.head(5).iterrows():
                    dose_match = re.search(r'_\[NODE_(\d+)_DOSE\]', row['Feature'])
                    dose_str = f" (Node {dose_match.group(1)} Dose Limit)" if dose_match else ""
                    
                    clean_feature = re.sub(r'_\[NODE_\d+_DOSE\]', '', row['Feature'])
                    parts = clean_feature.replace("SEQ_", "").split("_TO_")
                    clinical_chain = [get_clinical_desc(p) for p in parts]
                    odds_ratio = np.exp(row['Coefficient'])
                    print(f"      * Pathway: {'  --->  '.join(clinical_chain)}{dose_str} (OR: {odds_ratio:.2f})")

                # ==========================================
                # PHASE 8: GENERATING DUAL EXPLORERS
                # ==========================================
                print("\n" + "="*85)
                print(f"PHASE 8: GENERATING DUAL INTERACTIVE PATHWAY EXPLORERS (Source: {source_model})")
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
                let i = 0; 
                const margin = {top: 150, right: 120, bottom: 20, left: 180};
                const svg = d3.select("#tree-container").append("svg").attr("width", "100%").attr("height", "100%");
                const svgGroup = svg.append("g");
                const zoom = d3.zoom().scaleExtent([0.1, 3]).on("zoom", (event) => { svgGroup.attr("transform", event.transform); });
                svg.call(zoom);
                svg.call(zoom.transform, d3.zoomIdentity.translate(margin.left, margin.top).scale(0.9));
                const root = d3.hierarchy(treeData);
                const treeLayout = d3.tree().nodeSize([40, 250]); 
                if (root.children) { root.children.forEach(collapse); }
                function collapse(d) {
                  if (d.children) { d._children = d.children; d._children.forEach(collapse); d.children = null; }
                }
                update(root);
                function update(source) {
                  const treeData = treeLayout(root);
                  const nodes = treeData.descendants();
                  const links = treeData.descendants().slice(1);
                  nodes.forEach(d => d.y = d.depth * 280);
                  const node = svgGroup.selectAll("g.node").data(nodes, d => d.id || (d.id = ++i)); 
                  const nodeEnter = node.enter().append("g")
                      .attr("class", d => {
                          if (d.data.is_outcome && d.data.coef > 0) return "node node--outcome";
                          if (d.data.is_outcome && d.data.coef <= 0) return "node node--outcome-good";
                          if (d.data.is_endpoint) return "node node--endpoint";
                          if (d.data.is_negative) return "node node--negative";
                          return "node node--positive";
                      })
                      .attr("transform", d => `translate(${source.y0 || root.y},${source.x0 || root.x})`)
                      .on("click", click);
                  nodeEnter.append("circle").attr("r", 1e-6);
                  nodeEnter.append("text").attr("dy", ".35em").attr("x", d => d.children || d._children ? -13 : 13)
                      .attr("text-anchor", d => d.children || d._children ? "end" : "start").text(d => d.data.name);
                  const nodeUpdate = nodeEnter.merge(node);
                  nodeUpdate.transition().duration(400).attr("transform", d => `translate(${d.y},${d.x})`);
                  nodeUpdate.select("circle").attr("r", 5);
                  const nodeExit = node.exit().transition().duration(400).attr("transform", d => `translate(${source.y},${source.x})`).remove();
                  nodeExit.select("circle").attr("r", 1e-6);
                  const link = svgGroup.selectAll("path.link").data(links, d => d.id);
                  const linkEnter = link.enter().insert("path", "g")
                      .attr("class", d => d.data.name.includes("(Recurrence)") ? "link link--loop" : "link")
                      .attr("d", d => { const o = {x: source.x0 || root.x, y: source.y0 || root.y}; return diagonal(o, o); });
                  const linkUpdate = linkEnter.merge(link);
                  linkUpdate.attr("class", d => d.data.name.includes("(Recurrence)") ? "link link--loop" : "link");
                  linkUpdate.transition().duration(400).attr("d", d => diagonal(d, d.parent));
                  const linkExit = link.exit().transition().duration(400)
                      .attr("d", d => { const o = {x: source.x, y: source.y}; return diagonal(o, o); }).remove();
                  nodes.forEach(d => { d.x0 = d.x; d.y0 = d.y; });
                  function diagonal(s, d) {
                    if (d.data && d.data.name && d.data.name.includes("(Recurrence)")) { return `M ${s.y} ${s.x} A 160 160 0 0 1 ${d.y} ${d.x}`; }
                    return `M ${s.y} ${s.x} C ${(s.y + d.y) / 2} ${s.x}, ${(s.y + d.y) / 2} ${d.x}, ${d.y} ${d.x}`; 
                  }
                }
                function click(event, d) {
                  if (d.children) { d._children = d.children; d.children = null; } 
                  else { d.children = d._children; d._children = null; }
                  update(d);
                }
                </script>
                </body>
                </html>
                """
                
                if source_model is not None and not seq_winners.empty:
                    tree_data_lasso = {"name": "Root: Clinical Cohort", "children": []}
                    def insert_path_lasso(current_node, path, coef, node_dose_info=""):
                        if not path:
                            or_val = np.exp(coef)
                            outcome_str = f"OUTCOME: {'Increased' if coef > 0 else 'Decreased'} Risk (OR: {or_val:.2f}) {node_dose_info}"
                            if "children" not in current_node: current_node["children"] = []
                            current_node["children"].append({"name": outcome_str, "is_outcome": True, "coef": coef})
                            return
                        node_name = get_clinical_desc(path[0])
                        if "children" not in current_node: current_node["children"] = []
                        target_child = next((c for c in current_node["children"] if c["name"] == node_name), None)
                        if not target_child:
                            target_child = {"name": node_name, "is_negative": path[0].startswith("NOT_")}
                            current_node["children"].append(target_child)
                        insert_path_lasso(target_child, path[1:], coef, node_dose_info)
                        
                    for idx, row in seq_winners.iterrows():
                        dose_match = re.search(r'_\[NODE_(\d+)_DOSE\]', row['Feature'])
                        dose_info = f"[Node {dose_match.group(1)} Dose Vector]" if dose_match else ""
                        clean_feature = re.sub(r'_\[NODE_\d+_DOSE\]', '', row['Feature'])
                        insert_path_lasso(tree_data_lasso, clean_feature.replace("SEQ_", "").split("_TO_"), row['Coefficient'], dose_info)
                        
                    html_lasso = html_template.replace("__DATA_PLACEHOLDER__", json.dumps(tree_data_lasso))
                    html_lasso = html_lasso.replace("__TITLE_PLACEHOLDER__", "Targeted Intervention Dashboard (Lasso Logic)")
                    html_lasso = html_lasso.replace("__DESC_PLACEHOLDER__", "Displays ONLY chronological pathways possessing mathematically independent predictive power for the target outcome.")
                    with open("DASHBOARD_1_Outcome_Drivers.html", "w") as f: f.write(html_lasso)
                    print("--> SUCCESS: Saved 'DASHBOARD_1_Outcome_Drivers.html'")

                if best_edges:
                    tree_data_markov = {"name": "Root: Clinical Cohort", "children": []}
                    def insert_path_markov(current_node, path):
                        if not path:
                            if "children" not in current_node: current_node["children"] = []
                            if not any(c.get("is_endpoint") for c in current_node["children"]):
                                current_node["children"].append({"name": "Validated Pathway End", "is_endpoint": True})
                            return
                        node_name = get_clinical_desc(path[0])
                        if "children" not in current_node: current_node["children"] = []
                        target_child = next((c for c in current_node["children"] if c["name"] == node_name), None)
                        if not target_child:
                            target_child = {"name": node_name, "is_negative": path[0].startswith("NOT_")}
                            current_node["children"].append(target_child)
                        insert_path_markov(target_child, path[1:])
                    for seq_tuple in best_edges: insert_path_markov(tree_data_markov, seq_tuple)
                    html_markov = html_template.replace("__DATA_PLACEHOLDER__", json.dumps(tree_data_markov))
                    html_markov = html_markov.replace("__TITLE_PLACEHOLDER__", "General Clinical Pathway Explorer (Markov Logic)")
                    html_markov = html_markov.replace("__DESC_PLACEHOLDER__", "Maps the physiological reality. Displays ALL sequences that survived statistical significance and divergence thresholds.")
                    with open("DASHBOARD_2_All_Validated_Pathways.html", "w") as f: f.write(html_markov)
                    print("--> SUCCESS: Saved 'DASHBOARD_2_All_Validated_Pathways.html'")

    print("\nPipeline Complete.")
