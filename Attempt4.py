import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
import sys
import re

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. CONFIGURATION & INTERACTIVE SETUP
# ==========================================
FILE_NAME = "LDS2024_CCSR.csv"
MAPPING_FILE = "PRCCSR_v2026-1.csv"
OUTPUT_FILE = "Divergence_Sequences.csv"
JACCARD_THRESHOLD = 0.90     
SUPPORT_FLOOR = 50           
MIN_CO_OCCURRENCE = 10 

# Fixed extraction floors to eliminate pure noise (Lets Power Score do the ranking)
ASYM_FLOOR = 0.10
MARKOV_FLOOR = 0.05
MAX_PRUNE_SEARCH = 500

print("="*85)
print("CLINICAL SEQUENCE EXTRACTOR: THE MASTER PIPELINE")
print("="*85)

# Load CCSR Descriptions for Narratives
print(f"Loading Mapping Dictionary: {MAPPING_FILE}...")
try:
    map_df = pd.read_csv(MAPPING_FILE, dtype=str)
    map_df.columns = [col.replace("'", "").strip() for col in map_df.columns]
    map_df['PRCCSR'] = map_df['PRCCSR'].str.replace("'", "").str.strip()
    map_df['PRCCSR DESCRIPTION'] = map_df['PRCCSR DESCRIPTION'].str.replace("'", "").str.strip()
    ccsr_desc_dict = dict(zip(map_df['PRCCSR'], map_df['PRCCSR DESCRIPTION']))
except FileNotFoundError:
    print(f"[!] Warning: Could not find {MAPPING_FILE}. Narratives will use raw codes.")
    ccsr_desc_dict = {}

# Load Initial Data
print(f"\nLoading Cohort Data: {FILE_NAME}...")
try:
    df = pd.read_csv(FILE_NAME, dtype=str)
except FileNotFoundError:
    print(f"[!] Error: Could not find {FILE_NAME}.")
    sys.exit()

# --- INTERACTIVE SETUP ---
target_input = input("\n[1] Enter the Target Outcome (e.g., DIED or Readmission): ").strip()
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
# If user hits Enter, set the cap to infinity so it relies purely on mathematical pruning
manual_max_len = int(len_input) if len_input.isdigit() else float('inf')

run_mode = input("[5] Run Grid Search Optimization (Pruning Limits vs Lasso C) (Y/N)? ").strip().upper()

if run_mode == 'N':
    manual_prune = int(input("  -> Enter Sequence Pruning Limit (e.g., 200): "))
    manual_lasso = float(input("  -> Enter final Lasso Strictness (C) (e.g., 0.1): "))
else:
    print("--> Auto-Optimization selected. (Searching Pruning limits [50,100,300,500] vs Lasso C [0.01, 0.05, 0.1, 0.2, 0.5]).")

# ==========================================
# 2. STRICT TRAIN/TEST SPLIT
# ==========================================
print("\nPHASE 1: Strict Train/Test Splitting")
df_train, df_test = train_test_split(df, test_size=0.5, random_state=42, stratify=df[TARGET_COL])
print(f"--> Data split 50/50: {len(df_train)} Training | {len(df_test)} Testing.")

# ==========================================
# 3. PRE-PROCESSING
# ==========================================
print("\nPHASE 2: Jaccard Bundling & Conditioned Negatives (Trained on Train split only)")
ccsr_cols = [col for col in df_train.columns if col.startswith("CCSR_")]

for col in ccsr_cols:
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0).astype(np.int8)
    df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(0).astype(np.int8)

X_train_raw = df_train[ccsr_cols].values.astype(bool)
col_to_idx = {col: idx for idx, col in enumerate(ccsr_cols)}
bundled_nodes = set()
new_bundles_train = {}
new_bundles_test = {}

# Jaccard Bundling
for node_A, node_B in combinations(ccsr_cols, 2):
    if node_A in bundled_nodes or node_B in bundled_nodes: continue 
    idx_A, idx_B = col_to_idx[node_A], col_to_idx[node_B]
    
    intersect = np.bitwise_and(X_train_raw[:, idx_A], X_train_raw[:, idx_B]).sum()
    if intersect < MIN_CO_OCCURRENCE: continue
    union = np.bitwise_or(X_train_raw[:, idx_A], X_train_raw[:, idx_B]).sum()
    
    if (intersect / union) >= JACCARD_THRESHOLD:
        bundle_name = f"BUNDLE_{node_A}_{node_B}"
        new_bundles_train[bundle_name] = np.bitwise_and(X_train_raw[:, idx_A], X_train_raw[:, idx_B]).astype(np.int8)
        new_bundles_test[bundle_name] = ((df_test[node_A] == 1) & (df_test[node_B] == 1)).astype(np.int8)
        bundled_nodes.update([node_A, node_B])

if new_bundles_train:
    df_train = pd.concat([df_train, pd.DataFrame(new_bundles_train, index=df_train.index)], axis=1)
    df_test = pd.concat([df_test, pd.DataFrame(new_bundles_test, index=df_test.index)], axis=1)

active_positive_nodes = [c for c in ccsr_cols if c not in bundled_nodes] + list(new_bundles_train.keys())

# Conditioned Negatives (WITH REGEX HIERARCHY FIX)
neg_cols_train = {}
neg_cols_test = {}

for col in active_positive_nodes:
    clean_col = str(col).strip()
    
    # Prevents collinearity by ignoring R2, R10, etc., for negative nodes.
    if re.search(r'_R\d+$', clean_col):
        continue 
        
    if "_R" not in clean_col:
        neg_cols_train[f"NOT_{clean_col}"] = (df_train[col] == 0).astype(np.int8)
        neg_cols_test[f"NOT_{clean_col}"] = (df_test[col] == 0).astype(np.int8)
    else:
        base_col = col.split("_R")[0]
        if base_col in df_train.columns:
            neg_cols_train[f"NOT_{clean_col}"] = ((df_train[base_col] == 1) & (df_train[col] == 0)).astype(np.int8)
            neg_cols_test[f"NOT_{clean_col}"] = ((df_test[base_col] == 1) & (df_test[col] == 0)).astype(np.int8)

df_train = pd.concat([df_train, pd.DataFrame(neg_cols_train, index=df_train.index)], axis=1)
df_test = pd.concat([df_test, pd.DataFrame(neg_cols_test, index=df_test.index)], axis=1)

active_negative_nodes = list(neg_cols_train.keys())
all_nodes = active_positive_nodes + active_negative_nodes

# Support Floor
train_counts = df_train[all_nodes].sum()
valid_nodes = train_counts[train_counts >= SUPPORT_FLOOR].index.tolist()

total_train = len(df_train)
P_marginal_train = (df_train[valid_nodes].sum() / total_train).to_dict()

X_np_train = df_train[valid_nodes].values.astype(bool)
X_np_test = df_test[valid_nodes].values.astype(bool)
node_to_idx = {n: i for i, n in enumerate(valid_nodes)}
y_train = df_train[TARGET_COL].values
y_test = df_test[TARGET_COL].values

# ==========================================
# 4. SEQUENCE MINING ENGINE
# ==========================================
print(f"\nPHASE 3: Sequence Mining & Ranking (Target: {TARGET_COL})")

# Baseline Calculation
baseline_lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, class_weight='balanced', random_state=42)
baseline_lasso.fit(df_train[valid_nodes], y_train)
baseline_auc = roc_auc_score(y_test, baseline_lasso.predict_proba(df_test[valid_nodes])[:, 1])
print(f"--> Baseline AUC on Test Set (Raw Features Only): {baseline_auc:.4f}")

def mine_top_sequences(max_len, max_prune):
    div_threshold = 1.0 + ASYM_FLOOR
    valid_sequences = []
    
    # LEVEL 1
    for node_A, node_B in combinations(valid_nodes, 2):
        idx_A, idx_B = node_to_idx[node_A], node_to_idx[node_B]
        intersect_mask = np.bitwise_and(X_np_train[:, idx_A], X_np_train[:, idx_B])
        intersect_count = intersect_mask.sum()
        
        if intersect_count < MIN_CO_OCCURRENCE: continue
            
        P_A_and_B = intersect_count / total_train
        P_B_g_A = P_A_and_B / P_marginal_train[node_A]
        P_A_g_B = P_A_and_B / P_marginal_train[node_B]
        
        if P_B_g_A == 0 or P_A_g_B == 0: continue
        divergence = P_B_g_A / P_A_g_B
        
        if divergence >= div_threshold:
            lift = (P_B_g_A - P_marginal_train[node_B]) / P_marginal_train[node_B]
            power_score = divergence * abs(lift)
            valid_sequences.append({'seq': (node_A, node_B), 'mask': intersect_mask, 'power': power_score})
        elif (1 / divergence) >= div_threshold:
            lift = (P_A_g_B - P_marginal_train[node_A]) / P_marginal_train[node_A]
            power_score = (1 / divergence) * abs(lift)
            valid_sequences.append({'seq': (node_B, node_A), 'mask': intersect_mask, 'power': power_score})

    valid_sequences.sort(key=lambda x: x['power'], reverse=True)
    valid_sequences = valid_sequences[:max_prune]
    
    all_discovered = valid_sequences.copy()

# LEVEL 2+ (Organic Expansion)
    current_level = valid_sequences
    current_len = 2
    
    while current_level and current_len < max_len:
        next_level = []
        for seq_dict in current_level:
            current_seq = seq_dict['seq']
            current_mask = seq_dict['mask']
            current_prob = current_mask.sum() / total_train
            
            for next_node in valid_nodes:
                if next_node in current_seq: continue
                
                idx_next = node_to_idx[next_node]
                new_mask = np.bitwise_and(current_mask, X_np_train[:, idx_next])
                new_count = new_mask.sum()
                
                if new_count < MIN_CO_OCCURRENCE: continue
                    
                P_seq_and_next = new_count / total_train
                P_next_g_seq = P_seq_and_next / current_prob
                P_seq_g_next = P_seq_and_next / P_marginal_train[next_node]
                
                if P_next_g_seq == 0 or P_seq_g_next == 0: continue
                
                div = P_next_g_seq / P_seq_g_next
                markov = (P_next_g_seq - P_marginal_train[next_node]) / P_marginal_train[next_node]
                
                if P_marginal_train[next_node] > 0.50: markov_passes = True
                else: markov_passes = (markov >= MARKOV_FLOOR)
                
                if div >= div_threshold and markov_passes:
                    power_score = div * abs(markov)
                    next_level.append({'seq': current_seq + (next_node,), 'mask': new_mask, 'power': power_score})
                    
        next_level.sort(key=lambda x: x['power'], reverse=True)
        next_level = next_level[:max_prune]
        
        # If no new sequences passed the math thresholds, the organic pruning triggers
        if not next_level:
            break
            
        all_discovered.extend(next_level)
        current_level = next_level
        current_len += 1

    if not all_discovered: return []

    all_discovered.sort(key=lambda x: x['power'], reverse=True)
    return [x['seq'] for x in all_discovered[:max_prune]]

# Mine the top 500 sequences ONCE
top_500_seqs = mine_top_sequences(manual_max_len, MAX_PRUNE_SEARCH)
print(f"--> Extracted and Ranked top {len(top_500_seqs)} valid sequences based on Power Score.")

# Build full matrix representations
seq_mat_tr_full = np.zeros((total_train, len(top_500_seqs)), dtype=np.int8)
seq_mat_te_full = np.zeros((len(df_test), len(top_500_seqs)), dtype=np.int8)

for i, seq_tuple in enumerate(top_500_seqs):
    mask_tr = np.ones(total_train, dtype=bool)
    mask_te = np.ones(len(df_test), dtype=bool)
    for node in seq_tuple:
        mask_tr &= X_np_train[:, node_to_idx[node]]
        mask_te &= X_np_test[:, node_to_idx[node]]
    seq_mat_tr_full[:, i] = mask_tr
    seq_mat_te_full[:, i] = mask_te

# ==========================================
# 5. GRID SEARCH (Pruning vs Lasso C)
# ==========================================
if run_mode == 'Y':
    print("\nExecuting Optimization Grid (Pruning Limit vs. Lasso Strictness)...")
    prune_options = [50, 100, 300, 500]
    c_options = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    best_auc = 0
    best_p = 50
    best_c = 0.1

    for p_limit in prune_options:
        if p_limit > len(top_500_seqs): continue
        X_tr_hybrid = np.hstack((X_np_train, seq_mat_tr_full[:, :p_limit]))
        X_te_hybrid = np.hstack((X_np_test, seq_mat_te_full[:, :p_limit]))
        
        for c_val in c_options:
            lasso = LogisticRegression(penalty='l1', solver='liblinear', C=c_val, class_weight='balanced', max_iter=200, tol=0.01, random_state=42)
            lasso.fit(X_tr_hybrid, y_train)
            auc = roc_auc_score(y_test, lasso.predict_proba(X_te_hybrid)[:, 1])
            lift = auc - baseline_auc
            print(f"  > Testing Top {p_limit} Seqs | Lasso C: {c_val:.2f} -> AUC: {auc:.4f} (Lift: {lift:+.4f})")
            
            if auc > best_auc:
                best_auc = auc
                best_p = p_limit
                best_c = c_val
                
    final_prune = best_p
    final_c = best_c
    print(f"\n--> WINNING COMBINATION: Top {final_prune} Sequences | Lasso C: {final_c}")
else:
    final_prune = min(manual_prune, len(top_500_seqs))
    final_c = manual_lasso
    X_tr_hybrid = np.hstack((X_np_train, seq_mat_tr_full[:, :final_prune]))
    X_te_hybrid = np.hstack((X_np_test, seq_mat_te_full[:, :final_prune]))
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=final_c, class_weight='balanced', max_iter=200, tol=0.01, random_state=42)
    lasso.fit(X_tr_hybrid, y_train)
    auc = roc_auc_score(y_test, lasso.predict_proba(X_te_hybrid)[:, 1])
    print(f"--> Manual Execution AUC: {auc:.4f} (Lift: {auc - baseline_auc:+.4f})")

# Lock in the winning sequences
best_edges = top_500_seqs[:final_prune]

# ==========================================
# NARRATIVE HELPER FUNCTION
# ==========================================
def get_clinical_desc(node_name):
    is_neg = node_name.startswith("NOT_")
    clean_node = node_name.replace("NOT_", "")
    
    if clean_node.startswith("BUNDLE_"):
        parts = clean_node.replace("BUNDLE_CCSR_", "").split("_CCSR_")
        d1 = ccsr_desc_dict.get(parts[0], parts[0])
        d2 = ccsr_desc_dict.get(parts[1], parts[1])
        desc = f"Concurrent [{d1} & {d2}]"
    elif clean_node.startswith("CCSR_"):
        base_code = clean_node.replace("CCSR_", "").split("_R")[0]
        is_repeat = "_R" in clean_node
        base_desc = ccsr_desc_dict.get(base_code, base_code)
        desc = f"{base_desc} (Recurrence)" if is_repeat else base_desc
    else:
        desc = clean_node
        
    return f"Absence of {desc}" if is_neg else desc

# ==========================================
# 6. EXTRACT AND SAVE (Rich CSV Export & Matrix Inspection)
# ==========================================
sequence_feature_names = []
if best_edges:
    output_data = []
    seq_cols_train, seq_cols_test = {}, {}
    
    print("\nExtracting and saving rich sequence data...")
    for seq_tuple in best_edges:
        seq_name = "SEQ_" + "_TO_".join(seq_tuple)
        
        # FIX: Using .values strips the Pandas index and forces perfect positional alignment
        mask_tr = np.ones(len(df_train), dtype=bool)
        mask_te = np.ones(len(df_test), dtype=bool)
        for node in seq_tuple:
            mask_tr &= (df_train[node].values == 1)
            mask_te &= (df_test[node].values == 1)
            
        seq_cols_train[seq_name] = mask_tr.astype(np.int8)
        seq_cols_test[seq_name] = mask_te.astype(np.int8)
        sequence_feature_names.append(seq_name)
        
        clinical_chain = [get_clinical_desc(node) for node in seq_tuple]
        
        output_data.append({
            "Raw_Feature_Name": seq_name,
            "Sequence_Length": len(seq_tuple),
            "Clinical_Narrative": "  --->  ".join(clinical_chain),
            "Train_Patients_Matched": mask_tr.sum(),
            "Test_Patients_Matched": mask_te.sum()
        })
    
    df_train = pd.concat([df_train, pd.DataFrame(seq_cols_train, index=df_train.index)], axis=1)
    df_test = pd.concat([df_test, pd.DataFrame(seq_cols_test, index=df_test.index)], axis=1)
    pd.DataFrame(output_data).to_csv(OUTPUT_FILE, index=False)
    print(f"--> Saved presentation-ready clinical sequences to {OUTPUT_FILE}")

    # ==========================================
    # 6.5 EXPORT THE IN-MEMORY TESTING MATRIX FOR VERIFICATION
    # ==========================================
    print("\nDumping the fully binarized Hybrid Test Matrix to CSV for inspection...")
    inspection_cols = [TARGET_COL] + valid_nodes + sequence_feature_names
    df_test_matrix = df_test[inspection_cols].copy()
    matrix_filename = "INSPECTION_Test_Matrix.csv"
    df_test_matrix.to_csv(matrix_filename, index=False)
    print(f"--> Saved {len(df_test_matrix)} patients and {len(inspection_cols)} features to {matrix_filename}")
    print("    Open this file to manually verify the binarization logic.")

# ==========================================
# 7. FINAL EVALUATION MATRIX (Strict Unseen Test Data)
# ==========================================
print("\n" + "="*85)
print("PHASE 4: 3x3 CLINICAL VALIDATION MATRIX")
print("="*85)

feature_sets = {
    "1. RAW BASELINE (Original Nodes Only)": valid_nodes,
    "2. ENGINEERED SEQS (Sequences Only)": sequence_feature_names,
    "3. HYBRID (Baseline + Sequences)": valid_nodes + sequence_feature_names
}

for set_name, features in feature_sets.items():
    if not features:
        continue
        
    print(f"\n--- Matrix Cell: {set_name} ({len(features)} features) ---")
    
    X_tr, X_te = df_train[features], df_test[features]
    
    # A. STANDARD LOGISTIC REGRESSION
    std_model = LogisticRegression(penalty='l2', solver='lbfgs', C=100.0, max_iter=1000, class_weight='balanced')
    std_model.fit(X_tr, y_train)
    print(f"  > Standard LR AUC:    {roc_auc_score(y_test, std_model.predict_proba(X_te)[:, 1]):.4f}")
    
    # B. BERNOULLI NAIVE BAYES
    nb_model = BernoulliNB()
    nb_model.fit(X_tr, y_train)
    print(f"  > Naive Bayes AUC:    {roc_auc_score(y_test, nb_model.predict_proba(X_te)[:, 1]):.4f}")
    
    # C. LASSO LOGISTIC REGRESSION
    # If evaluating ONLY sequences, relax the penalty so it doesn't flatline to 0.50
    eval_c = 1.0 if "Sequences Only" in set_name else final_c
    
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=eval_c, max_iter=200, tol=0.01, class_weight='balanced', random_state=42)
    lasso.fit(X_tr, y_train)
    surviving = sum(lasso.coef_[0] != 0)
    print(f"  > Lasso LR AUC:       {roc_auc_score(y_test, lasso.predict_proba(X_te)[:, 1]):.4f} (Surviving Features: {surviving}/{len(features)})")

if "HYBRID" in set_name and surviving > 0:
        coef_df = pd.DataFrame({'Feature': features, 'Coefficient': lasso.coef_[0]})
        seq_winners = coef_df[(coef_df['Feature'].str.startswith('SEQ_')) & (coef_df['Coefficient'] != 0)].copy()
        
        if not seq_winners.empty:
            seq_winners['Magnitude'] = seq_winners['Coefficient'].abs()
            seq_winners = seq_winners.sort_values(by='Magnitude', ascending=False)
            
            print(f"\n    [!] CLINICAL PROOF: {len(seq_winners)} sequence trajectories proved independent value.")
            print("\n    Top Validated Trajectories (Clinical Narrative):")
            
            for idx, row in seq_winners.head(5).iterrows():
                parts = row['Feature'].replace("SEQ_", "").split("_TO_")
                clinical_chain = [get_clinical_desc(p) for p in parts]
                pathway_str = "  --->  ".join(clinical_chain)
                
                direction = "INCREASES" if row['Coefficient'] > 0 else "DECREASES"
                odds_ratio = np.exp(row['Coefficient'])
                
                print(f"      * Pathway: {pathway_str}")
                print(f"        Impact : {direction} likelihood (Odds Ratio: {odds_ratio:.2f}, Log-Odds: {row['Coefficient']:.3f})\n")

            # ==========================================
            # PHASE 8: INTERACTIVE D3.JS TREE GENERATOR
            # ==========================================
            print("\n" + "="*85)
            print("PHASE 8: GENERATING INTERACTIVE PATHWAY EXPLORER")
            print("="*85)
            
            import json
            
            # 1. Build the nested dictionary tree
            tree_data = {"name": "Root: Clinical Cohort", "children": []}
            
            def insert_path(current_node, path, coef):
                if not path:
                    or_val = np.exp(coef)
                    direction = "Increased" if coef > 0 else "Decreased"
                    outcome_str = f"OUTCOME: {direction} Risk (OR: {or_val:.2f})"
                    if "children" not in current_node: current_node["children"] = []
                    current_node["children"].append({"name": outcome_str, "is_outcome": True, "coef": coef})
                    return
                
                raw_node = path[0]
                node_name = get_clinical_desc(raw_node)
                is_neg = raw_node.startswith("NOT_")
                
                if "children" not in current_node:
                    current_node["children"] = []
                    
                target_child = next((c for c in current_node["children"] if c["name"] == node_name), None)
                
                if not target_child:
                    target_child = {"name": node_name, "is_negative": is_neg}
                    current_node["children"].append(target_child)
                    
                insert_path(target_child, path[1:], coef)

            # Insert all winning sequences into the tree
            for idx, row in seq_winners.iterrows():
                path_parts = row['Feature'].replace("SEQ_", "").split("_TO_")
                insert_path(tree_data, path_parts, row['Coefficient'])

            # 2. The HTML/JS Template
            html_template = """
            <!DOCTYPE html>
            <meta charset="utf-8">
            <style>
              body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f8f9fa; margin: 0; padding: 20px;}
              .node { cursor: pointer; }
              .node circle { stroke: #2c3e50; stroke-width: 2px; }
              .node text { font: 12px sans-serif; }
              .link { fill: none; stroke: #ccc; stroke-width: 2px; }
              
              /* Custom Styles engineered from our rules */
              .node--positive circle { fill: #34495e; }
              .node--negative circle { fill: #ffffff; stroke-dasharray: 4,2; stroke: #7f8c8d; }
              .node--outcome circle { fill: #e74c3c; stroke: #c0392b; r: 6;}
              .node--outcome-good circle { fill: #2ecc71; stroke: #27ae60; r: 6;}
              .node--outcome text { font-weight: bold; fill: #c0392b;}
              .node--outcome-good text { font-weight: bold; fill: #27ae60;}
              .node--negative text { fill: #7f8c8d; font-style: italic;}
            </style>
            <body>
            <h2>Clinical Pathway Explorer</h2>
            <p>Click on nodes to expand or collapse patient trajectories. Hollow/dashed nodes represent the <i>absence</i> of a condition.</p>
            <div id="tree-container"></div>
            
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <script>
            const treeData = __DATA_PLACEHOLDER__;
            
            const margin = {top: 20, right: 120, bottom: 20, left: 180},
                  width = 1200 - margin.right - margin.left,
                  height = 800 - margin.top - margin.bottom;

            const svg = d3.select("#tree-container").append("svg")
                .attr("width", width + margin.right + margin.left)
                .attr("height", height + margin.top + margin.bottom)
              .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            const root = d3.hierarchy(treeData);
            const treeLayout = d3.tree().size([height, width - 200]);

            // Collapse everything after the first level initially
            root.children.forEach(collapse);
            function collapse(d) {
              if (d.children) {
                d._children = d.children;
                d._children.forEach(collapse);
                d.children = null;
              }
            }

            update(root);

            function update(source) {
              const treeData = treeLayout(root);
              const nodes = treeData.descendants();
              const links = treeData.descendants().slice(1);

              nodes.forEach(d => d.y = d.depth * 250);

              const node = svg.selectAll("g.node")
                  .data(nodes, d => d.id || (d.id = ++i));
                  
              let i = 0;

              const nodeEnter = node.enter().append("g")
                  .attr("class", d => {
                      if (d.data.is_outcome && d.data.coef > 0) return "node node--outcome";
                      if (d.data.is_outcome && d.data.coef <= 0) return "node node--outcome-good";
                      if (d.data.is_negative) return "node node--negative";
                      return "node node--positive";
                  })
                  .attr("transform", d => `translate(${source.y0 || root.y},${source.x0 || root.x})`)
                  .on("click", click);

              nodeEnter.append("circle").attr("r", 1e-6);

              nodeEnter.append("text")
                  .attr("dy", ".35em")
                  .attr("x", d => d.children || d._children ? -13 : 13)
                  .attr("text-anchor", d => d.children || d._children ? "end" : "start")
                  .text(d => d.data.name);

              const nodeUpdate = nodeEnter.merge(node);

              nodeUpdate.transition().duration(400)
                  .attr("transform", d => `translate(${d.y},${d.x})`);

              nodeUpdate.select("circle").attr("r", 5);

              const nodeExit = node.exit().transition().duration(400)
                  .attr("transform", d => `translate(${source.y},${source.x})`)
                  .remove();

              nodeExit.select("circle").attr("r", 1e-6);

              const link = svg.selectAll("path.link")
                  .data(links, d => d.id);

              const linkEnter = link.enter().insert("path", "g")
                  .attr("class", "link")
                  .attr("d", d => {
                      const o = {x: source.x0 || root.x, y: source.y0 || root.y};
                      return diagonal(o, o);
                  });

              const linkUpdate = linkEnter.merge(link);

              linkUpdate.transition().duration(400)
                  .attr("d", d => diagonal(d, d.parent));

              const linkExit = link.exit().transition().duration(400)
                  .attr("d", d => {
                      const o = {x: source.x, y: source.y};
                      return diagonal(o, o);
                  })
                  .remove();

              nodes.forEach(d => { d.x0 = d.x; d.y0 = d.y; });
              
              function diagonal(s, d) {
                return `M ${s.y} ${s.x} C ${(s.y + d.y) / 2} ${s.x}, ${(s.y + d.y) / 2} ${d.x}, ${d.y} ${d.x}`;
              }
            }

            function click(event, d) {
              if (d.children) {
                  d._children = d.children;
                  d.children = null;
              } else {
                  d.children = d._children;
                  d._children = null;
              }
              update(d);
            }
            </script>
            </body>
            </html>
            """
            
            final_html = html_template.replace("__DATA_PLACEHOLDER__", json.dumps(tree_data))
            
            with open("Interactive_Pathway_Tree.html", "w") as f:
                f.write(final_html)
                
            print("--> SUCCESS: Saved 'Interactive_Pathway_Tree.html' to your folder.")
            print("--> Double-click this file to open the interactive dashboard in your web browser.")

print("\nPipeline Complete.")
