import pandas as pd
import sys
import numpy as np
from itertools import permutations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel

# ==========================================
# 1. DATA INGESTION & USER CONFIGURATION
# ==========================================
input_file = "2024_SEED_CCS.csv" 
print(f"Loading dataset: {input_file}...")
df = pd.read_csv(input_file, dtype=str)

df['DIED'] = (df['STUS_CD'].str.strip() == '20').astype(int)
if 'Readmission' in df.columns:
    df['Readmission'] = pd.to_numeric(df['Readmission'], errors='coerce').fillna(0).astype(int)
else:
    print("\n[!] Warning: 'Readmission' column not found in the dataset.")

print("\n=== CLINICAL TRAJECTORY OPTIMIZATION PIPELINE ===")
print("Select your target outcome variable:")
print("1: DIED")
print("2: Readmission")
target_choice = input("Enter 1 or 2 [Default: 1]: ").strip()
TARGET_OUTCOME = 'Readmission' if target_choice == '2' else 'DIED'
outcome_label = "Readmission" if TARGET_OUTCOME == 'Readmission' else "Death"
print(f"--> Target locked: {TARGET_OUTCOME}")

adm_input = input("Include Admitting Diagnoses in trajectories? (y/n) [Default: y]: ").strip().lower()
INCLUDE_ADM = adm_input in ['y', '', 'yes']

max_len_input = input("Enter maximum sequence length for PREDICTORS (e.g., 2, 3, 4) [Default: 3]: ").strip()
max_seq_length = int(max_len_input) if max_len_input else 3

ccs_cols = [col for col in df.columns if col.startswith('CCS_')]
if not INCLUDE_ADM:
    ccs_cols = [col for col in ccs_cols if not col.startswith('CCS_ADM_DX_')]

for col in ccs_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

floor_support = 5
floor_asym = 0.01
floor_lift = 0.01

drg_input = input("\nEnter DRG_CD to analyze (Press ENTER for ALL): ").strip()
df_cohort = df.copy() if drg_input == "" else df[df['DRG_CD'] == drg_input].copy()

sample_input = input(f"Random sample size? (ENTER for all {len(df_cohort)}): ").strip()
if sample_input:
    df_cohort = df_cohort.sample(n=int(sample_input), random_state=42)
y_train = df_cohort[TARGET_OUTCOME]

# ==========================================
# 2. PHASE 1: PROCEDURAL BUNDLING (JACCARD)
# ==========================================
print("\n--- Phase 1: Procedural Bundling (Jaccard Similarity) ---")
jaccard_input = input("Enter minimum Jaccard Similarity threshold for bundles [Default: 0.75]: ").strip()
jaccard_threshold = float(jaccard_input) if jaccard_input else 0.75

bundle_map = {}
standard_positive_ccs = [c for c in ccs_cols if not c.startswith('CCS_ADM_DX_') and c != TARGET_OUTCOME]

for A, B in permutations(standard_positive_ccs, 2):
    if A >= B: continue 
    count_A, count_B = df_cohort[A].sum(), df_cohort[B].sum()
    count_A_and_B = (df_cohort[A] & df_cohort[B]).sum()
    count_A_or_B = count_A + count_B - count_A_and_B

    if count_A_or_B == 0: continue
    jaccard = count_A_and_B / count_A_or_B

    if jaccard >= jaccard_threshold:
        bundle_name = f"BUNDLE_{A}_{B}"
        df_cohort.loc[:, bundle_name] = (df_cohort[A] & df_cohort[B]).astype(int)
        bundle_map[bundle_name] = (A, B)

print(f"--> Discovered and fused {len(bundle_map)} procedure bundles.")

def get_count(*columns):
    condition = True
    for col in columns: condition = condition & (df_cohort[col] == 1) 
    return df_cohort[condition].shape[0]

def is_path_valid(elements_list):
    elements = set(elements_list)
    if len(elements) != len(elements_list): return False 
    if sum(1 for x in elements if x.startswith('CCS_ADM_DX_')) > 1: return False 
    
    pos_atoms, neg_atoms = set(), set()
    for node in elements:
        if node.startswith('NOT_'): neg_atoms.add(node.replace('NOT_', ''))
        elif node.startswith('BUNDLE_'):
            compA, compB = bundle_map[node]
            if compA in elements or compB in elements: return False
            pos_atoms.update([compA, compB])
        else: pos_atoms.add(node)
            
    if pos_atoms.intersection(neg_atoms): return False
    return True

# ==========================================
# 3. PHASE 2: CONTEXTUAL NEGATIVE SPACE
# ==========================================
print("\n--- Phase 2: Contextual Negative Space Generation ---")
exp_input = input("Enter expected procedure threshold (e.g., 0.50) [Default: 0.50]: ").strip()
exp_thresh = float(exp_input) if exp_input else 0.50

pos_variables = [col for col in df_cohort.columns if (col.startswith('CCS_') or col.startswith('BUNDLE_')) and col != TARGET_OUTCOME]
contextual_negatives_generated = 0

for A, B in permutations(pos_variables, 2):
    if A >= B: continue 
    if not is_path_valid([A, B]): continue
    
    cA, cB, cAB = get_count(A), get_count(B), get_count(A, B)
    if cAB < floor_support: continue
    
    pBA, pAB = cAB/cA if cA > 0 else 0, cAB/cB if cB > 0 else 0
    
    if A.startswith('CCS_ADM_DX_'): cond_prob, target_node = pBA, B
    elif B.startswith('CCS_ADM_DX_'): cond_prob, target_node = pAB, A
    else: cond_prob, target_node = (pBA, B) if pBA > pAB else (pAB, A)
        
    if cond_prob >= exp_thresh:
        neg_target = f"NOT_{target_node}"
        if neg_target not in df_cohort.columns:
            df_cohort[neg_target] = 1 - df_cohort[target_node]
            contextual_negatives_generated += 1

print(f"--> Dynamically generated {contextual_negatives_generated} 'NOT_' features based on contextual expectation.")

# ==========================================
# 4. PHASE 3: MASTER VAULT EXTRACTION
# ==========================================
print(f"\n--- Phase 3: Building Master Vault (Up to {max_seq_length} Predictors + Outcome) ---")
vault_variables = [col for col in df_cohort.columns if (col.startswith('CCS_') or col.startswith('BUNDLE_') or col.startswith('NOT_')) and col != TARGET_OUTCOME] + [TARGET_OUTCOME]
paths_by_length = {}
target_terminated_paths = []

v2 = []   
for A, B in permutations(vault_variables, 2):
    if A >= B: continue 
    if not is_path_valid([A, B]): continue
    
    cA, cB, cAB = get_count(A), get_count(B), get_count(A, B)
    if cAB < floor_support: continue
    
    pBA, pAB = cAB/cA if cA > 0 else 0, cAB/cB if cB > 0 else 0
    
    if A == TARGET_OUTCOME:
        direction, asym = (B, A), abs(pAB - pBA)
    elif B == TARGET_OUTCOME:
        direction, asym = (A, B), abs(pBA - pAB)
    elif A.startswith('CCS_ADM_DX_'):
        direction, asym = (A, B), abs(pBA - pAB)
    elif B.startswith('CCS_ADM_DX_'):
        direction, asym = (B, A), abs(pAB - pBA)
    else:
        direction, asym = ((A, B) if pBA > pAB else (B, A)), abs(pBA - pAB)
        
    if asym < floor_asym: continue
    
    path_dict = {'path': direction, 'support': cAB, 'root_asymmetry': asym, 'markov_lift': None}
    v2.append(path_dict)
    
    # Store if sequence correctly terminates in Outcome
    if direction[-1] == TARGET_OUTCOME:
        target_terminated_paths.append(path_dict)

paths_by_length[2] = v2

# Dynamically expand sequences. Total length = predictor length + 1 outcome.
for current_len in range(3, max_seq_length + 2):
    current_paths = []
    for p in paths_by_length[current_len - 1]:
        base, b_sup, last = p['path'], p['support'], p['path'][-1]
        
        # Stop expanding if path already reached Outcome
        if last == TARGET_OUTCOME: continue 
        
        for C in vault_variables:
            new_path = base + (C,)
            if not is_path_valid(new_path): continue
            if C.startswith('CCS_ADM_DX_'): continue 
            
            cBC = get_count(*new_path)
            if cBC < floor_support: continue
            
            cL, cLC = get_count(last), get_count(last, C)
            pCL, pCB = cLC/cL if cL > 0 else 0, cBC/b_sup if b_sup > 0 else 0
            lift = pCB - pCL
            
            if lift >= floor_lift:
                path_dict = {'path': new_path, 'support': cBC, 'root_asymmetry': p['root_asymmetry'], 'markov_lift': lift}
                current_paths.append(path_dict)
                if C == TARGET_OUTCOME:
                    target_terminated_paths.append(path_dict)
                        
    paths_by_length[current_len] = current_paths

print(f"Master Vault Complete: {len(target_terminated_paths)} valid trajectories terminating in {TARGET_OUTCOME}.")

# ==========================================
# 5. PHASE 4: FEATURE EXTRACTION GRID
# ==========================================
print("\n--- Phase 4: Feature Extraction Thresholds ---")
run_opt = input("Run Optimization Grid Search to find peak AUC? (y/n) [Default: y]: ").strip().lower()

X_raw_full = df_cohort[ccs_cols].fillna(0).astype(int)
X_raw = X_raw_full[[c for c in X_raw_full.columns if X_raw_full[c].sum() >= 5]]

def eval_m(X, y):
    if X.empty: return 0.5
    m = LogisticRegression(max_iter=500, C=0.1, class_weight='balanced', solver='liblinear', random_state=42).fit(X, y)
    return roc_auc_score(y, m.predict_proba(X)[:, 1])

baseline_auc = eval_m(X_raw, y_train)

if run_opt not in ['n', 'no']:
    print("Executing Grid Search (0.05 - 0.50)...")
    grid = np.arange(0.05, 0.51, 0.05)
    results = []

    for ra_t in grid:
        for ml_t in grid:
            f_series = []
            for p in target_terminated_paths:
                ra, ml = p.get('root_asymmetry', 0) or 0, p.get('markov_lift', 0) or 0
                if ra >= ra_t and ml >= ml_t:
                    s = df_cohort[list(p['path'][:-1])].all(axis=1).astype(int)
                    s.name = f"P_{len(f_series)}"
                    f_series.append(s)
            
            auc = eval_m(pd.concat(f_series, axis=1), y_train) if f_series else baseline_auc
            results.append({'Asym': ra_t, 'Lift': ml_t, 'Paths': len(f_series), 'AUC': auc, 'Boost': auc - baseline_auc})

    df_res = pd.DataFrame(results)
    best = df_res.sort_values(by='Boost', ascending=False).iloc[0]
    print(f"\n🏆 OPTIMAL PEAK: Asym {best['Asym']:.2f}, Lift {best['Lift']:.2f} (Boost: {best['Boost']:+.4f})")
    
    confirm = input("Proceed with these optimal parameters? (y/n) or enter custom (Asym,Lift): ").strip().lower()
    if confirm not in ['y', '', 'yes']:
        try: opt_asym, opt_lift = map(float, confirm.split(','))
        except: opt_asym, opt_lift = best['Asym'], best['Lift']
    else: opt_asym, opt_lift = best['Asym'], best['Lift']
else:
    manual_input = input("Enter custom extraction thresholds (Asym,Lift) [Default: 0.10,0.10]: ").strip()
    try: opt_asym, opt_lift = map(float, manual_input.split(',')) if manual_input else (0.10, 0.10)
    except: opt_asym, opt_lift = 0.10, 0.10

# Final Extraction (Strictly paths that hit Outcome)
final_paths = []
final_features = []
for p in target_terminated_paths:
    ra, ml = p.get('root_asymmetry', 0) or 0, p.get('markov_lift', 0) or 0
    if ra >= opt_asym and ml >= opt_lift:
        final_paths.append(p)
        s = df_cohort[list(p['path'][:-1])].all(axis=1).astype(int)
        s.name = f"PATH_{len(final_features)}"
        final_features.append(s)

X_seq = pd.concat(final_features, axis=1) if final_features else pd.DataFrame()

# ==========================================
# 6. PHASE 5: SEQUENCE-SPECIFIC LASSO
# ==========================================
print("\n--- Phase 5: Lasso Regularization on Extracted Trajectories ---")
lasso_input = input("Enter Lasso 'C' parameter (Higher = keeps more features, Lower = stricter) [Default: 0.05]: ").strip()
lasso_c_val = float(lasso_input) if lasso_input else 0.05

def get_lean_matrix(X_matrix, y_target, c_val):
    if X_matrix.empty: return X_matrix
    sel = SelectFromModel(LogisticRegression(penalty='l1', C=c_val, solver='liblinear', random_state=42)).fit(X_matrix, y_target)
    lean_matrix = X_matrix.loc[:, sel.get_support()]
    if lean_matrix.empty:
        sel = SelectFromModel(LogisticRegression(penalty='l1', C=0.5, solver='liblinear', random_state=42)).fit(X_matrix, y_target)
        lean_matrix = X_matrix.loc[:, sel.get_support()]
    return lean_matrix

X_lean_raw = get_lean_matrix(X_raw, y_train, lasso_c_val)
X_lean_seq = get_lean_matrix(X_seq, y_train, lasso_c_val) if not X_seq.empty else pd.DataFrame()

print("\n" + "="*85)
print(f"{'Performance Showdown':<32} | {'Features':<10} | {'AUC-ROC'}")
print("-" * 85)
print(f"{'1. Raw Baseline':<32} | {X_raw.shape[1]:<10} | {eval_m(X_raw, y_train):.4f}")
print(f"{'2. Lean Raw (Lasso)':<32} | {X_lean_raw.shape[1]:<10} | {eval_m(X_lean_raw, y_train):.4f}")
if not X_seq.empty:
    print(f"{'3. Engineered Trajectories':<32} | {X_seq.shape[1]:<10} | {eval_m(X_seq, y_train):.4f}")
    print(f"{'4. Lean Trajectories (Lasso)':<32} | {X_lean_seq.shape[1]:<10} | {eval_m(X_lean_seq, y_train):.4f}")
print("="*85)

surviving_path_indices = [int(col.split('_')[1]) for col in X_lean_seq.columns]
surviving_paths = [final_paths[idx] for idx in surviving_path_indices]

print(f"\nSURVIVING TRAJECTORIES IN LEAN MODEL FOR {TARGET_OUTCOME.upper()}:")
print("-" * 50)
print(f"Total Predictive Sequences Retained: {len(surviving_paths)}")
for p in surviving_paths: print(f" - {' -> '.join(p['path'])}")

# ==========================================
# 7. CLINICAL NARRATIVES
# ==========================================
try:
    mapping_df = pd.read_csv("icd-ccs.csv", dtype=str)
    mapping_df.columns = [col.replace("'", "").strip() for col in mapping_df.columns]
    mapping_df = mapping_df.apply(lambda x: x.str.replace("'", "").str.strip())
    map_dict = {str(ccs).lstrip('0') or '0': str(desc) for ccs, desc in zip(mapping_df['CCS CATEGORY'], mapping_df['CCS CATEGORY DESCRIPTION'])}
except Exception: map_dict = {}

def get_node_desc(node_str):
    if node_str == TARGET_OUTCOME: return outcome_label
    elif node_str.startswith('BUNDLE_'):
        compA, compB = bundle_map[node_str]
        clnA, clnB = compA.replace('CCS_', '').lstrip('0') or '0', compB.replace('CCS_', '').lstrip('0') or '0'
        return f"Clinical Bundle [{map_dict.get(clnA, clnA)} & {map_dict.get(clnB, clnB)}]"
    elif node_str.startswith('NOT_BUNDLE_'):
        clean_node = node_str.replace('NOT_', '')
        compA, compB = bundle_map[clean_node]
        clnA, clnB = compA.replace('CCS_', '').lstrip('0') or '0', compB.replace('CCS_', '').lstrip('0') or '0'
        return f"Skipped Bundle [{map_dict.get(clnA, clnA)} & {map_dict.get(clnB, clnB)}]"
    elif node_str.startswith('NOT_CCS_'):
        cln = node_str.replace('NOT_CCS_', '').lstrip('0') or '0'
        return f"Skipped Procedure: {map_dict.get(cln, cln)}"
    elif node_str.startswith('CCS_ADM_DX_'):
        cln = node_str.replace('CCS_ADM_DX_', '').lstrip('0') or '0'
        return f"Admitting Diagnosis of {map_dict.get(cln, cln)}"
    else:
        cln = node_str.replace('CCS_', '').lstrip('0') or '0'
        return f"Procedure: {map_dict.get(cln, cln)}"

print(f"\n--- TOP 5 CLINICAL NARRATIVES ({outcome_label.upper()}) ---")
target_paths = []
for p in surviving_paths:
    c_first, c_prec = get_count(p['path'][0]), get_count(*p['path'][:-1])
    s_risk = get_count(p['path'][0], TARGET_OUTCOME) / c_first if c_first > 0 else 0
    t_risk = p['support'] / c_prec if c_prec > 0 else 0
    target_paths.append({'p': p['path'], 'j': t_risk - s_risk, 's': s_risk, 't': t_risk, 'n': p['support']})

for i, d in enumerate(sorted(target_paths, key=lambda x: x['j'], reverse=True)[:5], 1):
    path_names = [get_node_desc(step) for step in d['p']]
    narrative = f"Patient presented with an {path_names[0]} " if d['p'][0].startswith('CCS_ADM_DX_') else f"Patient underwent {path_names[0]} "
    narrative += f"(Initial {outcome_label} Risk: {d['s']:.1%}). "
    if len(path_names) > 2: narrative += f"Followed by {path_names[1:-1]}, "
    narrative += f"risk escalated to {d['t']:.1%} (+{d['j']:.1%}) resulting in {outcome_label.lower()}."
    
    print(f"\nRANK #{i} | Risk Escalation: +{d['j']:.1%} | Support: {d['n']} patients")
    print(f"Sequence: {' -> '.join(path_names)}")
    print(f"Narrative: \"{narrative}\"")
    print("-" * 60)
