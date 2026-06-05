import pandas as pd
import sys
import numpy as np
import time
from itertools import permutations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, log_loss
from sklearn.feature_selection import SelectFromModel

# ==========================================
# 1. DATA INGESTION & TARGET SELECTION
# ==========================================
input_file = "2024_SEED_CCS.csv" 
print(f"Loading dataset: {input_file}...")
df = pd.read_csv(input_file, dtype=str)

ccs_cols = [col for col in df.columns if col.startswith('CCS_')]
for col in ccs_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

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

# Floor thresholds to capture the "Master Vault"
floor_support = 5
floor_asym = 0.01
floor_lift = 0.01

# ==========================================
# 2. COHORT ISOLATION
# ==========================================
drg_input = input("\nEnter DRG_CD to analyze (Press ENTER for ALL): ").strip()
df_cohort = df if drg_input == "" else df[df['DRG_CD'] == drg_input]
total_n = len(df_cohort)

sample_input = input(f"Random sample size? (ENTER for all {total_n}): ").strip()
if sample_input:
    df_cohort = df_cohort.sample(n=int(sample_input), random_state=42)

y_train = df_cohort[TARGET_OUTCOME]
variables = [col for col in df_cohort.columns if col.startswith('CCS_') or col == TARGET_OUTCOME]

def get_count(*columns):
    condition = True
    for col in columns:
        condition = condition & (df_cohort[col] == 1) 
    return df_cohort[condition].shape[0]

# ==========================================
# 3. MASTER VAULT EXTRACTION (WITH CHRONOLOGICAL LOCK)
# ==========================================
print("\n--- Phase 1: Building Master Vault (Enforcing Chronology) ---")
paths_by_length = {}

v2 = []   
for A, B in permutations(variables, 2):
    if A > B: continue 
    
    is_adm_A = A.startswith('CCS_ADM_DX_')
    is_adm_B = B.startswith('CCS_ADM_DX_')
    
    # Rule 1: Two admitting diagnoses happen simultaneously. They cannot form a sequence.
    if is_adm_A and is_adm_B: continue
    
    cA, cB, cAB = get_count(A), get_count(B), get_count(A, B)
    if cAB < floor_support: continue
    
    pBA = cAB/cA if cA > 0 else 0
    pAB = cAB/cB if cB > 0 else 0
    asym = abs(pBA - pAB)
    
    if asym < floor_asym: continue
    
    # Rule 2: If one of the events is an Admitting DX, chronologically it MUST be the starting point
    if is_adm_A:
        direction = (A, B)
    elif is_adm_B:
        direction = (B, A)
    else:
        # Standard logic for Procedure -> Procedure sequences
        direction = (A, B) if pBA > pAB else (B, A)
        
    # Rule 3: The Target Outcome can NEVER be the starting point
    if direction[0] == TARGET_OUTCOME: continue
    
    v2.append({'path': direction, 'support': cAB, 'root_asymmetry': asym, 'markov_lift': None})

paths_by_length[2] = v2

c3 = []
for p in paths_by_length[2]:
    base, b_sup, last = p['path'], p['support'], p['path'][-1]
    
    if last == TARGET_OUTCOME: continue
    
    for C in [v for v in variables if v not in base]:
        # Rule 4: An Admitting DX can NEVER be added to the middle/end of a sequence
        if C.startswith('CCS_ADM_DX_'): continue
        
        cBC = get_count(*base, C)
        if cBC < floor_support: continue
        
        cL, cLC = get_count(last), get_count(last, C)
        pCL = cLC/cL if cL > 0 else 0
        pCB = cBC/b_sup if b_sup > 0 else 0
        lift = pCB - pCL
        
        if lift >= floor_lift:
            c3.append({'path': base+(C,), 'support': cBC, 'root_asymmetry': p['root_asymmetry'], 'markov_lift': lift})

paths_by_length[3] = c3
print(f"Master Vault Complete: {sum(len(v) for v in paths_by_length.values())} chronological paths stored for {TARGET_OUTCOME}.")

# ==========================================
# 4. HIGH-RES GRID SEARCH
# ==========================================
print("\n--- Phase 2: High-Resolution Optimization (0.01 - 0.51) ---")
X_raw_full = df_cohort[ccs_cols].fillna(0).astype(int)
X_raw = X_raw_full[[c for c in X_raw_full.columns if X_raw_full[c].sum() >= 5]]

def eval_m(X, y):
    if X.empty: return 0.5
    m = LogisticRegression(max_iter=500, C=0.1, class_weight='balanced', solver='liblinear').fit(X, y)
    return roc_auc_score(y, m.predict_proba(X)[:, 1])

baseline_auc = eval_m(X_raw, y_train)
grid = np.arange(0.01, 0.52, 0.02)
results = []

for ra_t in grid:
    for ml_t in grid:
        f_series = []
        for length, paths in paths_by_length.items():
            for p in paths:
                ra, ml = p.get('root_asymmetry', 0) or 0, p.get('markov_lift', 0) or 0
                if ra >= ra_t and ml >= ml_t:
                    s = df_cohort[[c for c in p['path'] if c != TARGET_OUTCOME]].all(axis=1).astype(int)
                    s.name = f"P_{len(f_series)}"
                    f_series.append(s)
        
        if f_series:
            X_hyb_loop = pd.concat([X_raw, pd.concat(f_series, axis=1)], axis=1)
            auc = eval_m(X_hyb_loop, y_train)
        else:
            auc = baseline_auc
        results.append({'Asym': ra_t, 'Lift': ml_t, 'Paths': len(f_series), 'AUC': auc, 'Boost': auc - baseline_auc})

df_res = pd.DataFrame(results)

asym_impact = df_res.groupby('Asym')['Boost'].mean().std()
lift_impact = df_res.groupby('Lift')['Boost'].mean().std()
driver = "Root Asymmetry (Confidence)" if asym_impact > lift_impact else "Markov Lift Delta"
print(f"\n💡 DRIVER ANALYSIS: Predictability for {TARGET_OUTCOME} is more sensitive to {driver}.")

best = df_res.sort_values(by='Boost', ascending=False).iloc[0]
print(f"🏆 OPTIMAL PEAK: Asym {best['Asym']:.2f}, Lift {best['Lift']:.2f} (Boost: {best['Boost']:+.4f})")

# ==========================================
# 5. FINAL EVALUATION & ABLATION STUDY
# ==========================================
opt_asym, opt_lift = best['Asym'], best['Lift']
confirm = input("\nProceed with these optimal parameters? (y/n) or enter custom (Asym,Lift): ").strip().lower()
if confirm not in ['y', '', 'yes']:
    try:
        opt_asym, opt_lift = map(float, confirm.split(','))
    except: pass

final_paths = []
final_features = []
for length, paths in paths_by_length.items():
    for p in paths:
        ra, ml = p.get('root_asymmetry', 0) or 0, p.get('markov_lift', 0) or 0
        if ra >= opt_asym and ml >= opt_lift:
            final_paths.append(p)
            s = df_cohort[[c for c in p['path'] if c != TARGET_OUTCOME]].all(axis=1).astype(int)
            s.name = f"PATH_{len(final_features)}"
            final_features.append(s)

X_seq = pd.concat(final_features, axis=1) if final_features else pd.DataFrame()
X_hyb = pd.concat([X_raw, X_seq], axis=1) if not X_seq.empty else X_raw

def get_lean_matrix(X_matrix, y_target):
    if X_matrix.empty: return X_matrix
    sel = SelectFromModel(LogisticRegression(penalty='l1', C=0.05, solver='liblinear')).fit(X_matrix, y_target)
    lean_matrix = X_matrix.loc[:, sel.get_support()]
    if lean_matrix.empty:
        sel = SelectFromModel(LogisticRegression(penalty='l1', C=0.2, solver='liblinear')).fit(X_matrix, y_target)
        lean_matrix = X_matrix.loc[:, sel.get_support()]
    return lean_matrix

print("\nExecuting Comprehensive Lasso Experiments...")
X_lean_raw = get_lean_matrix(X_raw, y_train)
X_lean_seq = get_lean_matrix(X_seq, y_train) if not X_seq.empty else pd.DataFrame()
X_lean_hyb = get_lean_matrix(X_hyb, y_train)

print("\n" + "="*85)
print(f"{'Performance Showdown':<32} | {'Features':<10} | {'AUC-ROC'}")
print("-" * 85)
print(f"{'1. Raw CCS Baseline':<32} | {X_raw.shape[1]:<10} | {baseline_auc:.4f}")
print(f"{'2. Lean Raw Data (Lasso)':<32} | {X_lean_raw.shape[1]:<10} | {eval_m(X_lean_raw, y_train):.4f}")
if not X_seq.empty:
    print(f"{'3. Engineered Sequences Only':<32} | {X_seq.shape[1]:<10} | {eval_m(X_seq, y_train):.4f}")
    print(f"{'4. Lean Sequences (Lasso)':<32} | {X_lean_seq.shape[1]:<10} | {eval_m(X_lean_seq, y_train):.4f}")
print(f"{'5. Full Hybrid Model':<32} | {X_hyb.shape[1]:<10} | {eval_m(X_hyb, y_train):.4f}")
print(f"{'6. Lean Hybrid (Lasso)':<32} | {X_lean_hyb.shape[1]:<10} | {eval_m(X_lean_hyb, y_train):.4f}")
print("="*85)

# --- LASSO HERO FEATURE OUTPUT ---
print(f"\nSURVIVING FEATURES IN LEAN HYBRID MODEL FOR {TARGET_OUTCOME.upper()}:")
print("-" * 50)
surviving_paths = [f for f in X_lean_hyb.columns if f.startswith("PATH_")]
surviving_raw = [f for f in X_lean_hyb.columns if not f.startswith("PATH_")]

print(f"Total Features: {X_lean_hyb.shape[1]} ({len(surviving_paths)} Sequences, {len(surviving_raw)} Raw Codes)")
print("\nSurviving Sequences:")
for p_feat in surviving_paths:
    idx = int(p_feat.split('_')[1])
    path_str = " -> ".join(final_paths[idx]['path'])
    print(f" - {p_feat}: {path_str}")

print("\nSurviving Raw CCS Codes:")
print(", ".join(surviving_raw) if surviving_raw else "None")
print("-" * 50)

# ==========================================
# 6. TEXT-BASED CLINICAL NARRATIVES
# ==========================================
try:
    mapping_df = pd.read_csv("icd-ccs.csv", dtype=str)
    mapping_df.columns = [col.replace("'", "").strip() for col in mapping_df.columns]
    mapping_df = mapping_df.apply(lambda x: x.str.replace("'", "").str.strip())
    map_dict = {str(ccs).lstrip('0') or '0': str(desc) for ccs, desc in zip(mapping_df['CCS CATEGORY'], mapping_df['CCS CATEGORY DESCRIPTION'])}
except Exception:
    map_dict = {}

print(f"\n--- TOP 5 CLINICAL NARRATIVES ({outcome_label.upper()} ESCALATION) ---")
target_paths = []
for p in final_paths:
    if p['path'][-1] == TARGET_OUTCOME:
        c_first = get_count(p['path'][0])
        s_risk = get_count(p['path'][0], TARGET_OUTCOME) / c_first if c_first > 0 else 0
        c_prec = get_count(*p['path'][:-1])
        t_risk = p['support'] / c_prec if c_prec > 0 else 0
        target_paths.append({'p': p['path'], 'j': t_risk - s_risk, 's': s_risk, 't': t_risk, 'n': p['support']})

for i, d in enumerate(sorted(target_paths, key=lambda x: x['j'], reverse=True)[:5], 1):
    path_names = []
    for step in d['p']:
        if step == TARGET_OUTCOME:
            path_names.append(outcome_label)
        elif step.startswith('CCS_ADM_DX_'):
            cln = step.replace('CCS_ADM_DX_', '').lstrip('0') or '0'
            path_names.append(f"Admitting Diagnosis of {map_dict.get(cln, cln)}")
        else:
            cln = step.replace('CCS_', '').lstrip('0') or '0'
            path_names.append(f"Procedure: {map_dict.get(cln, cln)}")
            
    narrative = f"A patient presented with {path_names[0]} (Initial {outcome_label} Risk: {d['s']:.1%}). "
    if len(path_names) > 2:
        narrative += f"This was followed by {path_names[1]}, "
    narrative += f"at which point the risk escalated to {d['t']:.1%} (+{d['j']:.1%}) resulting in {outcome_label.lower()}."
    
    print(f"\nRANK #{i} | Risk Escalation: +{d['j']:.1%} | Support: {d['n']} patients")
    print(f"Sequence: {' -> '.join(path_names)}")
    print(f"Narrative: \"{narrative}\"")
    print("-" * 60)
