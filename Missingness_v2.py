import pandas as pd
import numpy as np
import itertools
from openai import OpenAI
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, SimpleImputer
from xgboost import XGBClassifier 
import warnings

# Suppress convergence warnings for clean output
warnings.filterwarnings("ignore")

# ==========================================
# MODULE 1: CORE ENTROPY MATH
# ==========================================

def calc_entropy(target_array):
    p_1 = np.mean(target_array)
    if p_1 == 0 or p_1 == 1:
        return 0.0
    p_0 = 1.0 - p_1
    return - (p_1 * np.log2(p_1) + p_0 * np.log2(p_0))

def missingness_information_gain(df, col_name, target_name):
    target = df[target_name].astype(int)
    base_entropy = calc_entropy(target)
    
    is_missing = df[col_name].isna().astype(int)
    
    target_missing = target[is_missing == 1]
    target_present = target[is_missing == 0]
    
    w_missing = len(target_missing) / len(target)
    w_present = len(target_present) / len(target)
    
    ent_missing = calc_entropy(target_missing) if len(target_missing) > 0 else 0
    ent_present = calc_entropy(target_present) if len(target_present) > 0 else 0
    
    return base_entropy - ((w_missing * ent_missing) + (w_present * ent_present))

def joint_missingness_ig(df, cols, target_name):
    target = df[target_name].astype(int)
    base_entropy = calc_entropy(target)
    
    joint_state = pd.Series([""] * len(df), index=df.index)
    for col in cols:
        is_miss = df[col].isna().astype(int).astype(str)
        joint_state = joint_state + "_" + is_miss
        
    conditional_entropy = 0
    for state in joint_state.unique():
        subset_target = target[joint_state == state]
        weight = len(subset_target) / len(target)
        conditional_entropy += weight * calc_entropy(subset_target)
        
    return base_entropy - conditional_entropy

def baseline_information_gain(df, col_name, target_name):
    df_obs = df[~df[col_name].isna()].copy()
    
    if len(df_obs) == 0:
        return 0.0
        
    target_obs = df_obs[target_name].astype(int)
    base_entropy = calc_entropy(target_obs)
    
    try:
        binned_values = pd.qcut(df_obs[col_name].astype(float), q=3, labels=["Low", "Med", "High"], duplicates='drop')
    except ValueError:
        binned_values = df_obs[col_name].astype(str)
        
    conditional_entropy = 0
    for category in binned_values.unique():
        subset_target = target_obs[binned_values == category]
        weight = len(subset_target) / len(target_obs)
        conditional_entropy += weight * calc_entropy(subset_target)
        
    return base_entropy - conditional_entropy

# ==========================================
# MODULE 2: STRUCTURAL REDUNDANCY SCANNER
# ==========================================

def find_missingness_redundancies(df, feature_cols, target_name, ii_threshold=-0.01, max_interaction_order=2):
    missing_vars = [col for col in feature_cols if df[col].isna().any()]
    redundancy_log = []
    redundant_variables_map = {} 
    
    individual_igs = {col: missingness_information_gain(df, col, target_name) for col in missing_vars}
    
    for order in range(2, max_interaction_order + 1):
        for combo in itertools.combinations(missing_vars, order):
            joint_ig = joint_missingness_ig(df, combo, target_name)
            sum_individual_ig = sum([individual_igs[col] for col in combo])
            ii = joint_ig - sum_individual_ig
            
            if ii < ii_threshold:
                redundancy_log.append({
                    'Order': order,
                    'Combination': " & ".join(combo),
                    'Interaction_Info': round(ii, 4)
                })
                
                for var in combo:
                    if var not in redundant_variables_map:
                        redundant_variables_map[var] = {
                            'Detected_Order': order,
                            'Trigger_Combo': " & ".join([c for c in combo if c != var]) 
                        }
                
    if redundancy_log:
        print(f"--- REDUNDANT BLOCKS DETECTED (Up to Order {max_interaction_order}) ---")
        for log in redundancy_log:
            print(f"Order {log['Order']} | Combo: {log['Combination']} | II: {log['Interaction_Info']}")
            
    return redundant_variables_map

# ==========================================
# MODULE 3: RELATIVE RISK MAPPING
# ==========================================

def map_relative_risk(df, col_name, target_name):
    target = df[target_name].astype(int)
    is_missing = df[col_name].isna()
    
    if is_missing.sum() == 0:
        return {"Zone": "No Missing Data", "Explanation": "N/A", "P_Missing": None}
    
    p_missing = target[is_missing].mean()
    df_obs = df[~is_missing].copy()
    
    if len(df_obs) == 0:
        return {"Zone": "100% Missing", "Explanation": "N/A", "P_Missing": p_missing}
        
    try:
        binned = pd.qcut(df_obs[col_name].astype(float), q=3, labels=["Low", "Med", "High"], duplicates='drop')
    except ValueError:
        return {"Zone": "Unmappable", "Explanation": "Insufficient variance to build continuum.", "P_Missing": p_missing}
    
    if len(binned.unique()) < 3:
        return {"Zone": "Unmappable", "Explanation": "Not enough distinct bins to form a continuum.", "P_Missing": p_missing}

    p_low = target[df_obs.index[binned == "Low"]].mean()
    p_med = target[df_obs.index[binned == "Med"]].mean()
    p_high = target[df_obs.index[binned == "High"]].mean()
    
    is_increasing = (p_low <= p_med) and (p_med <= p_high)
    is_decreasing = (p_low >= p_med) and (p_med >= p_high)
    
    if not (is_increasing or is_decreasing):
        return {
            "Zone": "Non-Monotonic",
            "Explanation": f"Baseline risk is U-shaped (Low:{p_low:.2f}, Med:{p_med:.2f}, High:{p_high:.2f}). Mapping aborted to prevent hallucination.",
            "P_Missing": p_missing
        }
        
    p_min = min(p_low, p_high)
    p_max = max(p_low, p_high)

    if p_missing > p_max:
        zone = "Extrapolated (Extreme Risk)"
        exp = f"Missing risk ({p_missing:.2f}) exceeds highest baseline risk ({p_max:.2f}). Indicates a severe phenotype."
    elif p_missing < p_min:
        zone = "Extrapolated (Benign)"
        exp = f"Missing risk ({p_missing:.2f}) is lower than safest baseline risk ({p_min:.2f}). Indicates a highly stable phenotype."
    else:
        zone = "Interpolated (Proxy)"
        exp = f"Missing risk ({p_missing:.2f}) falls within baseline boundaries ({p_min:.2f} to {p_max:.2f}). Acts as a proxy."
        
    return {"Zone": zone, "Explanation": exp, "P_Missing": round(p_missing, 4)}

# ==========================================
# MODULE 4: THE MASTER DIAGNOSTIC PIPELINE
# ==========================================

def run_robust_diagnostics(df, feature_cols, target_col, signal_threshold=0.01, max_interaction_order=2, ii_threshold=-0.01):
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    print(f"Executing Stage 2: Structural Scan (Max Order: {max_interaction_order}, II Threshold: {ii_threshold})...")
    known_redundancies = find_missingness_redundancies(
        df, feature_cols, target_col, ii_threshold=ii_threshold, max_interaction_order=max_interaction_order
    )
    
    print("\nExecuting Stage 3: Explanatory Engine (Relative Risk Mapping)...")
    results = []
    
    for col in feature_cols:
        baseline_ig = baseline_information_gain(df, col, target_col)
        missing_ig = missingness_information_gain(df, col, target_col)
        
        diag_zone = "N/A"
        llm_explanation = "N/A"
        detection_level = "Single Variable" 
        linked_to = "N/A" 
        
        if missing_ig > signal_threshold:
            if baseline_ig > signal_threshold:
                classification = "Predictive Missingness (Informative)"
                action = "DO NOT IMPUTE. Valid baseline with high missingness signal."
                
                engine_output = map_relative_risk(df, col, target_col)
                diag_zone = engine_output['Zone']
                llm_explanation = engine_output['Explanation']
                    
            else:
                if col in known_redundancies:
                    classification = "Structural Block (Redundant)"
                    action = "Create missingness feature. Confirmed structural redundancy."
                    
                    order_num = known_redundancies[col]['Detected_Order']
                    if order_num == 2:
                        detection_level = "Order 2 (Pair)"
                    elif order_num == 3:
                        detection_level = "Order 3 (Triplet)"
                    else:
                        detection_level = f"Order {order_num}"
                        
                    linked_to = known_redundancies[col]['Trigger_Combo']
                else:
                    classification = "Isolated Predictive Missingness"
                    action = "DO NOT IMPUTE. Baseline truncated. Isolated predictive missingness."
                    
        else:
            if baseline_ig > signal_threshold:
                classification = "Non-Informative Missingness"
                action = "Safe to impute. Missingness is random, base values are strong."
            else:
                classification = "Pure Noise"
                action = "Drop variable. No clinical or structural utility."
                
        results.append({
            'Variable': col,
            'Detection_Level': detection_level,  
            'Linked_To': linked_to,              
            'Baseline_IG': round(baseline_ig, 4),
            'Missing_IG': round(missing_ig, 4),
            'Classification': classification,
            'Diagnostic_Zone': diag_zone,
            'LLM_Explanation': llm_explanation,
            'Recommended_Action': action
        })
        
    return pd.DataFrame(results)

# ==========================================
# MODULE 5: LOCAL LLM INTEGRATION 
# ==========================================

client = OpenAI(base_url="http://localhost:1234/v1", api_key="local-llm")

def generate_llm_summary(row):
    if row['Classification'] == "Pure Noise":
        return "Variable provides no predictive value. Recommend dropping."

    try:
        model_list = client.models.list()
        active_model = model_list.data[0].id if model_list.data else "local-model"
    except Exception:
        active_model = "local-model"

    instructions = (
        "Instruction: You are an expert clinical informatician and data scientist. "
        "Based on the mathematical analysis provided below, write a concise, 2-sentence "
        "summary explaining how to handle this variable and why. "
        "Do not use complex mathematical jargon; explain it strictly in clinical or workflow terms.\n\n"
        "Data Provided:\n"
        f"- Variable Name: {row['Variable']}\n"
        f"- Classification: {row['Classification']}\n"
        f"- Action Required: {row['Recommended_Action']}\n"
        f"- Pipeline Explanation: {row['LLM_Explanation']}\n"
        f"- Redundancy Link (if any): {row['Linked_To']}"
    )
    
    try:
        response = client.chat.completions.create(
            model=active_model,
            messages=[{"role": "user", "content": instructions}],
            temperature=0.2, 
            max_tokens=150
        )
        
        content = response.choices[0].message.content
        if not content or content.strip() == "":
            return f"[Empty generation from '{active_model}'. Ensure UI preset matches model architecture.]"
            
        return content.strip()
        
    except Exception as e:
        return f"LLM Connection Error: ({e})"

# ==========================================
# MODULE 6: STRICT NO-LEAKAGE EVALUATION
# ==========================================

def prepare_super_naive_strict(df_train, df_test, target_col):
    """Dataset A: Super Naive Baseline. Mean for continuous, Mode for categorical. Fit on train only."""
    y_train = df_train[target_col]
    X_train = df_train.drop(columns=[target_col])
    
    y_test = df_test[target_col]
    X_test = df_test.drop(columns=[target_col])
    
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
        
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
        
    return X_train, y_train, X_test, y_test

def prepare_advanced_baseline_strict(df_train, df_test, target_col):
    """Dataset B: MICE Imputation Only. Fit on train, transform on test."""
    y_train = df_train[target_col]
    X_train = df_train.drop(columns=[target_col])
    
    y_test = df_test[target_col]
    X_test = df_test.drop(columns=[target_col])
    
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        mice_imputer = IterativeImputer(max_iter=10, random_state=42)
        X_train[numeric_cols] = mice_imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = mice_imputer.transform(X_test[numeric_cols])
        
    return X_train, y_train, X_test, y_test

def prepare_precision_engineered_strict(df_train, df_test, target_col, diagnostic_report):
    """Dataset C: Entropy Features + MICE. Driven entirely by Training data insights."""
    y_train = df_train[target_col]
    X_train = df_train.drop(columns=[target_col])
    
    y_test = df_test[target_col]
    X_test = df_test.drop(columns=[target_col])
    
    columns_to_drop = []
    
    for _, row in diagnostic_report.iterrows():
        var_name = row['Variable']
        classification = row['Classification']
        
        if var_name not in X_train.columns:
            continue
            
        if classification == "Pure Noise":
            columns_to_drop.append(var_name)
            
        elif classification in ["Isolated Predictive Missingness", "Predictive Missingness (Informative)", "Structural Block (Redundant)"]:
            X_train[f"{var_name}_missing"] = X_train[var_name].isna().astype(int)
            X_test[f"{var_name}_missing"] = X_test[var_name].isna().astype(int)

    X_train = X_train.drop(columns=columns_to_drop)
    X_test = X_test.drop(columns=columns_to_drop)
    
    numeric_cols = [c for c in X_train.columns if not c.endswith('_missing')]
    if len(numeric_cols) > 0:
        mice_imputer = IterativeImputer(max_iter=10, random_state=42)
        X_train[numeric_cols] = mice_imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = mice_imputer.transform(X_test[numeric_cols])
        
    return X_train, y_train, X_test, y_test

def evaluate_methodology_strict(X_train, y_train, X_test, y_test, methodology_name):
    """Runs Grid-Searched XGBoost on pre-split, leak-free data."""
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0]
    }
    
    base_model = XGBClassifier(
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), 
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    grid_search = GridSearchCV(
        estimator=base_model, param_grid=param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba) 
    report = classification_report(y_test, y_pred, output_dict=True)
    sensitivity = report['1']['recall']
    
    print(f"\n{'='*40}")
    print(f" {methodology_name} (XGBoost)")
    print(f"{'='*40}")
    print(f"Features utilized: {X_train.shape[1]}")
    print(f"ROC-AUC Score:     {roc_auc:.4f}")
    print(f"Brier Score:       {brier:.4f}")
    print(f"Sensitivity:       {sensitivity:.4f}")
    
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(5)
    
    print("\nTop 5 Influential Features:")
    for _, row in importances.iterrows():
        print(f" - {row['Feature']}: {row['Importance']:.4f}")
        
    return roc_auc, brier, sensitivity

# ==========================================
# TEST EXECUTION 
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" NEURO-SYMBOLIC MISSINGNESS ENGINE ")
    print("="*50 + "\n")
    
    file_input = input("Enter dataset filename [default: miss.csv]: ").strip()
    target_file = file_input if file_input else "miss.csv"
    
    len_input = input("Enter Max Interaction Order (e.g., 2 for pairs) [default: 3]: ").strip()
    max_order = int(len_input) if len_input else 3
    
    ii_input = input("Enter Interaction Information (II) threshold [default: -0.01]: ").strip()
    ii_thresh = float(ii_input) if ii_input else -0.01
    
    sig_input = input("Enter Signal threshold for Missingness IG [default: 0.01]: ").strip()
    sig_thresh = float(sig_input) if sig_input else 0.01
    
    print(f"\n[INITIALIZING SCAN] File: {target_file} | Max Order: {max_order} | II: {ii_thresh} | Sig: {sig_thresh}\n")

    try:
        df = pd.read_csv(target_file)
        target_col = 'outcome' 
        
        if 'patient_id' in df.columns:
            df = df.set_index('patient_id')
            
        features_to_test = [col for col in df.columns if col != target_col]
        
        # Binarize Target Guardrail
        df = df.dropna(subset=[target_col]).copy()
        unique_outcomes = df[target_col].dropna().unique()
        if len(unique_outcomes) > 2:
            median_val = df[target_col].median()
            print(f"Notice: Auto-binarizing '{target_col}' at the median...")
            df[target_col] = (df[target_col] > median_val).astype(int)
            
        # =======================================================
        # CRITICAL FIX: SPLIT DATA BEFORE ANY MATH OCCURS
        # =======================================================
        print("\n[STAGE 1] Securing Test Set (Preventing Data Leakage)...")
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target_col])
        print(f"Training Cohort: {len(df_train)} | Holdout Cohort: {len(df_test)}")
                
        # RUN DIAGNOSTICS ONLY ON TRAINING DATA
        final_report_df = run_robust_diagnostics(
            df_train, 
            features_to_test, 
            target_col, 
            signal_threshold=sig_thresh, 
            max_interaction_order=max_order,
            ii_threshold=ii_thresh
        )

        print("\n=== FINAL DIAGNOSTIC REPORT (TRAINING DATA ONLY) ===")
        print(final_report_df.to_string(index=False))

        print("\nExecuting Stage 4: Connecting to Local LLM for Clinical Summaries...")
        final_report_df['Final_Clinical_Summary'] = final_report_df.apply(generate_llm_summary, axis=1)
        
        print("\n=== FINAL AI-ENHANCED DIAGNOSTIC REPORT ===")
        for index, row in final_report_df.iterrows():
            print(f"\nFeature: {row['Variable']}")
            print(f"Classification: {row['Classification']} | Linked To: {row['Linked_To']}")
            print(f"Summary: {row['Final_Clinical_Summary']}")
            
        print("\n" + "="*50)
        print(" STAGE 5: STRICT OUT-OF-SAMPLE EVALUATION ")
        print("="*50)
        print("Running head-to-head XGBoost models on unseen test data...\n")
        
        # 1. The Absolute Floor
        X_train_naive, y_train_naive, X_test_naive, y_test_naive = prepare_super_naive_strict(df_train, df_test, target_col)
        auc_naive, brier_naive, sens_naive = evaluate_methodology_strict(X_train_naive, y_train_naive, X_test_naive, y_test_naive, "Dataset A: Super Naive (Mean/Mode)")

        # 2. The Traditional Gold Standard
        X_train_base, y_train_base, X_test_base, y_test_base = prepare_advanced_baseline_strict(df_train, df_test, target_col)
        auc_mice, brier_mice, sens_mice = evaluate_methodology_strict(X_train_base, y_train_base, X_test_base, y_test_base, "Dataset B: MICE Baseline")
        
        # 3. The Informatics Pipeline
        X_train_eng, y_train_eng, X_test_eng, y_test_eng = prepare_precision_engineered_strict(df_train, df_test, target_col, final_report_df)
        auc_eng, brier_eng, sens_eng = evaluate_methodology_strict(X_train_eng, y_train_eng, X_test_eng, y_test_eng, "Dataset C: Entropy Engineered + MICE")
        
        print("\n" + "="*50)
        print(" FINAL COMPARISON SUMMARY ")
        print("="*50)
        
        print("--- ROC-AUC Progression ---")
        print(f"Super Naive: {auc_naive:.4f}")
        print(f"MICE:        {auc_mice:.4f}")
        print(f"Engineered:  {auc_eng:.4f}")
        
        print("\n--- Mortality Sensitivity Progression ---")
        print(f"Super Naive: {sens_naive:.4f}")
        print(f"MICE:        {sens_mice:.4f}")
        print(f"Engineered:  {sens_eng:.4f}")
        print("="*50)
        
    except FileNotFoundError:
        print(f"Error: '{target_file}' not found.")