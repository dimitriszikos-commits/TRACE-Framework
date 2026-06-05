import pandas as pd
import numpy as np
import itertools

# ==========================================
# MODULE 1: CORE ENTROPY MATH
# ==========================================

def calc_entropy(target_array):
    """Calculates Shannon Entropy for a binary array."""
    p_1 = np.mean(target_array)
    if p_1 == 0 or p_1 == 1:
        return 0.0
    p_0 = 1.0 - p_1
    return - (p_1 * np.log2(p_1) + p_0 * np.log2(p_0))

def missingness_information_gain(df, col_name, target_name):
    """Calculates Information Gain of a single column's missingness pattern."""
    target = df[target_name].astype(int)
    base_entropy = calc_entropy(target)
    
    # Identify missingness (robust to both empty strings and NaNs)
    is_missing = df[col_name].isin(["", np.nan]).astype(int)
    
    target_missing = target[is_missing == 1]
    target_present = target[is_missing == 0]
    
    w_missing = len(target_missing) / len(target)
    w_present = len(target_present) / len(target)
    
    ent_missing = calc_entropy(target_missing) if len(target_missing) > 0 else 0
    ent_present = calc_entropy(target_present) if len(target_present) > 0 else 0
    
    return base_entropy - ((w_missing * ent_missing) + (w_present * ent_present))

def joint_missingness_ig(df, cols, target_name):
    """Calculates the Joint Information Gain of N missingness indicators."""
    target = df[target_name].astype(int)
    base_entropy = calc_entropy(target)
    
    # Dynamically create joint states for any number of columns
    joint_state = pd.Series([""] * len(df), index=df.index)
    for col in cols:
        is_miss = df[col].isin(["", np.nan]).astype(int).astype(str)
        joint_state = joint_state + "_" + is_miss
        
    conditional_entropy = 0
    for state in joint_state.unique():
        subset_target = target[joint_state == state]
        weight = len(subset_target) / len(target)
        conditional_entropy += weight * calc_entropy(subset_target)
        
    return base_entropy - conditional_entropy

def baseline_information_gain(df, col_name, target_name):
    """Calculates Information Gain of the OBSERVED values."""
    # Filter only rows where data is present
    df_obs = df[~df[col_name].isin(["", np.nan])].copy()
    
    if len(df_obs) == 0:
        return 0.0
        
    target_obs = df_obs[target_name].astype(int)
    base_entropy = calc_entropy(target_obs)
    
    # Bin continuous values to calculate entropy natively
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
    """
    Scans combinations of variables up to max_interaction_order to find 
    redundant missingness patterns (Workflow Artifacts).
    """
    # Filter to variables that actually contain missing data
    missing_vars = [col for col in feature_cols if df[col].isin(["", np.nan]).any()]
    
    redundancy_log = []
    redundant_variables = set()
    
    # Pre-calculate individual IGs to avoid redundant compute in the loops
    individual_igs = {col: missingness_information_gain(df, col, target_name) for col in missing_vars}
    
    # Iterate from pairs (order 2) up to the specified max_interaction_order
    for order in range(2, max_interaction_order + 1):
        for combo in itertools.combinations(missing_vars, order):
            
            # Calculate joint IG for the N-tuple
            joint_ig = joint_missingness_ig(df, combo, target_name)
            
            # Sum of individual IGs for the current combination
            sum_individual_ig = sum([individual_igs[col] for col in combo])
            
            # Calculate Interaction Information (Synergy vs Redundancy)
            ii = joint_ig - sum_individual_ig
            
            # Negative II indicates redundancy
            if ii < ii_threshold:
                redundant_variables.update(combo)
                redundancy_log.append({
                    'Order': order,
                    'Combination': " & ".join(combo),
                    'Interaction_Info': round(ii, 4)
                })
                
    if redundancy_log:
        print(f"--- REDUNDANT BLOCKS DETECTED (Up to Order {max_interaction_order}) ---")
        for log in redundancy_log:
            print(f"Order {log['Order']} | Combo: {log['Combination']} | II: {log['Interaction_Info']}")
            
    return redundant_variables

# ==========================================
# MODULE 3: THE MASTER DIAGNOSTIC PIPELINE
# ==========================================

def run_robust_diagnostics(df, feature_cols, target_col, signal_threshold=0.01, max_interaction_order=2):
    """
    Executes the full diagnostic matrix across all specified features,
    using the redundancy scan as a tie-breaker for heavily truncated signals.
    """
    print(f"Executing Stage 2: Structural Redundancy Scan (Max Order: {max_interaction_order})...")
    known_redundancies = find_missingness_redundancies(
        df, feature_cols, target_col, max_interaction_order=max_interaction_order
    )
    
    print("\nExecuting Stage 3: Building the Baseline-to-Missingness Matrix...")
    results = []
    
    for col in feature_cols:
        baseline_ig = baseline_information_gain(df, col, target_col)
        missing_ig = missingness_information_gain(df, col, target_col)
        
        # Matrix Classification Logic - Updated Terminology
        if missing_ig > signal_threshold:
            if baseline_ig > signal_threshold:
                classification = "Predictive Missingness (Informative)"
                action = "DO NOT IMPUTE. Valid baseline with high missingness signal."
            else:
                # Truncation check: Admin block vs. Isolated erasure
                if col in known_redundancies:
                    classification = "Structural Block (Redundant)"
                    action = "Retain pattern proxy. Confirmed structural redundancy."
                else:
                    classification = "Isolated Predictive Missingness"
                    action = "DO NOT IMPUTE. Baseline truncated. Isolated predictive missingness."
                    
        else: # Missing_ig <= signal_threshold
            if baseline_ig > signal_threshold:
                classification = "Non-Informative Missingness"
                action = "Safe to impute. Missingness is random, base values are strong."
            else:
                classification = "Pure Noise"
                action = "Drop variable. No clinical or structural utility."
                
        results.append({
            'Variable': col,
            'Baseline_IG': round(baseline_ig, 4),
            'Missing_IG': round(missing_ig, 4),
            'Classification': classification,
            'Recommended_Action': action
        })
        
    return pd.DataFrame(results)

# ==========================================
# TEST EXECUTION 
# ==========================================
if __name__ == "__main__":
    print("Generating test data...")
    # Quickly rebuild the 100-row test dataset to feed the pipeline
    np.random.seed(42)
    n_records = 100
    true_var1 = np.random.normal(loc=50, scale=15, size=n_records)
    admin_routing = np.random.choice([0, 1], size=n_records, p=[0.6, 0.4])

    df = pd.DataFrame({
        'var1': true_var1.copy(),
        'var2': np.random.normal(100, 10, n_records),
        'var3': np.random.normal(5, 1, n_records),
        'var4': np.random.normal(20, 5, n_records),
        'outcome': np.random.binomial(1, 1 / (1 + np.exp(-((true_var1 * 0.05) + (admin_routing * 1.2) - 3.5))))
    })

    # Prepare dtypes and inject missingness
    cols_to_modify = ['var1', 'var2', 'var3', 'var4']
    df[cols_to_modify] = df[cols_to_modify].astype(object)
    df.loc[df['var1'] > 65, 'var1'] = ""
    df.loc[admin_routing == 1, ['var2', 'var3']] = ""
    df.loc[np.random.rand(n_records) < 0.15, 'var4'] = ""

    # Define features
    features_to_test = ['var1', 'var2', 'var3', 'var4']
    
    # --- INTERACTIVE PROMPT ---
    while True:
        try:
            user_input = input(f"\nEnter max interaction order (min 2, max {len(features_to_test)}): ")
            max_order = int(user_input)
            if max_order < 2:
                print("Order must be at least 2 to check for interactions. Try again.")
            elif max_order > len(features_to_test):
                print(f"Cannot check an order higher than the number of variables ({len(features_to_test)}). Try again.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter an integer.")
            
    # Execute the final pipeline with the dynamic parameter
    final_report_df = run_robust_diagnostics(
        df, 
        features_to_test, 
        'outcome', 
        signal_threshold=0.025, 
        max_interaction_order=max_order
    )

    print("\n=== FINAL DIAGNOSTIC REPORT ===")
    print(final_report_df.to_string(index=False))