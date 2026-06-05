import pandas as pd
import sys
from itertools import permutations
import json

# ==========================================
# 1. LOAD DATA & FIX DATA TYPES
# ==========================================
input_file = "2024_SEED_CCS.csv"
print(f"Loading dataset: {input_file}...")

# Load treating everything as a string to preserve leading zeros in DRG and ICD codes
df = pd.read_csv(input_file, dtype=str)

print("Formatting binary columns and generating outcomes...")
# Fix: Convert binary CCS columns back to mathematical integers so the counting works
ccs_cols = [col for col in df.columns if col.startswith('CCS_')]
for col in ccs_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# Create the binary 'DIED' terminal outcome variable
df['DIED'] = (df['STUS_CD'].str.strip() == '20').astype(int)


# ==========================================
# 2. USER INPUTS & COHORT FILTERING
# ==========================================
print("\n=== CLINICAL PATHWAY EXTRACTION TOOL ===")

drg_input = input("Enter the DRG_CD you want to analyze (Press ENTER for ALL patients): ").strip()

if drg_input == "":
    df_cohort = df
    target_drg_label = "ALL PATIENTS"
else:
    df_cohort = df[df['DRG_CD'] == drg_input]
    target_drg_label = f"DRG {drg_input}"

total_n = len(df_cohort)

if total_n == 0:
    print(f"\nError: No patients found with DRG_CD '{drg_input}'. Exiting.")
    sys.exit()

print(f"\nCohort Isolated: {total_n} patients found for {target_drg_label}.")

# Random Sampling for Quick Testing
sample_input = input(f"Enter random sample size for testing (or press ENTER to use all {total_n} patients): ").strip()
if sample_input:
    try:
        sample_size = int(sample_input)
        if sample_size < total_n:
            df_cohort = df_cohort.sample(n=sample_size, random_state=42)
            total_n = len(df_cohort)
            print(f"--> [TESTING MODE] Randomly sampled {total_n} patients for this run.")
        else:
            print(f"--> Sample size requested is greater than cohort. Using all {total_n} patients.")
    except ValueError:
        print("  [!] Invalid input. Using full cohort.")

print("\n--- Set Parameters (Press ENTER to use defaults) ---")

def get_input(prompt_text, default_val, cast_type):
    user_in = input(f"{prompt_text} [Default: {default_val}]: ").strip()
    if not user_in:
        return default_val
    try:
        return cast_type(user_in)
    except ValueError:
        print(f"  [!] Invalid input. Using default: {default_val}")
        return default_val

max_length = get_input("Max pathway length (e.g., 3, 4, 5)", 3, int)
min_support = get_input("Minimum Support (patient count)", 10, int)

conf_input = get_input("Min Confidence Difference (e.g., 0.15 for 15%)", 0.10, float)
min_conf_diff = conf_input if conf_input < 1 else conf_input / 100

delta_input = get_input("Min Markov Lift Delta (e.g., 0.05 for 5%)", 0.05, float)
markov_delta = delta_input if delta_input < 1 else delta_input / 100

target_died_in = input("Only show trajectories ending in Death? (y/n) [Default: n]: ").strip().lower()
target_died = True if target_died_in == 'y' else False


# ==========================================
# 3. PREPARATION & HELPER FUNCTIONS
# ==========================================
variables = [col for col in df_cohort.columns if col.startswith('CCS_') or col == 'DIED']

def get_count(*columns):
    """Returns the count of rows where all specified columns equal 1."""
    condition = True
    for col in columns:
        condition = condition & (df_cohort[col] == 1) 
    return df_cohort[condition].shape[0]

paths_by_length = {}


# ==========================================
# 4. PHASE 1: PAIRWISE (2-LEVEL) EXTRACTION
# ==========================================
print(f"\n--- Extracting Level 2 Pathways ---")
valid_2_paths = []   

for A, B in permutations(variables, 2):
    if A > B: continue 
        
    count_A, count_B, count_AB = get_count(A), get_count(B), get_count(A, B)
    if count_AB < min_support: continue
        
    prob_B_giv_A = count_AB / count_A if count_A > 0 else 0
    prob_A_giv_B = count_AB / count_B if count_B > 0 else 0
    asymmetry = abs(prob_B_giv_A - prob_A_giv_B)
    
    if asymmetry < min_conf_diff: continue
    
    direction = (A, B) if prob_B_giv_A > prob_A_giv_B else (B, A)

    # --- TERMINAL STATE LOCK (Phase 1) ---
    if direction[0] == 'DIED': 
        continue
    
    valid_2_paths.append({
        'path': direction,
        'var_set': frozenset([A, B]),
        'support': count_AB,
        'root_asymmetry': asymmetry,
        'markov_lift': None
    })

paths_by_length[2] = valid_2_paths
print(f"Found {len(valid_2_paths)} base pairs.")


# ==========================================
# 5. PHASE 2: DYNAMIC N-LEVEL EXTRACTION
# ==========================================
for current_len in range(3, max_length + 1):
    print(f"--- Extracting Level {current_len} Pathways ---")
    current_candidates = []
    prev_paths = paths_by_length.get(current_len - 1, [])
    
    if not prev_paths:
        print("  No previous paths to extend. Stopping early.")
        break
        
    for p in prev_paths:
        base_path = p['path'] 

        # --- TERMINAL STATE LOCK (Phase 2) ---
        if base_path[-1] == 'DIED':
            continue
        
        root_asym = p['root_asymmetry']
        base_support = p['support']
        last_node = base_path[-1] 
        
        for C in [v for v in variables if v not in base_path]:
            count_base_C = get_count(*base_path, C)
            if count_base_C < min_support: continue
                
            count_last = get_count(last_node)
            count_last_C = get_count(last_node, C)
            
            prob_C_giv_last = count_last_C / count_last if count_last > 0 else 0
            prob_C_giv_base = count_base_C / base_support if base_support > 0 else 0
            
            markov_lift = prob_C_giv_base - prob_C_giv_last
            
            if markov_lift >= markov_delta:
                new_path = base_path + (C,)
                current_candidates.append({
                    'path': new_path,
                    'var_set': frozenset(new_path),
                    'root_asymmetry': root_asym,
                    'markov_lift': markov_lift,
                    'support': count_base_C
                })
                
    grouped = {}
    for cand in current_candidates:
        v_set = cand['var_set']
        if v_set not in grouped: grouped[v_set] = []
        grouped[v_set].append(cand)

    final_for_len = []
    for v_set, cands in grouped.items():
        best = max(cands, key=lambda x: x['root_asymmetry'])
        final_for_len.append(best)
        
    paths_by_length[current_len] = final_for_len
    print(f"Found {len(final_for_len)} valid {current_len}-step sequences.")


# ==========================================
# 6. FINAL OUTPUT & TARGET FILTERING
# ==========================================
print("\n==========================================")
if target_died:
    print(f" HIGH RISK TRAJECTORIES ENDING IN DEATH ({target_drg_label})")
else:
    print(f" ALL VALIDATED CLINICAL PATHWAYS ({target_drg_label})")
print("==========================================")

output_count = 0
valid_visualization_paths = []

for length in range(2, max_length + 1):
    paths = paths_by_length.get(length, [])
    paths = sorted(paths, key=lambda x: x['support'], reverse=True)
    
    for p in paths:
        path_tuple = p['path']
        
        # Target Filter
        if target_died and path_tuple[-1] != 'DIED':
            continue
            
        valid_visualization_paths.append(p)
        clean_path = " -> ".join([node.replace('CCS_', '') for node in path_tuple])
        lift_str = f" | Markov Lift: {p['markov_lift']:.1%}" if p['markov_lift'] is not None else ""
        print(f"[{length}-Step] {clean_path} | Support: {p['support']}{lift_str}")
        output_count += 1

if output_count == 0:
    print("\nNo pathways matched your criteria.")
    sys.exit()

# ==========================================
# 7. INTERACTIVE VISUALIZATION (HIGH-CONTRAST TREE)
# ==========================================
import json

vis_in = input("\nGenerate High-Contrast Interactive Tree (HTML)? (y/n) [Default: y]: ").strip().lower()

if vis_in in ['', 'y', 'yes'] and output_count > 0:
    print("Building high-contrast interactive tree...")
    
    tree_data = {"name": f"Start: {target_drg_label}", "children": []}
    
    def add_to_tree(node, path_tuple, support, asym, lift):
        current_step = path_tuple[0].replace('CCS_', '')
        current_strength = lift if lift is not None else asym
        
        child = next((c for c in node.get("children", []) if c["name"] == current_step), None)
        
        if not child:
            child = {"name": current_step, "value": support, "strength": current_strength}
            if "children" not in node: node["children"] = []
            node["children"].append(child)
        else:
            if current_strength > child.get("strength", 0):
                child["strength"] = current_strength
            child["value"] += support
            
        if len(path_tuple) > 1:
            add_to_tree(child, path_tuple[1:], support, asym, lift)

    for p in valid_visualization_paths:
        add_to_tree(tree_data, p['path'], p['support'], p['root_asymmetry'], p['markov_lift'])

    tree_json = json.dumps(tree_data)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Clinical Pathways Tree - {target_drg_label}</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{ font-family: -apple-system, sans-serif; background: #f9f9f9; padding: 20px; }}
            h2 {{ color: #333; margin-bottom: 5px; }}
            p {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
            .node circle {{ fill: #fff; stroke: #4682b4; stroke-width: 2.5px; cursor: pointer; }}
            .node text {{ font: 12px sans-serif; cursor: pointer; }}
            .node--terminal circle {{ fill: #ffeaea; stroke: #cc0000; }}
            .link {{ fill: none; stroke: #4682b4; stroke-opacity: 0.35; transition: stroke-width 0.3s; }}
            #chart {{ background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <h2>Interactive Clinical Pathways: {target_drg_label}</h2>
        <p><b>Visual Key:</b> Thick lines = Strong predictors. Thin lines = Baseline/Secondary paths.</p>
        <div id="chart"></div>

        <script>
            const treeData = {tree_json};
            const margin = {{top: 40, right: 120, bottom: 40, left: 160}},
                  width = window.innerWidth - margin.left - margin.right - 60,
                  height = 800;

            const svg = d3.select("#chart").append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height)
              .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            const tree = d3.tree().size([height - 80, width - 250]);
            const root = d3.hierarchy(treeData);
            root.x0 = height / 2; root.y0 = 0;

            root.descendants().forEach((d, i) => {{
                d.id = i;
                if (d.depth > 0) {{ d._children = d.children; d.children = null; }}
            }});

            update(root);

            function update(source) {{
                const treeData = tree(root);
                const nodes = treeData.descendants(), links = treeData.descendants().slice(1);
                nodes.forEach(d => d.y = d.depth * 280);

                const node = svg.selectAll("g.node").data(nodes, d => d.id);
                const nodeEnter = node.enter().append("g")
                    .attr("class", d => "node" + (d.data.name === "DIED" ? " node--terminal" : ""))
                    .attr("transform", d => "translate(" + source.y0 + "," + source.x0 + ")")
                    .on("click", click);

                nodeEnter.append("circle").attr("r", 7);
                nodeEnter.append("text")
                    .attr("dy", ".35em")
                    .attr("x", d => d.children || d._children ? -13 : 13)
                    .attr("text-anchor", d => d.children || d._children ? "end" : "start")
                    .text(d => d.data.name);

                const nodeUpdate = nodeEnter.merge(node);
                nodeUpdate.transition().duration(500).attr("transform", d => "translate(" + d.y + "," + d.x + ")");
                node.exit().remove();

                const link = svg.selectAll("path.link").data(links, d => d.id);
                
                const linkEnter = link.enter().insert("path", "g").attr("class", "link")
                    .attr("d", d => {{
                        const o = {{x: source.x0, y: source.y0}};
                        return diagonal(o, o);
                    }})
                    .style("stroke-width", d => {{
                        const str = d.data.strength || 0.05;
                        // EXAGGERATED SCALE: Base 1.5px + (Lift * 50)
                        return (1.5 + (str * 50)) + "px";
                    }});

                const linkUpdate = linkEnter.merge(link);
                linkUpdate.transition().duration(500).attr("d", d => diagonal(d, d.parent));

                link.exit().remove();
                nodes.forEach(d => {{ d.x0 = d.x; d.y0 = d.y; }});
            }}

            function diagonal(s, d) {{
                return `M ${{s.y}} ${{s.x}} C ${{ (s.y + d.y) / 2 }} ${{s.x}}, ${{ (s.y + d.y) / 2 }} ${{d.x}}, ${{d.y}} ${{d.x}}`;
            }}

            function click(event, d) {{
                if (d.children) {{ d._children = d.children; d.children = null; }}
                else {{ d.children = d._children; d._children = null; }}
                update(d);
            }}
        </script>
    </body>
    </html>
    """

    file_name = f"High_Contrast_Tree_{target_drg_label.replace(' ', '_')}.html"
    with open(file_name, "w") as file:
        file.write(html_content)
    print(f"\n--> SUCCESS! High-contrast tree saved as: '{file_name}'")

# ==========================================
# 8. NARRATIVE GENERATOR (DETAILED TOP 5)
# ==========================================
print("\n--- Identifying Top 5 Paths of Greatest Escalation ---")

# 1. Load and Clean Descriptor Mapping
try:
    # Load the file as raw text first to avoid quote-parsing issues
    mapping_df = pd.read_csv("icd-ccs.csv", dtype=str)
    
    # BRUTE FORCE CLEANING: 
    # 1. Remove single quotes and whitespace from column names
    mapping_df.columns = [col.replace("'", "").strip() for col in mapping_df.columns]
    
    # 2. Remove single quotes and whitespace from every single cell in the data
    mapping_df = mapping_df.apply(lambda x: x.str.replace("'", "").str.strip())
    
    ccs_col = 'CCS CATEGORY'
    desc_col = 'CCS CATEGORY DESCRIPTION'
    
    # 3. Build dictionary using 'lstrip' to handle leading zero mismatches (e.g., '062' vs '62')
    mapping_dict = {
        str(ccs).lstrip('0'): str(desc) 
        for ccs, desc in zip(mapping_df[ccs_col], mapping_df[desc_col])
    }
    mapping_dict['DIED'] = "Death"
    
    # Debug: Print the first 3 items to console to verify the mapping is working
    sample_keys = list(mapping_dict.keys())[:3]
    print(f"--> Map Verification: { {k: mapping_dict[k] for k in sample_keys} }")

except Exception as e:
    print(f"Warning: Could not map descriptors ({e}). Using raw codes.")
    mapping_dict = {}

# 2. Extract and Calculate Gains (Logic remains the same)
all_death_metrics = []
for length in paths_by_length:
    for p in paths_by_length[length]:
        path = p['path']
        if path[-1] == 'DIED' and len(path) > 1:
            count_s1 = get_count(path[0])
            count_s1_died = get_count(path[0], 'DIED')
            prob_initial = count_s1_died / count_s1 if count_s1 > 0 else 0
            
            preceding_seq = path[:-1]
            count_seq = get_count(*preceding_seq)
            count_full_path = p['support'] 
            prob_terminal = count_full_path / count_seq if count_seq > 0 else 0
            
            risk_jump = prob_terminal - prob_initial
            all_death_metrics.append({
                'path': path, 'initial_risk': prob_initial, 
                'terminal_risk': prob_terminal, 'jump': risk_jump, 
                'support': count_full_path
            })

top_5_paths = sorted(all_death_metrics, key=lambda x: x['jump'], reverse=True)[:5]

if top_5_paths:
    print("\n" + "="*85)
    print("   TOP 5 CLINICAL SEQUENCES BY INFORMATION GAIN (DEATH)")
    print("="*85)
    
    for rank, data in enumerate(top_5_paths, 1):
        path_tuple = data['path']
        
        def get_formatted_desc(raw_code):
            # Clean the code from the dataset to match the cleaned dictionary keys
            clean_code = raw_code.replace('CCS_', '').strip().lstrip('0')
            if clean_code == "DIED": return "Death"
            
            # Lookup in the dictionary
            desc = mapping_dict.get(clean_code)
            
            if desc:
                return f"{desc} (CCS={clean_code})"
            else:
                # Fallback if the code still isn't found
                return f"CCS Code {clean_code}"

        drg_context = target_drg_label if target_drg_label != "ALL PATIENTS" else "all analyzed DRGs"
        
        start_desc = get_formatted_desc(path_tuple[0])
        narrative = f"For patients in {drg_context}, this sequence describes a specific high-risk trajectory. "
        narrative += f"A patient underwent {start_desc} (Starting Risk: {data['initial_risk']:.1%}). "
        
        for i in range(1, len(path_tuple) - 1):
            mid_desc = get_formatted_desc(path_tuple[i])
            narrative += f"This was followed by {mid_desc}, "
        
        narrative += f"at which point the probability of death escalated to {data['terminal_risk']:.1%} (+{data['jump']:.1%}), and the patient died."

        print(f"\nRANK #{rank} | Risk Escalation: +{data['jump']:.1%} | Support: {data['support']} patients")
        print(f"Sequence: {' -> '.join(path_tuple)}")
        print(f"Narrative: \"{narrative}\"")
        print("-" * 60)
    print("="*85 + "\n")

# ==========================================
# 9. PREDICTIVE MODELING (FULL OPTIMIZED SET)
# ==========================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np

print("\n--- Engineering Training Set from ALL Extracted Pathways ---")

# 1. High-Performance Feature Engineering (Prevents Fragmentation Warning)
feature_series_list = []
y_train = df_cohort['DIED']

path_count = 0
for length, paths in paths_by_length.items():
    for p in paths:
        sequence_cols = [c for c in p['path'] if c != 'DIED']
        if not sequence_cols:
            continue
            
        path_count += 1
        f_name = f"PATH_{path_count}_LEN_{len(sequence_cols)}"
        
        # Create Series and add to list for bulk concatenation
        s = df_cohort[sequence_cols].all(axis=1).astype(int)
        s.name = f_name
        feature_series_list.append(s)

# 2. Bulk Concatenation
if feature_series_list:
    X_train = pd.concat(feature_series_list, axis=1)
    print(f"--> Successfully generated {path_count} unique trajectory features.")
else:
    X_train = pd.DataFrame(index=df_cohort.index)
    print("--> No valid trajectory features found.")

# 3. Model Training & Analysis
if path_count > 0 and y_train.sum() > 0:
    # Logistic Regression with L2 Regularization (C=0.1) and Balanced Weights
    model = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 4. Generate Predictions and Metrics
    y_pred = model.predict(X_train)
    y_probs = model.predict_proba(X_train)[:, 1]
    
    auc = roc_auc_score(y_train, y_probs)
    cm = confusion_matrix(y_train, y_pred)
    
    print("\n" + "="*60)
    print("   PREDICTIVE MODEL PERFORMANCE REPORT")
    print("="*60)
    print(f"Total Trajectory Features: {path_count}")
    print(f"Overall Model AUC-ROC:     {auc:.3f}")
    print("-" * 40)
    print("CONFUSION MATRIX DETAILS:")
    print(f"True Negatives (Survived): {cm[0,0]}")
    print(f"False Positives (Type I):  {cm[0,1]}")
    print(f"False Negatives (Type II): {cm[1,0]}")
    print(f"True Positives (Died):     {cm[1,1]}")
    print("-" * 40)
    print("CLASSIFICATION REPORT:")
    # Handling target names safely in case classes are missing
    target_names = ['Survived', 'Died'] if len(np.unique(y_train)) > 1 else None
    print(classification_report(y_train, y_pred, target_names=target_names))
    
    # 5. Extract Feature Importance (Top Weights)
    coefficients = model.coef_[0]
    top_indices = np.argsort(coefficients)[-5:][::-1]
    
    print("TOP 5 TRAJECTORIES DRIVING THE PREDICTION:")
    for idx in top_indices:
        print(f"- {X_train.columns[idx]} | Coefficient: {coefficients[idx]:.4f}")
    print("="*60 + "\n")

    # 6. Visualization (Check for Seaborn)
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Survived', 'Predicted Died'], 
                    yticklabels=['Actual Survived', 'Actual Died'])
        plt.title(f"Confusion Matrix: Trajectory Predictors ({target_drg_label})")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("[Note] Seaborn/Matplotlib not found. Skipping graphical matrix.")

else:
    print("\n[!] Error: Insufficient death events or features to train the model.")

# ==========================================
# 10. PERFORMANCE COMPARISON: RAW VS. SEQUENCES
# ==========================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np

print("\n--- Running Head-to-Head Comparison ---")

# 1. Prepare Model A: Raw CCS Codes (Strictly Numeric)
safe_dtypes = df_cohort.select_dtypes(include=[np.number, bool])
raw_ccs_cols = [col for col in safe_dtypes.columns if str(col).startswith('CCS_')]

# Filter for minimum support to prevent sparse matrix math hangs
X_raw_temp = safe_dtypes[raw_ccs_cols].fillna(0).astype(int)
frequent_raw_cols = [col for col in X_raw_temp.columns if X_raw_temp[col].sum() >= 5]
X_raw = X_raw_temp[frequent_raw_cols]

# 2. Prepare Model B: Trajectory Sequences
X_seq = X_train.fillna(0).astype(int)

# 3. Training & Evaluation Function
def evaluate_model(X, y, label):
    if X.empty: 
        print(f"[!] {label} dataset is empty.")
        return None
        
    # max_iter set to 500; solver liblinear is highly optimized for binary features
    m = LogisticRegression(max_iter=500, C=0.1, class_weight='balanced', solver='liblinear')
    m.fit(X, y)
    
    probs = m.predict_proba(X)
    
    return {
        'label': label,
        'auc': roc_auc_score(y, probs[:, 1]),
        'log_loss': log_loss(y, probs),
        'features': X.shape[1]
    }

# 4. Execute Comparison
print(f"--> Training Raw Model ({X_raw.shape[1]} features)...")
results_raw = evaluate_model(X_raw, y_train, "Raw CCS Codes")

print(f"--> Training Sequence Model ({X_seq.shape[1]} features)...")
results_seq = evaluate_model(X_seq, y_train, "Trajectory Sequences")

# 5. Output Results Table
print("\n" + "="*75)
print("   COMPARATIVE PERFORMANCE RESULTS")
print("="*75)
print(f"{'Model Type':<25} | {'Features':<10} | {'AUC-ROC':<10} | {'Log-Loss':<10}")
print("-" * 75)

for res in [results_raw, results_seq]:
    if res:
        print(f"{res['label']:<25} | {res['features']:<10} | {res['auc']:.4f}     | {res['log_loss']:.4f}")

# 6. Scientific Conclusion
print("-" * 75)
if results_seq and results_raw:
    auc_diff = results_seq['auc'] - results_raw['auc']
    if auc_diff > 0:
        print(f"RESULT: Trajectories improved AUC by {auc_diff:.4f}.")
    else:
        print(f"RESULT: Raw codes maintained higher AUC (Diff: {abs(auc_diff):.4f}).")
print("="*75 + "\n")

# ==========================================
# 11. THE HYBRID MODEL: COMBINING RAW + SEQUENCES
# ==========================================
import pandas as pd

print("\n--- Running the Hybrid Model Comparison ---")

# 1. Create the Hybrid Feature Set
# We combine the raw binary columns with the sequence binary columns.
# Because X_raw uses 'CCS_' and X_seq uses 'PATH_', there are no column name collisions.
X_hybrid = pd.concat([X_raw, X_seq], axis=1)

# 2. Train the Hybrid Model
print(f"--> Training Hybrid Model ({X_hybrid.shape[1]} total combined features)...")
results_hybrid = evaluate_model(X_hybrid, y_train, "Hybrid (Raw + Sequences)")

# 3. Output the Final Showdown Table
print("\n" + "="*85)
print("   FINAL SHOWDOWN: RAW vs. SEQUENCES vs. HYBRID")
print("="*85)
print(f"{'Model Type':<30} | {'Features':<10} | {'AUC-ROC':<10} | {'Log-Loss':<10}")
print("-" * 85)

for res in [results_raw, results_seq, results_hybrid]:
    if res:
        print(f"{res['label']:<30} | {res['features']:<10} | {res['auc']:.4f}     | {res['log_loss']:.4f}")

# 4. Scientific Conclusion for the Hybrid Model
print("-" * 85)
if results_hybrid and results_raw:
    hybrid_boost = results_hybrid['auc'] - results_raw['auc']
    
    if hybrid_boost > 0.001: 
        print(f"HYBRID WIN: Adding trajectories improved the AUC by {hybrid_boost:.4f}!")
        print("Conclusion: The temporal order of procedures carries independent, additive predictive value.")
    elif hybrid_boost < -0.001:
        print(f"HYBRID PENALTY: Adding trajectories lowered the AUC by {abs(hybrid_boost):.4f}.")
        print("Conclusion: The trajectories added mathematical 'noise' or excessive collinearity.")
    else:
        print("HYBRID NEUTRAL: The trajectories did not significantly change the predictive power.")
        print("Conclusion: The raw procedures alone capture the vast majority of the variance in mortality.")
print("="*85 + "\n")

# ==========================================
# 12. FEATURE IMPORTANCE: THE HERO FEATURES
# ==========================================
import numpy as np
from sklearn.linear_model import LogisticRegression

print("\n--- Extracting Top 10 'Hero Features' from the Hybrid Model ---")

# 1. Re-fit the Hybrid Model to access the mathematical weights
hybrid_model = LogisticRegression(max_iter=500, C=0.1, class_weight='balanced', solver='liblinear')
hybrid_model.fit(X_hybrid, y_train)

# 2. Extract and Sort Coefficients
coefficients = hybrid_model.coef_[0]
feature_names = X_hybrid.columns

# Get the indices of the top 10 highest positive weights (strongest predictors of death)
top_10_indices = np.argsort(coefficients)[-10:][::-1]

# 3. Output the Leaderboard
print("\n" + "="*75)
print("   HYBRID MODEL LEADERBOARD: STRONGEST PREDICTORS OF MORTALITY")
print("="*75)
print(f"{'Rank':<5} | {'Feature Name':<45} | {'Coefficient':<10}")
print("-" * 75)

sequence_count = 0
for rank, idx in enumerate(top_10_indices, 1):
    feat = feature_names[idx]
    weight = coefficients[idx]
    
    # Add a visual flag to easily spot your engineered trajectories
    if feat.startswith("PATH_"):
        marker = "⭐ [SEQUENCE]"
        sequence_count += 1
    else:
        marker = "   [RAW CODE]"
        
    print(f"#{rank:<4} | {feat:<45} | {weight:.4f} {marker}")

print("-" * 75)
print(f"SUMMARY: {sequence_count} out of the Top 10 features are Temporal Sequences.")
print("="*75 + "\n")


# ==========================================
# 13. PARAMETER OPTIMIZATION (GRID SEARCH)
# ==========================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import time

print("\n--- Starting Pathway Parameter Optimization ---")

# 1. Define the Custom Parameter Grid
asym_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] 
lift_delta_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
min_support = 5 

# Get the baseline to beat 
baseline_raw_auc = results_raw['auc']
print(f"Target to Beat: Raw Model AUC of {baseline_raw_auc:.4f}\n")

results_log = []

# 2. The Optimization Loop
total_combinations = len(asym_grid) * len(lift_delta_grid)
current_run = 1

for asym_thresh in asym_grid:
    for lift_thresh in lift_delta_grid:
        t0 = time.time()
        
        feature_series = []
        path_count = 0
        
        # A. Filter using the EXACT keys from your dictionary
        for length, paths in paths_by_length.items():
            for p in paths:
                r_asym = p.get('root_asymmetry')
                m_lift = p.get('markov_lift')
                sup = p.get('support')
                
                # SAFETY CATCH: Handle NoneTypes
                r_asym = 0.0 if r_asym is None else r_asym
                m_lift = 0.0 if m_lift is None else m_lift
                sup = 0 if sup is None else sup
                
                # MATHEMATICAL FIX: Removed the "- 1.0" since data is pure delta
                if sup >= min_support and r_asym >= asym_thresh and m_lift >= lift_thresh:
                    sequence_cols = [c for c in p['path'] if c != 'DIED']
                    if sequence_cols:
                        path_count += 1
                        s = df_cohort[sequence_cols].all(axis=1).astype(int)
                        s.name = f"PATH_{path_count}"
                        feature_series.append(s)
        
        # B. Handle empty filters safely
        if not feature_series:
            results_log.append({
                'RootAsym': asym_thresh, 'MinLiftDelta': lift_thresh, 'Paths': 0, 
                'Hybrid_AUC': baseline_raw_auc, 'Improvement': 0.0
            })
            print(f"[{current_run}/{total_combinations}] RootAsym: {asym_thresh:.2f} | MinLiftDelta: {lift_thresh:.2f} --> Paths: 0    | Boost: +0.0000 ({time.time()-t0:.1f}s)")
            current_run += 1
            continue
            
        # C. Build Hybrid Matrix
        X_seq_temp = pd.concat(feature_series, axis=1)
        X_hybrid_temp = pd.concat([X_raw, X_seq_temp], axis=1)
        
        # D. Train Hybrid Model
        model = LogisticRegression(max_iter=500, C=0.1, class_weight='balanced', solver='liblinear')
        model.fit(X_hybrid_temp, y_train)
        
        # E. Evaluate
        probs = model.predict_proba(X_hybrid_temp)[:, 1]
        hybrid_auc = roc_auc_score(y_train, probs)
        improvement = hybrid_auc - baseline_raw_auc
        
        # F. Log Results
        results_log.append({
            'RootAsym': asym_thresh, 
            'MinLiftDelta': lift_thresh, 
            'Paths': path_count, 
            'Hybrid_AUC': hybrid_auc, 
            'Improvement': improvement
        })
        
        print(f"[{current_run}/{total_combinations}] RootAsym: {asym_thresh:.2f} | MinLiftDelta: {lift_thresh:.2f} --> Paths: {path_count:<4} | Boost: {improvement:+.4f} ({time.time()-t0:.1f}s)")
        current_run += 1

# 3. Analyze and Output the Best Combination
print("\n" + "="*75)
print("   OPTIMIZATION RESULTS LEADERBOARD")
print("="*75)

df_results = pd.DataFrame(results_log)
df_results = df_results.sort_values(by='Improvement', ascending=False).reset_index(drop=True)

print(df_results.head(5).to_string(index=False, formatters={
    'RootAsym': '{:.2f}'.format,
    'MinLiftDelta': '{:.2f}'.format,
    'Hybrid_AUC': '{:.4f}'.format,
    'Improvement': '{:+.4f}'.format
}))

print("-" * 75)
best_params = df_results.iloc[0]
if best_params['Improvement'] > 0:
    print(f"🏆 BEST PARAMETERS: Root Asym >= {best_params['RootAsym']:.2f} and Min Lift Delta >= {best_params['MinLiftDelta']:.2f}")
    print(f"Extracted {int(best_params['Paths'])} paths to maximize the AUC boost.")
else:
    print("No parameter combination successfully improved upon the raw baseline.")
print("="*75 + "\n")



# ==========================================
# 14. MINIMUM FEATURE SET EXTRACTION (L1 LASSO)
# ==========================================
from sklearn.feature_selection import SelectFromModel

print("\n--- Extracting Minimum Feature Set (L1 Regularization) ---")

# Note: We are running this on the X_hybrid created in Section 11. 
# If you want to run this on the "Best" optimized hybrid, ensure X_hybrid contains those optimal paths.

# 1. Define the "Ruthless" Selector Model
# penalty='l1' mathematically forces weak features to exactly 0.0
# C=0.05 is highly aggressive. If it kills too many features, raise it to 0.1 or 0.2.
selector_model = LogisticRegression(penalty='l1', C=0.05, max_iter=1000, class_weight='balanced', solver='liblinear')

# 2. Fit the selector to the Hybrid dataset
selector = SelectFromModel(selector_model)
selector.fit(X_hybrid, y_train)

# 3. Create the 'Lean' Matrix by keeping only the surviving columns
X_lean = X_hybrid.loc[:, selector.get_support()]

print(f"Original Hybrid Features: {X_hybrid.shape[1]}")
print(f"Minimum 'Lean' Features:  {X_lean.shape[1]}")
print(f"Features Eliminated:      {X_hybrid.shape[1] - X_lean.shape[1]}\n")

# 4. Evaluate the new, lightweight model
results_lean = evaluate_model(X_lean, y_train, "Lean Hybrid Model")

# 5. Output the Showdown
print("\n" + "="*80)
print("   LEAN MODEL PERFORMANCE COMPARISON")
print("="*80)
print(f"{'Model Type':<30} | {'Features':<10} | {'AUC-ROC':<10} | {'Log-Loss':<10}")
print("-" * 80)

# Assuming results_hybrid from Section 11 is still in memory
for res in [results_hybrid, results_lean]: 
    if res:
        print(f"{res['label']:<30} | {res['features']:<10} | {res['auc']:.4f}     | {res['log_loss']:.4f}")
print("="*80 + "\n")

# 6. Print the Surviving Roster
print("SURVIVING LEAN FEATURES:")
print("-" * 30)
sequence_survivors = 0
for feat in X_lean.columns:
    if feat.startswith("PATH_"):
        print(f"- {feat:<15} ⭐ [SEQUENCE]")
        sequence_survivors += 1
    else:
        print(f"- {feat}")

print("-" * 30)
print(f"SUMMARY: {sequence_survivors} engineered sequences survived the Lasso reduction.")
print("="*80 + "\n")
