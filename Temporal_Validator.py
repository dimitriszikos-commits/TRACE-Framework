import pandas as pd
import json
import re
import numpy as np
import math
import random

# ==========================================
# 1. CONFIGURATION
# ==========================================
SEQUENCES_FILE = "/Users/dimitrioszikos/Dropbox/SeedExperiments/Divergence_Sequences.csv" 
TIMELINES_FILE = "/Users/dimitrioszikos/Dropbox/SeedExperiments/Patient_Temporal_Timelines.json" 
OUTPUT_FILE = "/Users/dimitrioszikos/Dropbox/SeedExperiments/Validated_Temporal_Sequences.csv"

print("="*85)
print("PHASE 5: RIGOROUS TEMPORAL VALIDATION (COMBINATORIAL BASELINE & STRICT GATING)")
print("="*85)

# ==========================================
# 2. LOAD DATA
# ==========================================
print(f"Loading Extracted Sequences from: {SEQUENCES_FILE}...")
try:
    seq_df = pd.read_csv(SEQUENCES_FILE)
except FileNotFoundError:
    print(f"[!] Error: Could not find {SEQUENCES_FILE}.")
    exit()

print(f"Loading Patient Timelines from: {TIMELINES_FILE}...")
try:
    with open(TIMELINES_FILE, 'r') as f:
        patient_timelines = json.load(f)
except FileNotFoundError:
    print(f"[!] Error: Could not find {TIMELINES_FILE}.")
    exit()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def parse_sequence_name(raw_name):
    clean_name = re.sub(r'_\[NODE_\d+_DOSE\]', '', raw_name)
    clean_name = clean_name.replace("SEQ_", "")
    return clean_name.split("_TO_")

def get_temporal_nodes(parsed_seq):
    return [node for node in parsed_seq if not node.startswith("NOT_")]

def get_event_day(node, timeline):
    if node.startswith("BUNDLE_"):
        parts = node.replace("BUNDLE_CCSR_", "").split("_CCSR_")
        first_ccsr = f"CCSR_{parts[0]}"
        return timeline.get(first_ccsr)
    else:
        return timeline.get(node)

# -> NEW FIX 1: THE BOOLEAN LOGIC GATE
def matches_boolean_logic(parsed_seq, timeline):
    """Filters patient IDs by recreating the exact Boolean selection logic."""
    for node in parsed_seq:
        is_negative = node.startswith("NOT_")
        clean_node = node.replace("NOT_", "")
        day = get_event_day(clean_node, timeline)
        
        if is_negative and day is not None:
            return False  # Failed: Patient had the event they were supposed to miss
        if not is_negative and day is None:
            return False  # Failed: Patient is missing a required positive event
    return True

# -> NEW FIX 2: STRICT CHRONOLOGICAL EVALUATION
def calculate_temporal_accuracy(temporal_seq, timeline):
    """Evaluates if the events occurred in the exact specified chronological order."""
    N = len(temporal_seq)
    if N <= 1: return None 
        
    days = [get_event_day(node, timeline) for node in temporal_seq]
    if None in days: return None 
    
    # Apply random tie-breaking using tuples: (Date, RandomFloat)
    # This perfectly and robustly breaks same-day ties without parsing complex date strings
    resolved_days = [(day, random.random()) for day in days]
    
    # Check if the sequence is in perfect ascending chronological order
    is_perfect_order = all(resolved_days[i] < resolved_days[i+1] for i in range(N - 1))
    
    return 1.0 if is_perfect_order else 0.0

# -> NEW FIX 3: COMBINATORIAL NULL BASELINE
def get_null_baseline(N):
    """Calculates Expected Accuracy based on pure combinatorics (1 / N!)."""
    if N <= 1: return 1.0
    return 1.0 / math.factorial(N)

baseline_cache = {}

# ==========================================
# 4. EXECUTE VALIDATION
# ==========================================
print("\nValidating sequences against actual chronological data...")

results = []

for idx, row in seq_df.iterrows():
    raw_feature = row['Raw_Feature_Name']
    narrative = row['Clinical_Narrative']
    
    train_count = row.get('Train_Patients', 0)
    val_count = row.get('Val_Patients', 0)
    test_count = row.get('Test_Patients', 0)
    total_boolean_matches = train_count + val_count + test_count
    
    parsed_seq = parse_sequence_name(raw_feature)
    temporal_nodes = get_temporal_nodes(parsed_seq)
    N = len(temporal_nodes)
    
    if N < 2:
        results.append({
            "Raw_Feature_Name": raw_feature,
            "Clinical_Narrative": narrative,
            "Temporal_Nodes_Checked": N,
            "Total_Boolean_Matches": total_boolean_matches,
            "Patients_With_Timestamps": 0,
            "Coverage_Statistic": "0.0%",
            "Null_Baseline_Accuracy": "N/A",
            "Average_Temporal_Accuracy": "N/A",
            "Net_Lift_Over_Random": "N/A"
        })
        continue
        
    patient_scores = []
    for pat_id, timeline in patient_timelines.items():
        # APPLY THE STRICT BOOLEAN GATE
        if not matches_boolean_logic(parsed_seq, timeline):
            continue
            
        score = calculate_temporal_accuracy(temporal_nodes, timeline)
        if score is not None:
            patient_scores.append(score)
            
    matched = len(patient_scores)
    coverage = (matched / total_boolean_matches) if total_boolean_matches > 0 else 0.0
    
    if matched == 0:
        avg_score = 0.0
        null_baseline = 0.0
    else:
        avg_score = sum(patient_scores) / matched
        if N not in baseline_cache:
            baseline_cache[N] = get_null_baseline(N)
        null_baseline = baseline_cache[N]
        
    lift = avg_score - null_baseline
        
    results.append({
        "Raw_Feature_Name": raw_feature,
        "Clinical_Narrative": narrative,
        "Temporal_Nodes_Checked": N,
        "Total_Boolean_Matches": total_boolean_matches,
        "Patients_With_Timestamps": matched,
        "Coverage_Statistic": f"{coverage * 100:.1f}%",
        "Null_Baseline_Accuracy": f"{null_baseline * 100:.1f}%" if matched > 0 else "N/A",
        "Average_Temporal_Accuracy": f"{avg_score * 100:.1f}%" if matched > 0 else "N/A",
        "Net_Lift_Over_Random": f"{lift * 100:+.1f}%" if matched > 0 else "N/A"
    })

# ==========================================
# 5. EXPORT RESULTS
# ==========================================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n--> Complete! Saved Rigorous Temporal Validation metrics to '{OUTPUT_FILE}'.")

# Print top 5 best-performing multi-node sequences to console
print("\n--- TOP 5 MOST CHRONOLOGICALLY ACCURATE SEQUENCES (BY NET LIFT) ---")
valid_results = [r for r in results if "N/A" not in r["Net_Lift_Over_Random"] and r["Patients_With_Timestamps"] > 0]

# Sort by the true lift over random, then by coverage
valid_results.sort(key=lambda x: (float(x["Net_Lift_Over_Random"].replace("%", "")), float(x["Coverage_Statistic"].replace("%", ""))), reverse=True)

for i, res in enumerate(valid_results[:5]):
    print(f"\n{i+1}. {res['Clinical_Narrative']}")
    print(f"    Total Boolean Matches: {res['Total_Boolean_Matches']} patients")
    print(f"    Coverage Statistic:    {res['Coverage_Statistic']} ({res['Patients_With_Timestamps']} patients with full timestamps)")
    print(f"    Null Baseline:         {res['Null_Baseline_Accuracy']}")
    print(f"    Actual Accuracy:       {res['Average_Temporal_Accuracy']}")
    print(f"    Net Lift over Random:  {res['Net_Lift_Over_Random']}")