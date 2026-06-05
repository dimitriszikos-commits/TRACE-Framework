import pandas as pd
import numpy as np
import time
import sys
from collections import defaultdict

# ==========================================
# 1. FILE CONFIGURATION
# ==========================================
SEED_FILE = "2024_SEED_CCS.csv"
MAPPING_FILE = "PRCCSR_v2026-1.csv"
OUTPUT_FILE = "2024_SEED_CCSR_UPDATED.csv"

# ==========================================
# 2. LOAD & CLEAN MAPPING DICTIONARY
# ==========================================
print(f"Loading CCSR mapping file: {MAPPING_FILE}...")
try:
    map_df = pd.read_csv(MAPPING_FILE, dtype=str)
except FileNotFoundError:
    print(f"[!] Error: Could not find {MAPPING_FILE}. Please ensure it is in the same directory.")
    sys.exit()

map_df.columns = [col.replace("'", "").strip() for col in map_df.columns]
required_cols = ['ICD-10-PCS', 'PRCCSR']
for col in required_cols:
    if col not in map_df.columns:
        print(f"[!] Error: Mapping file is missing '{col}'.")
        sys.exit()

map_df['ICD-10-PCS'] = map_df['ICD-10-PCS'].str.replace("'", "").str.strip()
map_df['PRCCSR'] = map_df['PRCCSR'].str.replace("'", "").str.strip()

ccsr_dict = dict(zip(map_df['ICD-10-PCS'], map_df['PRCCSR']))
print(f"--> Successfully loaded {len(ccsr_dict)} ICD-10-PCS to CCSR mappings.")

# ==========================================
# 3. LOAD SEED DATA & PREPARE COLUMNS
# ==========================================
print(f"\nLoading Seed Data: {SEED_FILE}...")
df = pd.read_csv(SEED_FILE, dtype=str)

legacy_ccs_cols = [col for col in df.columns if col.startswith('CCS_')]
if legacy_ccs_cols:
    df = df.drop(columns=legacy_ccs_cols)
    print(f"--> Removed {len(legacy_ccs_cols)} legacy 'CCS_' columns.")

expected_pr_cols = [f"ICD_PRCDR_CD{i}" for i in range(1, 26)]
pr_cols = [col for col in expected_pr_cols if col in df.columns]

if not pr_cols:
    print(f"[!] Error: Could not find any columns matching 'ICD_PRCDR_CD1' to 'ICD_PRCDR_CD25'.")
    sys.exit()
else:
    print(f"--> Detected {len(pr_cols)} raw procedure columns.")

# ==========================================
# 4. ULTRA-FAST MAPPING (COLUMNAR ARRAYS)
# ==========================================
print("\nPre-cleaning text data (Vectorized)...")
for col in pr_cols:
    df[col] = df[col].fillna('').astype(str).str.strip().str.upper()
    df[col] = df[col].replace(['NAN', 'NONE', 'NULL'], '')

pr_matrix = df[pr_cols].values
total_rows = len(pr_matrix)
max_recurrence_depth = 0

print(f"\nBeginning mapping loop for {total_rows:,} patients...")
start_time = time.time()

# Create a dictionary of NumPy arrays (int8 saves massive memory)
# If a new CCSR code is found, it instantly creates a column of 404,000 zeros
cols_data = defaultdict(lambda: np.zeros(total_rows, dtype=np.int8))

# Iterate over the raw matrix
for idx, row in enumerate(pr_matrix):
    if idx > 0 and idx % 20000 == 0:
        print(f"  ... processed {idx:,} / {total_rows:,} patients...")
        
    ccsr_counts = {}
    
    for icd_code in row:
        if icd_code:
            if icd_code in ccsr_dict:
                ccsr_cat = ccsr_dict[icd_code]
                ccsr_counts[ccsr_cat] = ccsr_counts.get(ccsr_cat, 0) + 1
                
    for ccsr_cat, count in ccsr_counts.items():
        # Flip the 0 to a 1 for this specific patient (idx)
        cols_data[f"CCSR_{ccsr_cat}"][idx] = 1
        
        if count > 1:
            if count > max_recurrence_depth:
                max_recurrence_depth = count
                
            for i in range(1, count):
                suffix = "_R" if i == 1 else f"_R{i}"
                cols_data[f"CCSR_{ccsr_cat}{suffix}"][idx] = 1

print(f"\n--> Mapping finished in {time.time() - start_time:.2f} seconds.")
print("--> Constructing DataFrame (should be instant)...")

# Because cols_data is already a dictionary of full-length arrays, this takes milliseconds
ccsr_df = pd.DataFrame(cols_data)

print(f"--> Generated {ccsr_df.shape[1]} unique CCSR features.")

# ==========================================
# 5. MEMORY-SAFE MERGE & EXPORT
# ==========================================
print(f"\nMerging matrices and saving to {OUTPUT_FILE} (This may take a moment)...")

# Free up memory
del pr_matrix
del cols_data

df_final = pd.concat([df, ccsr_df], axis=1)

# Free up more memory
del df
del ccsr_df

df_final.to_csv(OUTPUT_FILE, index=False)
print("--> Update Complete! Your new CCSR dataset is ready.")
