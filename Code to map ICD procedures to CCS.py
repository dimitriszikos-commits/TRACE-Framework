import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. Load and Clean the Mapping File
# ---------------------------------------------------------
print("Loading mapping file...")
# Read the file and strip any single quotes from the column names
mapping_df = pd.read_csv('icd-ccs.csv')
mapping_df.columns = mapping_df.columns.str.replace("'", "").str.strip()

# Target the necessary columns, clean them of quotes/whitespace, and create a dictionary
mapping_df['ICD-10-PCS CODE'] = mapping_df['ICD-10-PCS CODE'].astype(str).str.replace(r"['\"]", "", regex=True).str.strip()
mapping_df['CCS CATEGORY'] = mapping_df['CCS CATEGORY'].astype(str).str.replace(r"['\"]", "", regex=True).str.strip()

# Create the mapping dictionary: { '00800ZZ': '1', ... }
icd_to_ccs = dict(zip(mapping_df['ICD-10-PCS CODE'], mapping_df['CCS CATEGORY']))

# ---------------------------------------------------------
# 2. Load the Main Dataset
# ---------------------------------------------------------
print("Loading 2024_SEED dataset...")
# Load dataset treating all columns as strings to prevent zero-dropping in ICD codes
df = pd.read_csv('2024_SEED.csv', dtype=str)

# ---------------------------------------------------------
# 3. Map ICD-10-PCS to CCS
# ---------------------------------------------------------
print("Mapping procedures to CCS codes...")
# Identify the procedure columns (ICD_PRCDR_CD1 through ICD_PRCDR_CD25)
# This list comprehension ensures we only try to process columns that actually exist in the file
proc_cols = [f'ICD_PRCDR_CD{i}' for i in range(1, 26)]
proc_cols = [col for col in proc_cols if col in df.columns]

# Create a temporary dataframe to hold just the mapped CCS codes
# We strip whitespace from the raw ICD codes before mapping to ensure a clean match
mapped_ccs = df[proc_cols].apply(
    lambda col: col.str.strip().map(icd_to_ccs)
)

# ---------------------------------------------------------
# 4. Generate Binary Dummy Variables
# ---------------------------------------------------------
print("Generating CCS dummy variables...")
# Stack the dataframe to create a single column of all CCS codes for all patients
# dropna() automatically removes any unmatched codes or empty procedure slots
stacked_ccs = mapped_ccs.stack()

if not stacked_ccs.empty:
    # Use get_dummies on the stacked series
    # groupby(level=0).max() ensures that if a patient has the same CCS code multiple times, 
    # it remains a '1' (binary presence) rather than counting up
    ccs_dummies = pd.get_dummies(stacked_ccs).groupby(level=0).max()
    
    # Prefix the column names for clarity (e.g., '1' becomes 'CCS_1')
    ccs_dummies.columns = [f'CCS_{col}' for col in ccs_dummies.columns]
    
    # Join the binary matrix back to the original dataframe
    df_out = df.join(ccs_dummies)
    
    # Fill any NaN values in the new dummy columns with 0 and convert to integers
    df_out[ccs_dummies.columns] = df_out[ccs_dummies.columns].fillna(0).astype(int)
else:
    print("Warning: No matching CCS codes were found. Outputting original dataframe.")
    df_out = df

# ---------------------------------------------------------
# 5. Export the Final Dataset
# ---------------------------------------------------------
output_file = '2024_SEED_CCS.csv'
print(f"Exporting to {output_file}...")
df_out.to_csv(output_file, index=False)
print("Process complete.")
