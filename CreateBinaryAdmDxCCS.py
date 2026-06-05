import pandas as pd
import numpy as np

# Define file paths
main_data_file = "2024_SEED_CCS.csv"
mapping_file = "icd-ccs_dx.csv"
output_file = "2024_SEED_CCS_Enriched.csv"

print(f"1. Loading main dataset: {main_data_file}...")
df = pd.read_csv(main_data_file, dtype=str)

print(f"2. Loading mapping file: {mapping_file}...")
dx_map = pd.read_csv(mapping_file, dtype=str)

# ---------------------------------------------------------
# MAPPING LOGIC
# ---------------------------------------------------------
print("3. Cleaning mapping data and building dictionary...")
# Strip single quotes and whitespace from the column headers
dx_map.columns = [col.replace("'", "").strip() for col in dx_map.columns]

# Auto-Detect the correct column names by looking for keywords
try:
    icd_col = [col for col in dx_map.columns if 'ICD' in col.upper()][0]
    ccs_col = [col for col in dx_map.columns if 'CCS' in col.upper()][0]
    print(f"   --> Auto-Detected ICD Column: '{icd_col}'")
    print(f"   --> Auto-Detected CCS Column: '{ccs_col}'")
except IndexError:
    print("\n[!] ERROR: Could not auto-detect columns containing 'ICD' or 'CCS'.")
    print(f"Available columns are: {dx_map.columns.tolist()}")
    exit()

# Strip single quotes and whitespace from all the actual data values
dx_map = dx_map.apply(lambda x: x.str.replace("'", "").str.strip() if x.dtype == "object" else x)

# Build the dictionary using the auto-detected columns
icd_to_ccs_dict = dict(zip(dx_map[icd_col], dx_map[ccs_col]))

print("4. Cleaning and mapping Admitting Diagnoses (ADMTG_DGNS_CD)...")
if 'ADMTG_DGNS_CD' not in df.columns:
    print("\n[!] ERROR: 'ADMTG_DGNS_CD' column not found in the main dataset.")
    exit()

# Clean the main dataset's admitting diagnosis column
df['ADMTG_DGNS_CD'] = df['ADMTG_DGNS_CD'].fillna("").str.strip().str.replace("'", "")

# Map the cleaned ICD-10 codes to the cleaned CCS Categories
df['MAPPED_ADM_CCS'] = df['ADMTG_DGNS_CD'].map(icd_to_ccs_dict)

# Calculate and print mapping success rate
mapped_count = df['MAPPED_ADM_CCS'].notna().sum()
total_count = len(df[df['ADMTG_DGNS_CD'] != ""])
if total_count > 0:
    print(f"   --> Successfully mapped {mapped_count} out of {total_count} populated records ({(mapped_count/total_count)*100:.1f}%).")
else:
    print("   --> No admitting diagnoses found to map.")

# ---------------------------------------------------------
# FEATURE GENERATION
# ---------------------------------------------------------
print("5. Generating binary one-hot encoded features (CCS_ADM_DX_X)...")
adm_dummies = pd.get_dummies(df['MAPPED_ADM_CCS'], prefix='CCS_ADM_DX')
adm_dummies = adm_dummies.astype(int)

# Join the new binary features back to the main dataframe
df = pd.concat([df, adm_dummies], axis=1)

# Drop the temporary mapping column
df = df.drop(columns=['MAPPED_ADM_CCS'])

# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------
print(f"6. Saving enriched dataset to: {output_file}...")
df.to_csv(output_file, index=False)

print("\n" + "="*50)
print("SUCCESS!")
print(f"Created {adm_dummies.shape[1]} new Admitting Diagnosis binary features.")
print(f"New dataset saved as: {output_file}")
print("="*50)
