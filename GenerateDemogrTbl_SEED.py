import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATASET
# ==========================================
file_path = "/Users/dimitrioszikos/Dropbox/SeedExperiments/LDS2024_CCSR.csv"
try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Total patients: {len(df):,}\n")
except FileNotFoundError:
    print(f"File not found. Please check the path: {file_path}")
    # Dummy data for demonstration if file is missing
    np.random.seed(42)
    df = pd.DataFrame({
        'DOB_DT': np.random.choice([1, 2, 3, 4, 5, 6], 100000, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]),
        'RACE_CD': np.random.choice([0, 1, 2, 3, 4, 5, 6], 100000), 
        'SEX': np.random.choice(['M', 'F'], 100000),
        'LOS': np.random.exponential(5, 100000).clip(1, 45),
        'DIED': np.random.choice([0, 1], 100000, p=[0.95, 0.05]),
        'Readmit_30': np.random.choice([0, 1], 100000, p=[0.85, 0.15]),
        'LOS_Flag': np.random.choice([0, 1], 100000, p=[0.7, 0.3])
    })

# ==========================================
# 2. DATA DICTIONARY MAPPING
# ==========================================
# We create a copy so we don't alter your raw data
df_table = df.copy()

# Map Age Groups
age_map = {1: '<65', 2: '65-69', 3: '70-74', 4: '75-79', 5: '80-84', 6: '>84'}
if 'DOB_DT' in df_table.columns:
    df_table['DOB_DT'] = pd.to_numeric(df_table['DOB_DT'], errors='coerce').map(age_map).fillna(df_table['DOB_DT'])

# Map Readmission
readmit_map = {0: 'Not Readmitted', 1: 'Readmitted'}
if 'Readmit_30' in df_table.columns:
    df_table['Readmit_30'] = pd.to_numeric(df_table['Readmit_30'], errors='coerce').map(readmit_map).fillna(df_table['Readmit_30'])

# Map Mortality
died_map = {0: 'Survived', 1: 'Died'}
if 'DIED' in df_table.columns:
    df_table['DIED'] = pd.to_numeric(df_table['DIED'], errors='coerce').map(died_map).fillna(df_table['DIED'])

# Map Race using the provided specific encoding
race_map = {
    0: 'Unknown',
    1: 'White',
    2: 'Black',
    3: 'Other',
    4: 'Asian',
    5: 'Hispanic',
    6: 'North American Native'
}
if 'RACE_CD' in df_table.columns:
     df_table['RACE_CD'] = pd.to_numeric(df_table['RACE_CD'], errors='coerce').map(race_map).fillna(df_table['RACE_CD'])

# ==========================================
# 3. DEFINE VARIABLES & CALCULATE
# ==========================================
continuous_vars = ['LOS']
categorical_vars = ['DOB_DT', 'SEX', 'RACE_CD', 'DIED', 'Readmit_30', 'LOS_Flag']

summary_data = []

# Process Continuous Variables: Mean (SD) and Median [IQR]
for var in continuous_vars:
    if var in df_table.columns:
        clean_col = pd.to_numeric(df_table[var], errors='coerce').dropna()
        
        mean_val = clean_col.mean()
        std_val = clean_col.std()
        median_val = clean_col.median()
        q25 = clean_col.quantile(0.25)
        q75 = clean_col.quantile(0.75)
        missing = df_table[var].isna().sum()
        
        summary_data.append({'Variable': f"{var} - Mean (SD)", 'Value': f"{mean_val:.2f} ({std_val:.2f})"})
        summary_data.append({'Variable': f"{var} - Median [IQR]", 'Value': f"{median_val:.2f} [{q25:.2f} - {q75:.2f}]"})
        if missing > 0:
            summary_data.append({'Variable': f"  Missing", 'Value': f"{missing:,}"})

# Process Categorical Variables: Count (%)
for var in categorical_vars:
    if var in df_table.columns:
        summary_data.append({'Variable': f"{var} - n (%)", 'Value': ""})
        
        counts = df_table[var].value_counts(dropna=False).sort_index()
        percentages = df_table[var].value_counts(normalize=True, dropna=False).sort_index() * 100
        
        for category, count in counts.items():
            pct = percentages[category]
            cat_label = "Missing" if pd.isna(category) else str(category)
            
            summary_data.append({
                'Variable': f"  {cat_label}", 
                'Value': f"{count:,} ({pct:.1f}%)"
            })

# ==========================================
# 4. EXPORT & DISPLAY
# ==========================================
table_1 = pd.DataFrame(summary_data)

print("======================================================")
print("TABLE 1: Baseline Demographics and Clinical Outcomes")
print("======================================================")
print(table_1.to_string(index=False, justify='left'))
print("======================================================\n")

output_file = "/Users/dimitrioszikos/Dropbox/SeedExperiments/Table_1_Demographics.csv"
table_1.to_csv(output_file, index=False)
print(f"--> Success! Table exported to: {output_file}")