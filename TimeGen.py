import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
RAW_FILE = "LDS2024_CCSR.csv"
MAPPING_FILE = "PRCCSR_v2026-1.csv"
OUTPUT_TIMELINES = "Patient_Temporal_Timelines.json"

print("="*85)
print("TEMPORAL MATRIX GENERATOR: Bulletproof Edition")
print("="*85)

# ==========================================
# 2. LOAD & CLEAN MAPPING DICTIONARY
# ==========================================
print(f"Loading Mapping Dictionary: {MAPPING_FILE}...")
try:
    map_df = pd.read_csv(MAPPING_FILE, dtype=str)
    map_df.columns = [col.replace("'", "").strip() for col in map_df.columns]
    
    icd_col = 'ICD-10-PCS CODE' if 'ICD-10-PCS CODE' in map_df.columns else map_df.columns[0]
    
    # Aggressively strip quotes, whitespace, and decimals from the dictionary keys
    icd_to_ccsr = dict(zip(
        map_df[icd_col].str.replace("'", "").str.replace(".", "", regex=False).str.strip(), 
        map_df['PRCCSR'].str.replace("'", "").str.strip()
    ))
except FileNotFoundError:
    print(f"[!] Error: Could not find {MAPPING_FILE}.")
    sys.exit()

# ==========================================
# 3. LOAD RAW DATA
# ==========================================
print(f"Loading Raw Cohort Data: {RAW_FILE}...")
try:
    df = pd.read_csv(RAW_FILE, dtype=str)
except FileNotFoundError:
    print(f"[!] Error: Could not find {RAW_FILE}.")
    sys.exit()

ID_COL = 'PATIENT_ID' if 'PATIENT_ID' in df.columns else 'ROW_INDEX'
if ID_COL == 'ROW_INDEX':
    df['ROW_INDEX'] = df.index.astype(str)

# ==========================================
# 4. CHRONOLOGICAL TIMELINE EXTRACTION
# ==========================================
print("\nExtracting and mapping temporal patient journeys...")

patient_timelines = {}
total_patients = len(df)
patients_with_events = 0
failed_dates_sample = set()
failed_maps_sample = set()

for idx, row in df.iterrows():
    patient_id = row[ID_COL]
    raw_events = []
    
    for i in range(1, 26):
        icd_val = row.get(f'ICD_PRCDR_CD{i}')
        dt_val = row.get(f'PRCDR_DT{i}')
        
        # Check if BOTH exist and are not 'nan' strings
        if pd.notna(icd_val) and pd.notna(dt_val) and str(icd_val).strip().lower() != 'nan' and str(dt_val).strip().lower() != 'nan':
            
            # --- THE SCRUBBERS ---
            # Remove decimals from ICD codes (e.g., '0DT.J0ZZ' -> '0DTJ0ZZ')
            clean_icd = str(icd_val).replace('.', '').replace("'", "").strip()
            
            # Remove '.0' from dates (e.g., '20240115.0' -> '20240115')
            clean_dt = str(dt_val).split('.')[0].strip() 
            
            # Try to parse the date
            try:
                event_date = datetime.strptime(clean_dt, '%Y%m%d')
            except ValueError:
                if len(failed_dates_sample) < 5: failed_dates_sample.add(clean_dt)
                continue 
                
            # Try to map to CCSR
            ccsr_code = icd_to_ccsr.get(clean_icd)
            if ccsr_code:
                raw_events.append({
                    'date': event_date,
                    'ccsr': ccsr_code
                })
            else:
                if len(failed_maps_sample) < 5: failed_maps_sample.add(clean_icd)
                
    if not raw_events:
        continue
        
    patients_with_events += 1
    
    # Sort chronologically by date
    raw_events.sort(key=lambda x: x['date'])
    
    # Calculate Relative Days
    baseline_date = raw_events[0]['date']
    ccsr_counts = {}
    timeline_dict = {}
    
    for event in raw_events:
        rel_day = (event['date'] - baseline_date).days + 1
        base_ccsr = f"CCSR_{event['ccsr']}"
        
        if base_ccsr not in ccsr_counts:
            ccsr_counts[base_ccsr] = 1
            node_name = base_ccsr
        else:
            ccsr_counts[base_ccsr] += 1
            node_name = f"{base_ccsr}_R{ccsr_counts[base_ccsr]}"
            
        timeline_dict[node_name] = rel_day
        
    patient_timelines[patient_id] = timeline_dict
    
    if (idx + 1) % 10000 == 0:
        print(f"  -> Processed {idx + 1} / {total_patients} patients...")

# ==========================================
# 5. DIAGNOSTIC REPORT & EXPORT
# ==========================================
print(f"\nExtraction complete. Found valid temporal timelines for {patients_with_events} patients.")

if patients_with_events == 0:
    print("\n[!] CRITICAL FAILURE: Still 0 patients.")
    print(f"    Sample of rejected dates (Format Issue?): {list(failed_dates_sample)}")
    print(f"    Sample of rejected ICDs (Mapping Issue?): {list(failed_maps_sample)}")
else:
    print(f"Saving timelines to {OUTPUT_TIMELINES}...")
    with open(OUTPUT_TIMELINES, 'w') as f:
        json.dump(patient_timelines, f, indent=4)
    print("Pipeline Ready! You can now load this JSON into the calculate_temporal_accuracy() function.")
