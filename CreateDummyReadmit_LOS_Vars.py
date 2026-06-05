import pandas as pd

def process_hospital_data(input_file, output_file, readmission_window=30):
    print(f"Loading data from {input_file}...")
    
    # Load with low_memory=False to prevent the DtypeWarning on large datasets
    df = pd.read_csv(input_file, low_memory=False)

    # Strip any accidental leading/trailing spaces from column headers
    df.columns = df.columns.str.strip()
    
    # Convert date columns to datetime objects
    df['THRU_DT'] = pd.to_datetime(df['THRU_DT'], format='%Y%m%d', errors='coerce')
    df['DSCHRGDT'] = pd.to_datetime(df['DSCHRGDT'], format='%Y%m%d', errors='coerce')

    # Calculate LOS (Length of Stay) in days
    print("Calculating Length of Stay (LOS)...")
    df['LOS'] = (df['DSCHRGDT'] - df['THRU_DT']).dt.days

    # Calculate Readmission
    print(f"Calculating {readmission_window}-day Readmission flags...")
    
    # Sort by patient ID and admission date to ensure chronological order
    df = df.sort_values(by=['DSYSRTKY', 'THRU_DT'])

    # Get the previous discharge date for each patient
    df['prev_discharge_date'] = df.groupby('DSYSRTKY')['DSCHRGDT'].shift(1)

    # Calculate the days between the current admission and the previous discharge
    df['days_since_prev_discharge'] = (df['THRU_DT'] - df['prev_discharge_date']).dt.days

    # Create the dichotomous Readmission column (1 = True, 0 = False)
    df['Readmission'] = (
        (df['days_since_prev_discharge'] >= 0) & 
        (df['days_since_prev_discharge'] <= readmission_window)
    ).astype(int)

    # Clean up temporary helper columns
    df = df.drop(columns=['prev_discharge_date', 'days_since_prev_discharge'])

    # Save the updated dataframe
    print(f"Saving updated data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Process complete!")

# --- Execution ---
if __name__ == "__main__":
    INPUT_CSV = '2024_SEED_CCS.csv'
    OUTPUT_CSV = '2024_SEED_CCS_with_Readmissions.csv'
    
    process_hospital_data(INPUT_CSV, OUTPUT_CSV)
