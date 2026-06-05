import pandas as pd
import numpy as np

# Define the file name
file_name = "LDS2024_CCSR.csv"

def calculate_hlos(file_path):
    print(f"Loading data from {file_path}...")
    
    # Read the CSV file
    # We read dates as strings initially to ensure the 'yyyymmdd' format is preserved
    df = pd.read_csv(file_path, dtype={'ADMSN_DT': str, 'THRU_DT': str})
    
    # Verify the columns exist
    if 'ADMSN_DT' not in df.columns or 'THRU_DT' not in df.columns:
        print("Error: 'ADMSN_DT' and/or 'THRU_DT' columns not found in the dataset.")
        return

    print("Calculating Hospital Length of Stay (HLOS)...")
    
    # Convert the date strings to datetime objects
    # errors='coerce' will turn invalid dates into NaT (Not a Time) instead of crashing
    admsn_dates = pd.to_datetime(df['ADMSN_DT'], format='%Y%m%d', errors='coerce')
    thru_dates = pd.to_datetime(df['THRU_DT'], format='%Y%m%d', errors='coerce')
    
    # Calculate the difference in days
    df['HLOS'] = (thru_dates - admsn_dates).dt.days
    
    # Handle cases where admission and discharge are on the same day 
    # (Optional: Sometimes healthcare data counts same-day as 1 day. 
    # If you prefer same-day to be 0 days, you can leave it as is or handle NaNs).
    # Here we are using exact mathematical difference (e.g., same day = 0 days).
    
    # Fill missing or invalid date calculations with NaN (or you can use 0 / -1)
    # df['HLOS'] = df['HLOS'].fillna(-1).astype(int) # Uncomment if you want missing to be -1

    # Save the updated DataFrame back to the same CSV file
    print(f"Saving updated data back to {file_path}...")
    df.to_csv(file_path, index=False)
    
    print("Success! The 'HLOS' column has been added to the file.")

if __name__ == "__main__":
    calculate_hlos(file_name)
