import pandas as pd
import numpy as np
import time

# Define the file name
file_name = "LDS2024_CCSR.csv"

def create_los_index(file_path):
    print(f"Loading data from {file_path}... (Using PyArrow engine for speed)")
    start_time = time.time()
    
    # Read the CSV file using the pyarrow engine for significantly faster loading
    # Keeping low_memory=False prevents the mixed-type DtypeWarning
    try:
        df = pd.read_csv(file_path, engine='pyarrow')
    except ImportError:
        print("Warning: 'pyarrow' not installed. Falling back to the slower default C engine.")
        print("Tip: Run 'pip install pyarrow' in your terminal for faster load times.")
        df = pd.read_csv(file_path, low_memory=False, engine='c')
        
    load_time = time.time() - start_time
    print(f"Data successfully loaded in {load_time:.2f} seconds.")
    
    # Verify the HLOS column exists from your previous step
    if 'HLOS' not in df.columns:
        print("Error: 'HLOS' column not found in the dataset. Please run the HLOS calculation script first.")
        return

    print("\nCalculating the median Hospital Length of Stay (HLOS)...")
    
    # Calculate the median, automatically ignoring blank (NaN) values
    median_hlos = df['HLOS'].median()
    print(f"--> The median HLOS across the cohort is: {median_hlos} days")
    
    print("Generating the dichotomous LOS_INDEX...")
    
    # Create the new column: 1 if strictly greater than the median, 0 otherwise.
    # np.where safely handles NaN values (missing HLOS will evaluate to False -> 0)
    df['LOS_INDEX'] = np.where(df['HLOS'] > median_hlos, 1, 0)
    
    # Save the updated DataFrame back to the CSV file
    print(f"\nSaving updated data back to {file_path}... (This may take a moment)")
    save_start = time.time()
    
    df.to_csv(file_path, index=False)
    
    save_time = time.time() - save_start
    print(f"File successfully saved in {save_time:.2f} seconds.")
    print("Success! The 'LOS_INDEX' column is ready for your Master Pipeline.")

if __name__ == "__main__":
    create_los_index(file_name)
