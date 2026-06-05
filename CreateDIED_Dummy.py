import pandas as pd

# 1. Load the dataset into memory
df = pd.read_csv("2024_SEED_CCS.csv", dtype=str)

# 2. Create the binary 'DIED' variable
df['DIED'] = (df['STUS_CD'].str.strip() == '20').astype(int)

# 3. Save the updated dataset back over the original CSV file
df.to_csv("2024_SEED_CCS.csv", index=False)

print("Successfully created the DIED variable AND saved it to the file!")
