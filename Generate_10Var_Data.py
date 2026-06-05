import pandas as pd
import numpy as np

# Lock seed for reproducibility in pipeline validation
np.random.seed(42)
n_cases = 5000

print("Generating 10-variable clinical matrix...")

# 1. Continuous Features (High Variance)
age = np.clip(np.random.normal(loc=65, scale=15, size=n_cases), 18, 100)
bmi = np.clip(np.random.normal(loc=28, scale=6, size=n_cases), 15, 50)
systolic_bp = np.clip(np.random.normal(loc=130, scale=20, size=n_cases), 70, 220)
heart_rate = np.clip(np.random.normal(loc=80, scale=15, size=n_cases), 40, 140)
# Heavy-tailed distribution for time-to-event or duration variables
length_of_stay = np.clip(np.random.lognormal(mean=1.2, sigma=0.8, size=n_cases), 1, 90)

# 2. Elixhauser Comorbidity Indices (Binary Flags with embedded collinearity)
elixhauser_diabetes = np.random.binomial(1, p=0.25, size=n_cases)

# CHF risk mathematically linked to advanced age and diabetes
p_chf = 1 / (1 + np.exp(-((age - 65) * 0.08 + elixhauser_diabetes * 0.5)))
elixhauser_chf = np.random.binomial(1, p=p_chf)

# Renal failure linked to both CHF and extreme systolic BP
p_renal = 1 / (1 + np.exp(-((systolic_bp - 130) * 0.03 + elixhauser_chf * 1.2)))
elixhauser_renal = np.random.binomial(1, p=p_renal)

elixhauser_pulmonary = np.random.binomial(1, p=0.15, size=n_cases)
elixhauser_liver = np.random.binomial(1, p=0.05, size=n_cases)

# 3. Clinical Target Outcome 
# The pipeline starts with this foundational ML prediction dynamic before the LLM synthesis
logit = (
    (age * 0.03) + 
    (length_of_stay * 0.05) + 
    (elixhauser_chf * 1.1) + 
    (elixhauser_renal * 0.9) + 
    (elixhauser_liver * 1.5) + 
    ((systolic_bp - 120) * 0.02) - 
    6.5
)
p_outcome = 1 / (1 + np.exp(-logit))
outcome = np.random.binomial(1, p_outcome)

# Assemble DataFrame
df = pd.DataFrame({
    'age': np.round(age, 1),
    'bmi': np.round(bmi, 1),
    'systolic_bp': np.round(systolic_bp, 1),
    'heart_rate': np.round(heart_rate, 0),
    'length_of_stay': np.round(length_of_stay, 1),
    'elixhauser_diabetes': elixhauser_diabetes,
    'elixhauser_chf': elixhauser_chf,
    'elixhauser_renal': elixhauser_renal,
    'elixhauser_pulmonary': elixhauser_pulmonary,
    'elixhauser_liver': elixhauser_liver,
    'outcome': outcome
})

print("Injecting complex missingness patterns...")

# --- Pattern 1: MCAR (Pure Noise) ---
# Completely independent random drops. Engine should classify as 'Non-Informative Missingness'.
df.loc[df.sample(frac=0.15, random_state=42).index, 'heart_rate'] = np.nan
df.loc[df.sample(frac=0.05, random_state=43).index, 'elixhauser_liver'] = np.nan

# --- Pattern 2: MAR (Structural Blocks) ---
# Missingness driven by *observed* proxies. Engine should catch the redundancy.
# 1. BMI missing heavily (60%) if Age > 80 (Simulating bedbound patients bypassing standing scales)
mar_bmi_idx = df[df['age'] > 80].sample(frac=0.60, random_state=44).index
df.loc[mar_bmi_idx, 'bmi'] = np.nan

# 2. Pulmonary missing (45%) if CHF is already coded (Administrative grouping)
mar_pulm_idx = df[df['elixhauser_chf'] == 1].sample(frac=0.45, random_state=45).index
df.loc[mar_pulm_idx, 'elixhauser_pulmonary'] = np.nan

# --- Pattern 3: MNAR (Extrapolated Extreme Risk) ---
# Missingness driven by the unobserved value itself, or the outcome.
# 1. Systolic BP missing (75%) if the true BP > 180 (Emergency response bypasses routine charting)
mnar_bp_idx = df[df['systolic_bp'] > 180].sample(frac=0.75, random_state=46).index
df.loc[mnar_bp_idx, 'systolic_bp'] = np.nan

# 2. Length of Stay missing (80%) when outcome == 1 (Mortality severely truncates LOS data collection)
mnar_los_idx = df[df['outcome'] == 1].sample(frac=0.80, random_state=47).index
df.loc[mnar_los_idx, 'length_of_stay'] = np.nan

# Export
filename = "rich_missingness_10var.csv"
df.to_csv(filename, index=False)
print(f"Dataset successfully saved to {filename}!")