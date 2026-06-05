import pandas as pd
import numpy as np

# Lock seed for reproducibility
np.random.seed(42)
n_cases = 3000

print("Generating clinical matrix for dual-trigger testing...")

# 1. Base Variables
age = np.clip(np.random.normal(loc=65, scale=15, size=n_cases), 18, 99)
bmi = np.clip(np.random.normal(loc=28, scale=6, size=n_cases), 15, 50)
elixhauser_diabetes = np.random.binomial(1, p=0.25, size=n_cases)

# CHF strongly linked to Age
p_chf = 1 / (1 + np.exp(-((age - 65) * 0.08)))
elixhauser_chf = np.random.binomial(1, p=p_chf)

# Routine A1C lab draw (Simulating a continuous lab value)
a1c_lab = np.clip(np.random.normal(loc=6.0, scale=1.5, size=n_cases) + (elixhauser_diabetes * 2.5), 4.0, 14.0)

# Target Outcome (Mortality)
logit = (age * 0.04) + (elixhauser_chf * 1.5) + (elixhauser_diabetes * 0.8) + (bmi * 0.02) - 6.0
outcome = np.random.binomial(1, p=1 / (1 + np.exp(-logit)))

df = pd.DataFrame({
    'age': np.round(age, 1),
    'bmi': np.round(bmi, 1),
    'elixhauser_diabetes': elixhauser_diabetes,
    'elixhauser_chf': elixhauser_chf,
    'a1c_lab': np.round(a1c_lab, 1),
    'outcome': outcome
})

print("Injecting DUAL-TRIGGER missingness patterns...")

# --- Pattern 1: MAR (Dual Clinical Trigger) ---
# Missing BMI (75% drop rate) IF the patient is elderly (Age > 75) AND has Heart Failure (CHF == 1).
# Simulates frail, fluid-overloaded patients where standing scales are bypassed.
dual_mar_condition = (df['age'] > 75) & (df['elixhauser_chf'] == 1)
mar_indices = df[dual_mar_condition].sample(frac=0.75, random_state=42).index
df.loc[mar_indices, 'bmi'] = np.nan

# --- Pattern 2: MAR (Dual Guideline Trigger) ---
# Missing A1C Lab (90% drop rate) IF the patient does NOT have diabetes AND BMI is normal (< 25).
# Simulates adherence to clinical guidelines (don't draw unnecessary labs on low-risk profiles).
dual_lab_condition = (df['elixhauser_diabetes'] == 0) & (df['bmi'] < 25)
lab_indices = df[dual_lab_condition].sample(frac=0.90, random_state=43).index
df.loc[lab_indices, 'a1c_lab'] = np.nan

# --- Pattern 3: MNAR (Dual Extreme Risk Trigger) ---
# Missing Age (50% drop rate) IF Mortality occurs (Outcome == 1) AND the true Age is < 40.
# Simulates John Doe emergency trauma protocols for unexpectedly young mortalities.
dual_mnar_condition = (df['outcome'] == 1) & (df['age'] < 40)
mnar_indices = df[dual_mnar_condition].sample(frac=0.50, random_state=44).index
df.loc[mnar_indices, 'age'] = np.nan

# Export
filename = "dual_trigger_miss.csv"
df.to_csv(filename, index=False)
print(f"Dataset successfully saved to {filename}!")