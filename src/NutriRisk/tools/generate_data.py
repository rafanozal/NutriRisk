import pandas as pd
import numpy as np

from NutriRisk.constants import folders

# Set a random seed so we get the exact same data every time we run this
np.random.seed(42)

# Number of rows (patients/individuals)
n_samples = 2000

# Generate 10 input features (Nutrition and Health metrics)
age          = np.random.randint(18, 90, n_samples)
bmi          = np.random.uniform(18.5, 45.0, n_samples)
calories     = np.random.normal(2500, 500, n_samples).clip(1000, 5000)
sugar_g      = np.random.exponential(30, n_samples).clip(0, 150) # Tends to have right-skewed outliers
fiber_g      = np.random.normal(20, 8, n_samples).clip(2, 60)
protein_g    = np.random.normal(100, 30, n_samples).clip(20, 300)
fat_g        = np.random.normal(80, 25, n_samples).clip(10, 200)
carbs_g      = np.random.normal(250, 70, n_samples).clip(50, 600)
water_l      = np.random.uniform(0.5, 5.0, n_samples)
exercise_hrs = np.random.exponential(2.5, n_samples).clip(0, 14)

# Create the hidden mathematical relationship for our target
# (Higher BMI, higher sugar, and higher age increase risk. Higher exercise and fiber decrease it)
raw_risk_score = (
    (bmi - 25) * 0.15 +          # BMI over 25 increases risk
    (sugar_g - 25) * 0.03 -      # Sugar over 25g increases risk
    (exercise_hrs - 3) * 0.3 -   # Exercise over 3 hours decreases risk
    (fiber_g - 20) * 0.05 +      # Fiber over 20g decreases risk
    (age - 50) * 0.02            # Age over 50 increases risk
)

# Add some random noise so the models don't get a perfect 100% score
noise           = np.random.normal(0, 0.5, n_samples)
raw_risk_score += noise

# Map the raw score to a strict 0 to 1 range using a Sigmoid function
risk = 1 / (1 + np.exp(-raw_risk_score))

# Combine everything into a Pandas DataFrame
df = pd.DataFrame({
    'age':             age,
    'bmi':             bmi,
    'calories_kcal':   calories,
    'sugar_g':         sugar_g,
    'fiber_g':         fiber_g,
    'protein_g':       protein_g,
    'fat_g':           fat_g,
    'carbs_g':         carbs_g,
    'water_l':         water_l,
    'exercise_hrs_wk': exercise_hrs,
    'risk':            risk
})

# Save the generated data to a CSV file in the designated folder
df.to_csv(folders.RANDOM_DATA_FILEPATH, index=False)

# Display the first 5 rows and verify the max/min of the risk column
print(df.head())
print("\nRisk Column Stats:")
print(f"Min Risk: {df['risk'].min():.4f}")
print(f"Max Risk: {df['risk'].max():.4f}")