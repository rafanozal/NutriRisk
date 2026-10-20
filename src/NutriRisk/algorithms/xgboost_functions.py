import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from NutriRisk.constants import folders

# Load the dataset
df = pd.read_csv(folders.RANDOM_DATA_FILEPATH)

# Separate the input features (X) from the target we want to predict (y)
X = df.drop(columns=['risk'])
y = df['risk']

# Split the data into a Training set (80%) and a Testing set (20%)
# We train on one set, and test on unseen data to ensure it isn't just memorizing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost Regressor
# We use 'hist' (histogram) as it is the modern, highly-optimized default for XGBoost
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', # Standard regression task
    tree_method='hist',           # Faster, modern tree building
    n_estimators=100,             # Number of trees to build
    learning_rate=0.1,            # How much each tree updates the previous ones
    max_depth=4,                  # Maximum depth of each tree
    random_state=42
)


# DEBUG
print("X_train shape:", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape: ", y_test.shape)

print("Head of training data:")
print(X_train.head())
print(y_train.head())
print("Head of testing data:")
print(X_test.head())
print(y_test.head())

# Train (fit) the model on the training data
print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)

# Make predictions on the unseen test data
raw_predictions = xgb_model.predict(X_test)

# Apply Post-Prediction Clipping
# XGBoost might theoretically predict -0.01 or 1.05. We force the bounds to [0, 1]
#final_predictions = np.clip(raw_predictions, 0, 1)
final_predictions = raw_predictions

# Evaluate how well the model did
mae = mean_absolute_error(y_test, final_predictions)
rmse = np.sqrt(mean_squared_error(y_test, final_predictions))

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE):      {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Let's see what the model learned! (Feature Importance)
print("\n--- Feature Importance ---")
# Get the importance scores from the model
importances = xgb_model.feature_importances_
feature_names = X.columns

# Pair them up, sort them, and print them
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df.to_string(index=False))

# Find the directional importance
print("\n--- Calculating Directional Importance (SHAP) ---")

# Initialize the SHAP Explainer with our trained XGBoost model
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for our test data
# (This calculates exactly how much each feature pushed the prediction up or down)
shap_values = explainer.shap_values(X_test)

# --- METHOD A: Print a DataFrame with Direction ---

# Calculate the average magnitude of impact (Importance)
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Figure out the direction by correlating the feature's actual value with its SHAP value
directions = []
for i, col in enumerate(X_test.columns):
    # If the feature goes up and the SHAP value goes up, it's a positive correlation
    corr = np.corrcoef(X_test[col], shap_values[:, i])[0, 1]
    
    if np.isnan(corr) or abs(corr) < 0.1:
        direction = "Neutral / No Impact"
    elif corr > 0:
        direction = "Increases Risk 📈"
    else:
        direction = "Decreases Risk 📉"
    
    directions.append(direction)

# Create and print the new DataFrame
directional_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance (SHAP)': mean_abs_shap,
    'Direction': directions
}).sort_values(by='Importance (SHAP)', ascending=False)

print("\nDirectional Feature Importance:")
print(directional_df.to_string(index=False))


# --- METHOD B: The Famous SHAP Summary Plot ---
print("\nGenerating SHAP Visual Plot... (Close the popup window to continue)")

# This plot is the gold standard for explaining XGBoost to stakeholders
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
plt.tight_layout()
plt.show()