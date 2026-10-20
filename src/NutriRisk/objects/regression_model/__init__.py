from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
import matplotlib.pyplot as plt
import warnings

# Suppress SHAP warnings for cleaner output
warnings.filterwarnings("ignore")

class BaseRiskModel(ABC):

    # Constructor
    def __init__(self, X_train, X_test, y_train, y_test, model_name = "Base Model"):
        """
        Initializes the base class with training and testing data.
        Every child class will inherit these attributes.
        """
        self.X_train    = X_train
        self.X_test     = X_test
        self.y_train    = y_train
        self.y_test     = y_test
        self.model_name = model_name
        self.model      = None # Will be instantiated by the child class

    @abstractmethod
    def train(self):
        """
        Every specific algorithm class MUST implement this method.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Every specific algorithm class MUST implement this method.
        Should return raw predictions.
        """
        pass

    def evaluate(self):
        """
        A common evaluation method for all models.
        It automatically clips predictions to [0, 1] and prints metrics.
        """
        print(f"\n{'='*40}")
        print(f"Evaluating {self.model_name}")
        print(f"{'='*40}")
        
        # Get raw predictions using the child's predict method
        raw_preds = self.predict(self.X_test)
        
        # Enforce the 0 to 1 bounds
        final_preds = np.clip(raw_preds, 0, 1)
        
        # Calculate metrics
        mae  = mean_absolute_error(self.y_test, final_preds)
        rmse = np.sqrt(mean_squared_error(self.y_test, final_preds))
        
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        return mae, rmse

    def explain_with_shap_generic(self, num_background_samples=100):
        """
        A model-agnostic SHAP explanation. Works on ANY algorithm.
        """
        print(f"\nCalculating Agnostic SHAP for {self.model_name}...")
        
        # For non-tree models, SHAP needs a "background" dataset to compare against.
        # We sample it to keep computation fast.
        background = shap.sample(self.X_train, num_background_samples)
        
        # By passing self.predict, SHAP treats the model as a black box.
        # This guarantees it works for XGBoost, Linear Regression, Neural Nets, etc.
        explainer = shap.Explainer(self.predict, background)
        
        # Calculate SHAP values on the test set
        shap_explanation = explainer(self.X_test)
        
        # Extract the raw numpy arrays for our directional calculation
        shap_values = shap_explanation.values
        
        # Calculate Directional Importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        directions = []
        
        for i, col in enumerate(self.X_test.columns):
            corr = np.corrcoef(self.X_test[col], shap_values[:, i])[0, 1]
            if np.isnan(corr) or abs(corr) < 0.1:
                direction = "Neutral / No Impact"
            elif corr > 0:
                direction = "Increases Risk 📈"
            else:
                direction = "Decreases Risk 📉"
            directions.append(direction)

        # Create DataFrame
        directional_df = pd.DataFrame({
            'Feature': self.X_test.columns,
            'Importance (SHAP)': mean_abs_shap,
            'Direction': directions
        }).sort_values(by='Importance (SHAP)', ascending=False)

        print(f"\nDirectional Feature Importance ({self.model_name}):")
        print(directional_df.to_string(index=False))

        # Generate and show the plot
        print(f"\nGenerating SHAP Visual Plot for {self.model_name}...")
        shap.summary_plot(shap_values, self.X_test, feature_names=self.X_test.columns, show=False)
        plt.title(f"SHAP Summary: {self.model_name}")
        plt.tight_layout()
        plt.show()

    def explain_with_shap_specific(self, num_background_samples=100):

        pass