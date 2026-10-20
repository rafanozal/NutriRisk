import shutil
import time
import json
import shap
import pandas            as pd
import numpy             as np
import xgboost           as xgb
import matplotlib.pyplot as plt
from dataclasses         import dataclass

# Abstract base class for regression models
from NutriRisk.objects.regression_model.regression_model import RegressionModel

@dataclass(slots=True)
class XGBoostRegressionModel(RegressionModel):

    # Class attributes
    objective:     str   = 'reg:squarederror'
    tree_method:   str   = 'hist'
    n_estimators:  int   = 100
    learning_rate: float = 0.1
    max_depth:     int   = 4
    random_state:  int   = 42

    # Constructor for the XGBoost Regression model
    def __init__(self,
                 model_name:  str         = "xgboost",
                 base_folder: str         = None,
                 data_folder: str         = None,
                 data_transformation: str = None,                # Tree methods do not require feature scaling
                 overwrite:     bool      = False,
                 objective:     str       = 'reg:squarederror',  # Standard regression task
                 tree_method:   str       = 'hist',              # Faster, modern tree building
                 n_estimators:  int       = 100,                 # Number of trees to build
                 learning_rate: float     = 0.1,                 # How much each tree updates the previous ones
                 max_depth:     int       = 4,                   # Maximum depth of each tree
                 random_state:  int       = 42):

        # Init the type of model
        self.model_type = "XGBoost"

        # Attributes of the XGBoost model
        self.objective     = objective
        self.tree_method   = tree_method
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.random_state  = random_state

        # Call the parent class's __init__ method and pass the data and model name
        parent_status = super().__init__(model_name          = model_name,
                                         base_folder         = base_folder,
                                         data_folder         = data_folder,
                                         data_transformation = data_transformation,
                                         overwrite           = overwrite)

        # print(f"Parent class initialization status: {parent_status}")
        # print("XGBoost model initialized.")


    # --------------------------------------------------------------------------
    # Abstract methods
    # --------------------------------------------------------------------------
    def train(self) -> xgb.XGBRegressor:

        """Implement the training logic for the XGBoost model"""

        print("Training XGBoost model...")

        # Get the training data
        files_path, files_existance = self.get_default_test_filenames()  
        x_train_exist = files_existance[0]
        y_train_exist = files_existance[2]
        x_train_path  = files_path[0]
        y_train_path  = files_path[2]
        X_train_data  = pd.read_csv(x_train_path) if x_train_exist else None
        y_train_data  = pd.read_csv(y_train_path) if y_train_exist else None


        # print("Training xgboost model with...")
        # print("    - Data paths:")
        # print(f"        X_train_data path: {x_train_path}")
        # print(f"        y_train_data path: {y_train_path}")
        # print(f"        X_test_data path:  {x_test_path}")
        # print(f"        y_test_data path:  {y_test_path}")
        # print("    - Data shapes:")
        # print(f"        X_train_data shape: {X_train_data.shape if X_train_data is not None else 'None'}")
        # print(f"        y_train_data shape: {y_train_data.shape if y_train_data is not None else 'None'}")
        # print(f"        X_test_data shape:  {X_test_data.shape  if X_test_data  is not None else 'None'}")
        # print(f"        y_test_data shape:  {y_test_data.shape  if y_test_data  is not None else 'None'}")
        # print("    - Model parameters:")
        # print(f"        objective:     {self.objective}")
        # print(f"        tree_method:   {self.tree_method}")
        # print(f"        n_estimators:  {self.n_estimators}")
        # print(f"        learning_rate: {self.learning_rate}")
        # print(f"        max_depth:     {self.max_depth}")
        # print(f"        random_state:  {self.random_state}")


        # Training Date
        self.train_date = f"{time.localtime().tm_year}-{time.localtime().tm_mon:02d}-{time.localtime().tm_mday:02d}"

        # Training Time (duration in seconds)
        start_training_time = time.localtime()

        # Initialize the XGBoost Regressor
        # We use 'hist' (histogram) as it is the modern, highly-optimized default for XGBoost
        xgb_model = xgb.XGBRegressor(
            objective     = self.objective,
            tree_method   = self.tree_method,
            n_estimators  = self.n_estimators,
            learning_rate = self.learning_rate,
            max_depth     = self.max_depth,                  
            random_state  = self.random_state
        )

        # Train (fit) the model on the training data
        xgb_model.fit(X_train_data, y_train_data)

        # Log the training time
        stop_training_time  = time.localtime()
        total_training_time = time.mktime(stop_training_time) - time.mktime(start_training_time)
        self.train_time     = total_training_time

        # Set the model
        self.model = xgb_model

        # Return the model
        return xgb_model
    
    def explain_with_shap_specific(self):
        """
        XGBoost-specific SHAP explanation using TreeExplainer.

        Much faster and computes exact Shapley values.
        """

        print(f"\nCalculating TreeSHAP for XGBoost ({self.model_name})...")

        # Find the x test data in the data folder
        files_path, _      = self.get_default_test_filenames()
        target_x_test_path = files_path[1]
        X_test_data        = pd.read_csv(target_x_test_path)

        # Initialize TreeExplainer
        # Pass the trained XGBRegressor/XGBClassifier instance
        explainer = shap.TreeExplainer(
            self.model,
            feature_perturbation = "tree_path_dependent"
        )

        # Calculate SHAP values
        # check_additivity=False can be set if minor floating point mismatches occur in deep trees
        shap_explanation = explainer(X_test_data, check_additivity=True)

        # shap_explanation.values has shape (n_samples, n_features) for regression
        shap_values = shap_explanation.values

        # Handle multi-output / multiclass if necessary
        # (shape becomes 3D: [samples, features, classes])
        if len(shap_values.shape) == 3:
            # e.g., take the target class or positive class
            shap_values = shap_values[:, :, 1]

        # Calculate Directional Importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        directions = []

        for i, col in enumerate(X_test_data.columns):
            # Avoid warnings on constant features
            col_std = np.std(X_test_data[col].values)
            val_std = np.std(shap_values[:, i])

            if col_std == 0 or val_std == 0:
                directions.append(0)
                continue

            corr = np.corrcoef(X_test_data[col], shap_values[:, i])[0, 1]
            if np.isnan(corr) or abs(corr) < 0.1:
                direction = 0
            elif corr > 0:
                direction = 1
            else:
                direction = -1
            directions.append(direction)

        # Create DataFrame & Save
        directional_df = pd.DataFrame(
            {
                "Feature":           X_test_data.columns,
                "Importance (SHAP)": mean_abs_shap,
                "Direction":         directions,
            }
        ).sort_values(by="Importance (SHAP)", ascending=False)

        # Save CSV
        directional_df.to_csv(self.shap_data_specific_path, index=False)

        # Generate and Save Plot
        print(f"Generating TreeSHAP Visual Plot for {self.model_name}...")
        plt.figure()  # Clear/initialize figure to avoid overlapping previous plots
        shap.summary_plot(
            shap_values, X_test_data, feature_names=X_test_data.columns, show=False
        )
        plt.title(f"TreeSHAP Summary: {self.model_name}")
        plt.tight_layout()
        plt.savefig(self.shap_plot_specific_path)
        plt.close()

        # We also make the SHAP Dependence Plots in here
        for i, col in enumerate(X_test_data.columns):
            if np.abs(shap_values[:, i]).mean() > 0.01:  # Only plot if the feature has some importance
                plt.figure()
                shap.dependence_plot(
                    i, shap_values, X_test_data, show=False
                )
                plt.title(f"SHAP Dependence: {col} - {self.model_name}")
                plt.tight_layout()
                plt.savefig(self.model_path / f"SHAP_dependence_{col}.png")
                plt.close()

    def save(self):
        """
            Saves the XGBoost model and its hyperparameters to disk.
            The model is saved in JSON format, and the hyperparameters are
            saved in a separate JSON file.
        """

        print("Saving XGBoost model...")

        # Implement the saving logic for the xgboost model
        model_file = self.model_core_folder / f"model.json"
        model_file.parent.mkdir(parents=True, exist_ok=True)

        # # DEBUG
        # print(self.model_path)
        # print(model_file)
        # print(model_file.parent)

        self.model.save_model(model_file)

        # Save the hyperparameters of the model
        hyperparams_file = self.model_path / f"model/hyperparameters.json"
        with open(hyperparams_file, "w") as f:
            json.dump(self.model.get_params(), f)

    def load(self) -> int:
        """
        Loads the XGBoost model using its native JSON format

        Returns:
            int: Status code indicating success (0) or failure (non-zero).
        """

        print("Loading XGBoost model...")

        return_status = -1

        # Check that the model file exists before attempting to load it
        if self.model_core_filepath.exists():
            
            # Even when loading, we must instantiate the XGBRegressor object first
            self.model = xgb.XGBRegressor()
            self.model.load_model(self.model_core_filepath)
            return_status = 0

        return return_status

    def get_model_file(self) -> str:
        """
        Returns the filename of the core model file for the XGBoost model.
        """
        return "model.json"
