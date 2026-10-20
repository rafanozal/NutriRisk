import shutil
import time
import json
import shap
import time
import joblib
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tabpfn      import TabPFNRegressor







# Abstract base class for regression models
from NutriRisk.objects.regression_model.regression_model import RegressionModel

@dataclass(slots=True)
class TabPFNRegressionModel(RegressionModel):

    # Class attributes
    device:                    str        = 'auto'   # 'auto', 'cuda', or 'cpu'
    n_estimators:              int        = 8        # Ensembles/forward passes (8 is standard for TabPFN-v2)
    random_state:              int        = 42
    checkpoint_path:           str        = None     # Optional path to local 'tabpfn-v2-regressor.ckpt'
    ignore_pretraining_limits: bool       = False    # Allows running datasets > 10,000 rows or > 500 features

    # Constructor for the TabPFN Regression model
    def __init__(self,
                 model_name:                str        = "tabpfn",
                 base_folder:               str        = None,
                 data_folder:               str        = None,
                 data_transformation:       str        = None,   # TabPFN internally normalizes & encodes features
                 overwrite:                 bool       = False,
                 device:                    str        = 'auto', # 'auto', 'cuda', or 'cpu'
                 n_estimators:              int        = 8,      # Number of ensemble evaluation passes
                 random_state:              int        = 42,
                 checkpoint_path:           str        = None,   # Path to downloaded TabPFN-v2 checkpoint
                 ignore_pretraining_limits: bool       = False): # Bypass max sample/feature limits

        # Init the type of model
        self.model_type = "TabPFN"

        # Attributes of the TabPFN model
        self.device                    = device
        self.n_estimators              = n_estimators
        self.random_state              = random_state
        self.checkpoint_path           = checkpoint_path
        self.ignore_pretraining_limits = ignore_pretraining_limits

        # Call the parent class's __init__ method and pass the data and model name
        parent_status = super().__init__(model_name          = model_name,
                                         base_folder         = base_folder,
                                         data_folder         = data_folder,
                                         data_transformation = data_transformation,
                                         overwrite           = overwrite)

        # print(f"Parent class initialization status: {parent_status}")
        # print("TabPFN model initialized.")


    # --------------------------------------------------------------------------
    # Abstract methods
    # --------------------------------------------------------------------------
    def train(self) -> TabPFNRegressor:

        """Implement the training logic for the TabPFN model"""

        print("Training TabPFN model...")

        # Get the training data
        files_path, files_existance = self.get_default_test_filenames()
        x_train_exist = files_existance[0]
        y_train_exist = files_existance[2]
        x_train_path  = files_path[0]
        y_train_path  = files_path[2]
        X_train_data  = pd.read_csv(x_train_path) if x_train_exist else None
        y_train_data  = pd.read_csv(y_train_path) if y_train_exist else None

        # TabPFN expects y to be a 1D vector (Series or 1D array)
        if y_train_data is not None and isinstance(y_train_data, pd.DataFrame):
            y_train_data = y_train_data.squeeze()

        # Training Date
        self.train_date = f"{time.localtime().tm_year}-{time.localtime().tm_mon:02d}-{time.localtime().tm_mday:02d}"

        # Training Time (duration in seconds)
        start_training_time = time.localtime()

        # Build constructor kwargs
        tabpfn_kwargs = {
            "device":                    self.device,
            "n_estimators":              self.n_estimators,
            "random_state":              self.random_state,
            "model_path":                self.checkpoint_path,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
        }

        # Initialize the TabPFN Regressor
        tabpfn_model = TabPFNRegressor(**tabpfn_kwargs)

        # "Train" (fit) the model on the training data
        # Note: TabPFN is an in-context learning model,
        # so fit() registers the context/data
        tabpfn_model.fit(X_train_data, y_train_data)

        # Log the training time
        stop_training_time  = time.localtime()
        total_training_time = time.mktime(stop_training_time) - time.mktime(start_training_time)
        self.train_time     = total_training_time

        # Set the model
        self.model = tabpfn_model

        # Return the model
        return tabpfn_model

    def explain_with_shap_specific(self):
        """
        TabPFN SHAP explanation using a background-masked model Explainer.
        """

        background_samples: int = 50
        max_evals: int = 500

        print(f"\nCalculating SHAP for TabPFN ({self.model_name})...")

        # 1. Retrieve data paths
        files_path, _      = self.get_default_test_filenames()
        x_train_path       = files_path[0]
        x_test_path        = files_path[1]
        y_train_path       = files_path[2]
        y_test_path        = files_path[3]

        X_train_data = pd.read_csv(x_train_path)
        y_train_data = pd.read_csv(y_train_path)
        X_test_data  = pd.read_csv(x_test_path)
        y_test_data  = pd.read_csv(y_test_path)

    

        # 2. Build a representative background distribution
        # Transformer-based SHAP relies on reference points to mask absent features.
        if X_train_data.shape[0] > background_samples:
            background_data = shap.sample(X_train_data, background_samples, random_state=42)
        else:
            background_data = X_train_data

        masker = shap.maskers.Independent(background_data)

        # 3. Initialize Explainer using TabPFN's predict function
        explainer = shap.Explainer(
            self.model.predict,
            masker=masker,
            feature_names=X_test_data.columns
        )

        # 4. Compute SHAP explanations
        # max_evals limits inference passes through the transformer attention layers
        shap_explanation = explainer(X_test_data, max_evals=max_evals)

        # Extract values (n_samples, n_features)
        shap_values = shap_explanation.values

        # Handle 3D multi-output shape if applicable
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 0]

        # 5. Calculate Directional Importance
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

        # 6. Create DataFrame & Save CSV
        directional_df = pd.DataFrame(
            {
                "Feature":           X_test_data.columns,
                "Importance (SHAP)": mean_abs_shap,
                "Direction":         directions,
            }
        ).sort_values(by="Importance (SHAP)", ascending=False)

        directional_df.to_csv(self.shap_data_specific_path, index=False)

        # 7. Generate and Save Summary Plot
        print(f"Generating SHAP Visual Plot for {self.model_name}...")
        plt.figure()
        shap.summary_plot(
            shap_values, X_test_data, feature_names=X_test_data.columns, show=False
        )
        plt.title(f"TabPFN SHAP Summary: {self.model_name}")
        plt.tight_layout()
        plt.savefig(self.shap_plot_specific_path)
        plt.close()

        # 8. Generate and Save SHAP Dependence Plots
        for i, col in enumerate(X_test_data.columns):
            if np.abs(shap_values[:, i]).mean() > 0.01:
                plt.figure()
                shap.dependence_plot(
                    i, shap_values, X_test_data, show=False
                )
                plt.title(f"SHAP Dependence: {col} - {self.model_name}")
                plt.tight_layout()
                plt.savefig(self.model_path / f"SHAP_dependence_{col}.png")
                plt.close()


        base_loss = compute_metric(y_test_data, self.model.predict(X_test_data))
        loco_importance = {}

        for col in X_train_data.columns:
            loco_model = TabPFNRegressor(device=self.device)
            loco_model.fit(X_train_data.drop(columns=[col]), y_train_data)
            drop_loss = compute_metric(y_test_data, loco_model.predict(X_test_data.drop(columns=[col])))
            loco_importance[col] = drop_loss - base_loss                

        # Save the LOCO importance to a JSON file
        loco_importance_path = self.model_path / "loco_importance.json"
        with open(loco_importance_path, "w") as f:
            json.dump(loco_importance, f, indent=4)

    def save(self):

        print("Saving TabPFN model...")

        # Ensure the /model folder exists
        self.model_core_folder.mkdir(parents=True, exist_ok=True)

        # Save the model artifact to self.model_core_filepath 
        # (e.g. /out/results/tabpfn_cancer_data/model/model.pt)
        if hasattr(self.model, "save_fit_state"):
            self.model.save_fit_state(self.model_core_filepath)
        else:
            joblib.dump(self.model, self.model_core_filepath)

        # Save the hyperparameters inside self.model_core_folder
        # (e.g. /out/results/tabpfn_cancer_data/model/hyperparameters.json)
        hyperparams_file = self.model_core_folder / "hyperparameters.json"
        with open(hyperparams_file, "w") as f:
            json.dump(self.model.get_params(), f, indent=4, default=str)    

    def load(self) -> int:
        """
        Loads the TabPFN model state from disk

        Returns:
            int: Status code indicating success (0) or failure (non-zero).
        """

        print("Loading TabPFN model...")

        return_status = -1

        # Check that the model file exists before attempting to load it
        if self.model_core_filepath.exists():
            try:
                # TabPFN provides a native classmethod load_from_fit_state
                # that restores the in-context state onto the target device
                if hasattr(TabPFNRegressor, "load_from_fit_state"):
                    self.model = TabPFNRegressor.load_from_fit_state(
                        self.model_core_filepath,
                        device=self.device
                    )
                else:
                    # Fallback if joblib was used to save
                    self.model = joblib.load(self.model_core_filepath)

                return_status = 0

            except Exception as e:
                print(f"Error loading TabPFN model from {self.model_core_filepath}: {e}")
                return_status = -1

        else:
            print(f"Model file not found at: {self.model_core_filepath}")

        return return_status

    def get_model_file(self) -> str:
        """
        Defines the filename for the core TabPFN saved model.
        Used by the parent class to construct self.model_core_filepath.
        """
        return "model.tabpfn_fit"           