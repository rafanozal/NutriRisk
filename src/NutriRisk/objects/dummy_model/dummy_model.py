import shutil
import time

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import ClassVar

from sklearn.base import BaseEstimator, RegressorMixin


# Abstract base class for regression models
from NutriRisk.objects.regression_model.regression_model import RegressionModel

# The BaseEstimator and RegressorMixin are included to make the dummy model
# compatible with scikit-learn's interface

@dataclass(slots=True)
class DummyRegressionModel(RegressionModel, RegressorMixin, BaseEstimator):

    # Constructor for the Linear Regression model
    def __init__(self,
                 model_name:  str         = "dummy",
                 base_folder: str         = None,
                 data_folder: str         = None,
                 data_transformation: str = None,
                 overwrite: bool          = False):

        # Init the type of model
        self.model_type = "Dummy"

        # Failsafe for scikit-learn estimator type check:
        _estimator_type: ClassVar[str] = "regressor"  

        # Dataclass slots requirement: Declare fitted attributes here 
        # so slots=True allows them to be assigned in fit()
        is_fitted_:        bool   = False
        n_features_in_:    int    = 0
        feature_names_in_: object = None              

        # Call the parent class's __init__ method and pass the data and model name
        super().__init__(model_name          = model_name,
                         base_folder         = base_folder,
                         data_folder         = data_folder,
                         data_transformation = data_transformation,
                         overwrite           = overwrite)

    # --------------------------------------------------------------------------
    # Abstract methods
    # --------------------------------------------------------------------------
    def train(self):
        """
            Implement the training logic for the dummy model
            The dummy model does not train anything, just fill the metadata
            variables properly
        """

        # Get the training data
        files_path, files_existance = self.get_default_test_filenames()  
        x_train_exist = files_existance[0]
        y_train_exist = files_existance[2]
        x_train_path  = files_path[0]
        y_train_path  = files_path[2]
        X_train_data  = pd.read_csv(x_train_path) if x_train_exist else None
        y_train_data  = pd.read_csv(y_train_path) if y_train_exist else None

        # Training Date
        self.train_date = f"{time.localtime().tm_year}-{time.localtime().tm_mon:02d}-{time.localtime().tm_mday:02d}"

        # Training Time (duration in seconds)
        # This will run the fit() function for dummy, which does not do anything
        # in particular, just set the flags as fitted.
        start_training_time = time.localtime()
        self.fit(X_train_data, y = y_train_data)
        stop_training_time  = time.localtime()
        total_training_time = time.mktime(stop_training_time) - time.mktime(start_training_time)
        self.train_time     = total_training_time

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
            Implement the prediction logic for the dummy model
        """

        # The dummy model just returns a random float as prediction
        #
        # In this case, we need to return an array-like structure with the
        # same length of X
        #
        # The model attributes also tell us the theoretical y values range
        # so the random predictions should be within this range

        n_predictions = len(X)

        return self.min_theoretical_y + (self.max_theoretical_y - self.min_theoretical_y) * np.random.rand(n_predictions)

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the dummy model to satisfy scikit-learn estimators interface.

        Args:
            X (pd.DataFrame): Input features for fitting the model.
            y (optional): Target values (ignored by the dummy model).

        Returns:
            self: Returns the fitted model instance.
        """

        # Store feature count and names (prevents sklearn validation warnings/errors)
        if hasattr(X, "shape"):
            self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        # Scikit-learn checks for any attribute with a trailing '_' to confirm fitting
        self.is_fitted_ = True

        # Always return self
        return self

    def explain_with_shap_specific(self):
        """
        Implement the SHAP explanation logic for the dummy model

        This is just copying the agnostic SHAP explanation to the specific version.
        """

        # The dummy model does not have any real explanations,
        # so it just copy the agnostic version to the specific version

        shutil.copy(self.shap_data_agnostic_path, self.shap_data_specific_path)
        shutil.copy(self.shap_plot_agnostic_path, self.shap_plot_specific_path)
        
    def save(self):
        """
            Implement the saving logic for the dummy model
            For the dummy model, we just create an empty placeholder .txt file
            with some text inside.
        """

        # Make sure the parent directory exists before creating the file
        self.model_core_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Fill the text content
        model_content = "This is a dummy model placeholder."

        # Open the file in write mode to create it
        with open(self.model_core_filepath, "w") as f:
            f.write(model_content)

        # The dummy model does not have any actual parameters to save

        # Update the model attribute
        # We use self so scikit-learn compatible methods can access the model instance
        # This ensures that any method expecting a fitted model can access it through self.model
        self.model = self

    def load(self) -> int:

        """Implement the loading logic for the dummy model

        Returns:
            int: Status code indicating success (0) or failure (non-zero).
        """

        print("Loading dummy model...")
    
        # Set the model to self object, so the fit and predict methods
        # can run from the model object.

        self.model = self

        return 0

    def get_model_file(self) -> str:
        """
        Returns the filename of the core model file for the dummy model.
        """
        return "model.txt"