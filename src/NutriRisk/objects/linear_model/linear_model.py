from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Abstract base class for regression models
from NutriRisk.objects.regression_model.regression_model import RegressionModel

class LinearRegressionModel(RegressionModel):

    # Constructor for the Linear Regression model
    def __init__(self,
                 model_name: str          = "Linear",
                 X_train:    pd.DataFrame = None,
                 X_test:     pd.DataFrame = None,
                 y_train:    pd.DataFrame = None,
                 y_test:     pd.DataFrame = None,
                 model:      str          = None):

        # Call the parent class's __init__ method and pass the data and model name
        super().__init__(model_name = model_name,
                         X_train    = X_train,
                         X_test     = X_test,
                         y_train    = y_train,
                         y_test     = y_test,
                         model      = model)
        
        # If the model file is not specified
        if(model == None):

            # Initialize the specific scikit-learn model
            self.model = LinearRegression()

            # Train the model
            self.train()

            # Save it into the folder

            # Evaluate
            self.evaluate()

            # Run the SHAP methods
            self.explain_with_shap_agnostic()

        # Otherwise load whatever we had already
        else:
            # Load the model from disk
            pass

    def train(self):
        """
        Satisfies the abstract method requirement.
        Fits the Linear Regression model to the training data.
        """
        print(f"Training {self.model_name}...")
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        """
        Satisfies the abstract method requirement.
        Returns the raw predictions for the given data.
        """
        return self.model.predict(X)