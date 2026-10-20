import pandas as pd


from NutriRisk.constants import folders

from NutriRisk.objects.dummy_model.dummy_model import DummyRegressionModel
from NutriRisk.objects.xgboost_model.xgboost_model import XGBoostRegressionModel

if __name__ == "__main__":

    # Initialize the models
    # - Dummy
    dummy_model   = DummyRegressionModel(  base_folder = folders.RESULT_FOLDER)
    # - XGBoost
    xgboost_model = XGBoostRegressionModel(base_folder = folders.RESULT_FOLDER)

    print("Model types:")
    print(type(dummy_model))
    print(type(xgboost_model))

    # Create a random patient with the necessary features:
    random_patient = {
        "age":             20,
        "bmi":             20.5,
        "calories_kcal":   2000,
        "sugar_g":         75.1,
        "fiber_g":         20.5,
        "protein_g":       40.3,
        "fat_g":           90.8,
        "carbs_g":         200.0,
        "water_l":         4.5,
        "exercise_hrs_wk": 2.5
    }
    random_patient = pd.DataFrame([random_patient])

    # Make a prediction with the loaded dummy model
    prediction = dummy_model.predict(random_patient)
    print(f"Dummy model prediction: {prediction}")

    # Make a prediction with the loaded XGBoost model
    prediction = xgboost_model.predict(random_patient)
    print(f"XGBoost model prediction: {prediction}")