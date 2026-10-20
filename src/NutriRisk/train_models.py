from NutriRisk.objects.dummy_model.dummy_model     import DummyRegressionModel
from NutriRisk.objects.xgboost_model.xgboost_model import XGBoostRegressionModel
from NutriRisk.objects.tabpfn.tabpfn_model         import TabPFNRegressionModel
from NutriRisk.constants import folders

if __name__ == "__main__":

    # Initialize the models

    print("Initializing Dummy Regression Model...")
    # dummy_model = DummyRegressionModel(base_folder = folders.RESULT_FOLDER,
    #                                    data_folder = folders.PREPARED_DATA_FOLDER,
    #                                    overwrite   = True)

    print("Initializing XGBoost Regression Model...")
    xgboost_model = XGBoostRegressionModel(base_folder = folders.RESULT_FOLDER,
                                           data_folder = folders.PREPARED_DATA_FOLDER,
                                           overwrite   = False)

    print("Initializing TabPFN Regression Model...")
    tabpfn_model = TabPFNRegressionModel(base_folder     = folders.RESULT_FOLDER,
                                         data_folder     = folders.PREPARED_DATA_FOLDER,
                                         overwrite       = True,
                                         checkpoint_path = folders.TABPFN_MODEL_FILEPATH,
                                         ignore_pretraining_limits = True)
