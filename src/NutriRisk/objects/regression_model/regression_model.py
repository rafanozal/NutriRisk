import json
import shap
import os
import warnings
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
from abc                 import ABC, abstractmethod
from sklearn.metrics     import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection  import PartialDependenceDisplay
from pathlib             import Path
from typing              import Self
from dataclasses         import dataclass

@dataclass(slots=True)
class RegressionModel(ABC):

    # Class attributes:
    # Init the variables to None in most cases

    #    - Metadata
    model_name:        str      = None # Name of the model
    model_type:        str      = None # Type of the model
    train_date:        str      = None # Training date of the model
    train_time:        str      = None # Training time of the model
    transformation:    str      = None # Data transformation that this model will use
    min_theoretical_y: float    = 0    # Define the range of the theoretical output values
    max_theoretical_y: float    = 1

    #    - Metrics
    mae:               float    = None # Mean Absolute Error of the model
    rmse:              float    = None # Root Mean Squared Error of the model
    r2:                float    = None # R-squared of the model

    #    - Folders and files
    base_folder:         str    = None # Folder that contain the model in disk
    model_path:          str    = None # Folder inside the base_folder
    model:               object = None # The actual model object
    metadata_path:       str    = None # JSON file with the metadata
    data_folder:         str    = None # Folder that contain the data in disk
    x_train_path:        str    = None # Path to the training data
    x_test_path:         str    = None # Path to the test data
    y_train_path:        str    = None # Path to the training labels
    y_test_path:         str    = None # Path to the test labels       
    metrics_path:        str    = None # JSON file with the MAE, etc...
    model_core_folder:   str    = None # Folder containing the core model file
    model_core_filename: str    = None # Name of the model file
    model_core_filepath: str    = None # Full path to the core model file

    #     - SHAP results
    shap_data_agnostic_path: str = None # CSV file with the SHAP agnostic results
    shap_data_specific_path: str = None # CSV file with the SHAP specific results
    shap_plot_agnostic_path: str = None # PNG file with the SHAP agnostic plot
    shap_plot_specific_path: str = None # PNG file with the SHAP specific plot


    # Constructor for the base regression model class
    def __init__(self,
                 model_name:  str         = None,
                 base_folder: str         = None,
                 data_folder: str         = None,
                 data_transformation: str = None,
                 overwrite: bool          = False
                 ) -> int:
        """
        This class serves as a blueprint for all regression models, ensuring
        that they follow a consistent structure and interface.

        If the model exist in the given folder, it will load it from disk.
        Otherwise it will train it from scratch and save into the selected
        folder.

        If the model folder does not exist, or the data folder does not contain
        the required files train and test datasets, this will result in an
        error.

        
        Attributes:

            model_name: (str) The name of the model. This is used for saving and
                        loading the model from disk. If set to None (default)
                        the name will be the same as the model type.

                        Examples of model names include:
                            "xgboost"
                            "linear"
                            "my_other_xgboost_test"

            base_folder: (str) Path to the folder where the model and its
                          related files are going to be stored. For example:

                          "/out/results"

                          Then each of the models will be saved according to the
                          model name like this:
                          
                          "/out/results/xgboost/"
                          "/out/results/linear/"
                          "/out/results/my_other_xgboost_test/"
                          "..."

                          Where each folder contains the files related to that
                          specific model with the same structure such as:

                          "/out/results/xgboost/metadata.json"
                          "/out/results/xgboost/metrics.json"
                          "/out/results/xgboost/shap_agnostic.json"
                          "/out/results/xgboost/shap_specific.json"
                          "/out/results/xgboost/model/<model_file>"

                          If set to None (default), the constructor will not
                          run.

            data_folder: (str) Path to the folder where the data is stored.
                         If set to None (default), the model cannot be trained,
                         but it can be loaded from disk.
                         
                         This folder needs to contain some of the following
                         files, depending on whether the model is being trained
                         with normalized data, standardized data, or raw data.
                         This selection is determined automatically based on the
                         `data_transformation` parameter.

                         "X_test_raw.csv"
                         "X_train_raw.csv"
                         "y_train_raw.csv"
                         "y_test_raw.csv"

                         "X_test_standarized.csv"
                         "X_train_standarized.csv"

                         "X_test_normalized.csv"
                         "X_train_normalized.csv"
                         "y_train_normalized.csv"
                         "y_test_normalized.csv"

             data_transformation: 
             
                (str) Data transformation that this model will use. Models such
                Neural Networks like, often use standarized data more. Whether
                the models are based on trees, they tend to not care about data
                transformation.
                
                In any case we need to keep track of what type of data the
                model is using so later on we apply the proper transformation
                to the input data, as well as to the explanation variables.
                
                Options are::

                    None (default) uses the RAW data.
                    "standarized"  uses mean and sd for scalling.
                    "normalized"   uses min and max for scaling.
                    any other value will default to None

            overwrite: (bool) Whether to overwrite existing model.
                        Defaults to False.

                        The class will try to load the model from disk if it
                        exists, if it does not exist it will try to train it
                        from scratch.

                        If this is set to True, the existing model will be
                        overwritten even if it exists on disk and train from
                        scratch regardless.

        """

        # Initialize the return status to indicate failure by default
        return_status = -1

        # In order to init the model from disk, both the model folder and model
        # name must be defined and exist in disk. Alongside with each of needed
        # files inside each model folder such as the model itself, JSON files,
        # etc., must also exist.
        #
        # These variables keep track that everything is in order
        base_folder_exist   = False
        model_path_exist    = False
        metadata_exist      = False
        metrics_exist       = False
        shap_agnostic_exist = False
        shap_specific_exist = False
        model_exist         = False
        data_folder_exist   = False
        x_train_exist       = False
        x_test_exist        = False
        y_train_exist       = False
        y_test_exist        = False

        # Init the instanciated object attributes from the provided parameters
        self.model_name     = model_name
        self.transformation = data_transformation
        if(self.transformation is not None):
            self.transformation = self.transformation.lower()

        # Check if the data folder is defined
        if(data_folder is not None):

            # If so, set the attribute for the data folder path
            self.data_folder    = Path(data_folder)

            # Check if the the folder exist
            if Path(data_folder).exists():

                # Set existence flag to true
                data_folder_exist = True

                # Check for the existence of the necessary data files
                files_path, files_existance = self.get_default_test_filenames()
                x_train_path  = files_path[0]
                x_test_path   = files_path[1]
                y_train_path  = files_path[2]
                y_test_path   = files_path[3]
                x_train_exist = files_existance[0]
                x_test_exist  = files_existance[1]
                y_train_exist = files_existance[2]
                y_test_exist  = files_existance[3]

                # Set the paths to the data files
                self.x_train_path = x_train_path
                self.x_test_path  = x_test_path
                self.y_train_path = y_train_path
                self.y_test_path  = y_test_path

                # Update the maximum and minimum theoretical y values

                # Get the metadata for the data folder, here it will indiciate
                # which are the min and max theoretical values.
                data_y_min = None
                data_y_max = None
                data_metadata_path = Path(data_folder) / "metadata.json"
                with open(data_metadata_path, "r") as f:
                    data_metadata = json.load(f)
                    data_y_max = data_metadata.get("max_theoreitcal_y")
                    data_y_min = data_metadata.get("min_theoreitcal_y")

                #print("debug:")
                #print(f"data boundaries data_y_min={data_y_min}, data_y_max={data_y_max}")

                # If the transformation is None, standarized, or any other case
                # then we use the raw values.
                #
                # Otherwise, if it is normalized min and max will be 0 and 1

                match self.transformation:

                    case "standarized":
                        self.max_theoretical_y = data_y_max
                        self.min_theoretical_y = data_y_min

                    case "normalized":

                        self.max_theoretical_y = 1
                        self.min_theoretical_y = 0

                    case None:
                        self.max_theoretical_y = data_y_max
                        self.min_theoretical_y = data_y_min

                    case _:                        
                        self.max_theoretical_y = data_y_max
                        self.min_theoretical_y = data_y_min

                #print("debug:")
                #print(f"final boundaries data_y_min={self.min_theoretical_y}, data_y_max={self.max_theoretical_y}")

        # Check if the base folder is provided and exists
        if(base_folder is not None):

            # Check if the the folder exist
            if Path(base_folder).exists():

                # Set the flag indicating that the base folder exists
                base_folder_exist = True

                # The folder that contain the folder with the model
                self.base_folder = Path(base_folder)

                # Check if the model name is defined
                if (model_name != None):

                    # The folder of the model, which is inside the base folder
                    self.model_path  = self.base_folder / model_name

                    # Paths to the various files related to the model
                    self.metadata_path           = self.model_path / "metadata.json"
                    self.metrics_path            = self.model_path / "metrics.json"
                    self.shap_data_agnostic_path = self.model_path / "shap_data_agnostic.csv"
                    self.shap_data_specific_path = self.model_path / "shap_data_specific.csv"
                    self.shap_plot_agnostic_path = self.model_path / "shap_plot_agnostic.png"
                    self.shap_plot_specific_path = self.model_path / "shap_plot_specific.png"

                    # Inside the model folder there is another folder that contains
                    # the actual model file is stored in a subfolder named "model"
                    self.model_core_folder       = self.model_path / f"model"

                    # Inside the core folder, is where the model is located
                    # This change depending on the specific model implementation so
                    # we ask the specific model implementation to define it.
                    self.model_core_filename     = self.get_model_file()

                    # The full path to the model file is constructed by combining the
                    # core folder and the core filename
                    self.model_core_filepath     = self.model_core_folder / self.model_core_filename

                    # Check if the model path actually exists
                    if self.model_path.exists():

                        # Set the flag indicating that the model path exists
                        model_path_exist = True

                        # Now check inside the model path for the
                        # mandatory files one by one
                        metadata_exist = self.metadata_path.exists()
                        metrics_exist  = self.metrics_path.exists()

                        # SHAP data is not mandatory. We can run predictions
                        # without the SHAP explanations, but we consider the model completed
                        # when this is included since it is included in the model evaluation
                        # process
                        shap_agnostic_exist = self.shap_data_agnostic_path.exists()
                        shap_specific_exist = self.shap_data_specific_path.exists()       

                        # Check if the actual model file exists within the core folder
                        model_exist = self.model_core_filepath.exists()

        # Now that everything is checked, determine if the model can be loaded
        # or trained
        #    - Init default
        can_be_loaded  = False
        can_be_trained = False
        #    - Determine if the model can be trained based on the existence of
        #      necessary files and folders
        can_be_trained = (base_folder_exist   and 
                          x_train_exist       and x_test_exist     and
                          y_train_exist       and y_test_exist
                         )
        can_be_loaded  = (
                         base_folder_exist   and model_path_exist    and
                         metadata_exist      and metrics_exist       and
                         shap_agnostic_exist and shap_specific_exist and
                         model_exist
                         )

        # Debug:
        # print(f"Can be loaded:       {can_be_loaded}")
        # print(f"Can be trained:      {can_be_trained}")
        # print(f"base_folder_exist:   {base_folder_exist}")
        # print(f"model_path_exist:    {model_path_exist}")
        # print(f"metadata_exist:      {metadata_exist}")
        # print(f"metrics_exist:       {metrics_exist}")
        # print(f"shap_agnostic_exist: {shap_agnostic_exist}")
        # print(f"shap_specific_exist: {shap_specific_exist}")
        # print(f"model_exist:         {model_exist}")
        
        # If overwrite is set to True, the model should not be loaded from disk
        can_be_loaded  = can_be_loaded and not overwrite

        # At this point we know if the model can be loaded from disk or not
        # If it can be loaded, proceed to initialize the model attributes
        if (can_be_loaded):

            # The model itself
            load_status = self.load()

            # If loaded correctly, continue to load metadata and metrics
            if(load_status == 0):

                # Load the metadata
                with open(self.metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.model_type        = metadata.get("model_type")
                    self.train_date        = metadata.get("train_date")
                    self.train_time        = metadata.get("train_time")
                    self.transformation    = metadata.get("transformation")
                    self.min_theoretical_y = metadata.get("y_min_theoretical")
                    self.max_theoretical_y = metadata.get("y_max_theoretical")

                # Load the metrics
                with open(self.metrics_path, "r") as f:
                    metrics = json.load(f)
                    self.mae  = metrics.get("MAE")
                    self.rmse = metrics.get("RMSE")
                    self.r2   = metrics.get("R2")

                return_status = 1

            else:
                print(f"Error: Failed to load the model from {self.model_core_filepath}")
                print(f"Load status: {load_status}")
                return_status = load_status

        # Otherwise, train a new model if possible
        else:

            # If it can be trained, then proceed to init a new model from scratch
            if can_be_trained:
    
                # Run the full pipeline for the model
                self.run_full_pipeline()

                return_status = 0

            # Otherwise, the model cannot be loaded or trained
            # Tell the user why.
            else:
                print("-" * 40)
                print(f"Model {model_name} cannot be loaded or trained.")
                print("-" * 40)

                if(base_folder is None):
                    print("Base folder is not specified.")
                else:
                    if(base_folder_exist == False):
                        print(f"Base folder {base_folder} does not exist.")

                if(model_name is None):
                    print("Model name is not specified.")
                else:
                    if(model_path_exist == False):
                        print(f"Model folder: \"{model_name}\" does not exist.")

                if (data_folder is None):
                    print("Data folder is not specified.")
                else:

                    if(data_folder_exist):
                        print(f"Data folder {data_folder} exists, but some files are missing.")
                        print()
                        print(f"You selected {data_transformation} method and...")

                        match data_transformation:

                            # None , use the RAW data
                            case None:

                                if (x_train_exist == False):
                                    print("X_train_raw.csv does not exist in the data folder.")
                                if (x_test_exist == False):
                                    print("X_test_raw.csv does not exist in the data folder.")
                                if (y_train_exist == False):
                                    print("y_train_raw.csv does not exist in the data folder.")
                                if (y_test_exist == False):
                                    print("y_test_raw.csv does not exist in the data folder.")

                            # Standarized data (only X_train and X_test are considered)
                            case "standarized":

                                if (x_train_exist == False):
                                    print("X_train_standarized.csv does not exist in the data folder.")
                                if (x_test_exist == False):
                                    print("X_test_standarized.csv does not exist in the data folder.")

                            # Normalized data
                            case "normalized":
                                if (x_train_exist == False):
                                    print("X_train_normalized.csv does not exist in the data folder.")
                                if (x_test_exist == False):
                                    print("X_test_normalized.csv does not exist in the data folder.")
                                if (y_train_exist == False):
                                    print("y_train_normalized.csv does not exist in the data folder.")
                                if (y_test_exist == False):
                                    print("y_test_normalized.csv does not exist in the data folder.")

                            # Default case is the same as None
                            case _: 
                                if (x_train_exist == False):
                                    print("X_train_raw.csv does not exist in the data folder.")
                                if (x_test_exist == False):
                                    print("X_test_raw.csv does not exist in the data folder.")
                                if (y_train_exist == False):
                                    print("y_train_raw.csv does not exist in the data folder.")
                                if (y_test_exist == False):
                                    print("y_test_raw.csv does not exist in the data folder.")       


                    else:
                        print(f"Data folder {data_folder} does not exist.")

        return return_status

    # --------------------------------------------------------------------------
    # Non abstract methods
    # --------------------------------------------------------------------------

    # Base on the data transformation, return the default test dataset filenames
    def get_default_test_filenames(self) -> tuple[tuple[str, str, str, str], tuple[bool, bool, bool, bool]]:

        # Filepaths will be stored here
        x_train_filepath = None
        x_test_filepath  = None
        y_train_filepath = None
        y_test_filepath  = None

        # Whether the files exist will be stored here
        x_train_exist    = False
        x_test_exist     = False
        y_train_exist    = False
        y_test_exist     = False

        match self.transformation:

            # None , use the RAW data
            case None:

                x_train_filepath = (self.data_folder) / "X_train_raw.csv"
                x_test_filepath  = (self.data_folder) / "X_test_raw.csv"
                y_train_filepath = (self.data_folder) / "y_train_raw.csv"
                y_test_filepath  = (self.data_folder) / "y_test_raw.csv"

            # Standarized data
            case "standarized":

                # Standarize y_train and y_test does not make sense
                # since the target can get sigmans beyond the 0,1 range
                # So instead of those, we use the y_raw files

                x_train_filepath = (self.data_folder) / "X_train_standarized.csv"
                x_test_filepath  = (self.data_folder) / "X_test_standarized.csv"
                y_train_filepath = (self.data_folder) / "y_train_raw.csv"
                y_test_filepath  = (self.data_folder) / "y_test_raw.csv"               

            # Normalized data
            case "normalized":

                x_train_filepath = (self.data_folder) / "X_train_normalized.csv"
                x_test_filepath  = (self.data_folder) / "X_test_normalized.csv"
                y_train_filepath = (self.data_folder) / "y_train_normalized.csv"
                y_test_filepath  = (self.data_folder) / "y_test_normalized.csv"

            # Default case is the same as None
            case _:                            

                x_train_filepath = (self.data_folder) / "X_train_raw.csv"
                x_test_filepath  = (self.data_folder) / "X_test_raw.csv"
                y_train_filepath = (self.data_folder) / "y_train_raw.csv"
                y_test_filepath  = (self.data_folder) / "y_test_raw.csv"


        # Evaluate the existance of these files
        if x_train_filepath.exists():
            x_train_exist = True
        if x_test_filepath.exists():
            x_test_exist  = True
        if y_train_filepath.exists():
            y_train_exist = True
        if y_test_filepath.exists():
            y_test_exist  = True

        # Return the two tuples indicating which files are selected and whether
        # they exist or not
        return (x_train_filepath, x_test_filepath, y_train_filepath, y_test_filepath), \
               (x_train_exist,    x_test_exist,    y_train_exist,    y_test_exist)

    # Evaluation method for the model
    def evaluate(self,
                 X_test_data: pd.DataFrame = None,
                 y_test_data: pd.DataFrame = None) -> tuple[float, float, float]:
        """
        A common evaluation method for all models.
        it will save the results as JSON in the evaluation.json file for each
        valid model
        """
        
        print(f"Evaluating {self.model_name}")

        # Select the default test dataset if none is provided
        target_x_test_path = ""
        target_y_test_path = ""

        # Check for the existence of the necessary data files
        files_path, files_existance = self.get_default_test_filenames()  
        target_x_test_path = files_path[1]
        target_y_test_path = files_path[3]
        x_test_exist       = files_existance[1]
        y_test_exist       = files_existance[3]


        # The function allows to give a custum test dataset, but otherwise the
        # model itself already has a pointer to the folder where the data is
        # stored, which shall contain the proper train and test datasets.
        if X_test_data is None:
            X_test_data = pd.read_csv(target_x_test_path)
            #print(f"Loaded X_test_data from {target_x_test_path}")
        if y_test_data is None:
            y_test_data = pd.read_csv(target_y_test_path)
            #print(f"Loaded y_test_data from {target_y_test_path}")

        # debug
        # print("Data shapes:")
        # print(f"X_test_data shape:  {X_test_data.shape  if X_test_data  is not None else 'None'}")
        # print(f"y_test_data shape:  {y_test_data.shape  if y_test_data  is not None else 'None'}")
        # print("Data heads:")
        # print("X_test_data head:")
        # print(X_test_data.head())
        # print("y_test_data head:")
        # print(y_test_data.head())

        # Get raw predictions using the child's predict method
        # This method automatically bound the predictions within the theoretical
        # range.
        final_preds = self.predict(X_test_data)
        
        # Enforce the output to 0 to 1 bounds if the normalization was applied
        # if self.transformation == "normalized":
        #     final_preds = np.clip(raw_preds, 0, 1)
        # else:
        #     final_preds = raw_preds

        # # debug
        # print("y test values:", y_test_data)
        # print("Final predictions:", final_preds)
        
        # Calculate metrics
        mae  = mean_absolute_error(y_test_data, final_preds)
        rmse = np.sqrt(mean_squared_error(y_test_data, final_preds))
        r2   = r2_score(y_test_data, final_preds)

        # Show on the terminal
        print(f"Mean Absolute Error (MAE):      {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R^2 Score:                      {r2:.4f}")

        # Save the evaluation results as JSON
        metrics_results = {
            "MAE":  mae,
            "RMSE": rmse,
            "R2":   r2
        }
        metrics_path = Path(self.model_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_results, f, indent=4)

        # Update the model's internal state with the latest evaluation metrics
        self.mae  = mae
        self.rmse = rmse
        self.r2   = r2

        # Return the values
        return [mae, rmse, r2]

    # SHAP generic explanation method
    def explain_with_shap_agnostic(self,
                                   num_background_samples: int  = 100,
                                   X_test_data:  pd.DataFrame   = None,
                                   X_train_data: pd.DataFrame   = None):
        """
        A model-agnostic SHAP explanation. Works on ANY algorithm.
        It will save two files
            - shap_agnostic.json with the values
            - shap_agnostic.png with the same information in a plot
        """

        print(f"\nCalculating Agnostic SHAP for {self.model_name}...")

        # Check for the existence of the necessary data files
        files_path, files_existance = self.get_default_test_filenames()  
        target_x_train_path = files_path[0]
        target_x_test_path  = files_path[1]
        target_y_test_path  = files_path[3]
        x_train_exist       = files_existance[0]
        x_test_exist        = files_existance[1]
        y_test_exist        = files_existance[3]

        # The function allows to give a custom test dataset, but otherwise the
        # model itself already has a pointer to the folder where the data is
        # stored, which shall contain the proper train and test datasets.

        if X_test_data is None:
            X_test_data      = pd.read_csv(target_x_test_path)
        if X_train_data is None:
            X_train_data     = pd.read_csv(target_x_train_path)
        
        # For non-tree models, SHAP needs a "background" dataset to compare against.
        # We sample it to keep computation fast.
        background = shap.sample(X_train_data, num_background_samples)
        
        # By passing self.predict, SHAP treats the model as a black box.
        # This guarantees it works for XGBoost, Linear Regression, Neural Nets, etc.
        explainer = shap.Explainer(self.predict, background)
        
        print(f"\n Explainer done")


        # Calculate SHAP values on the test set
        shap_explanation = explainer(X_test_data)
        
        # Extract the raw numpy arrays for our directional calculation
        shap_values = shap_explanation.values
        
        # Calculate Directional Importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        directions = []
        
        for i, col in enumerate(X_test_data.columns):
            print(f"Processing feature: {col}")
            corr = np.corrcoef(X_test_data[col], shap_values[:, i])[0, 1]
            if np.isnan(corr) or abs(corr) < 0.1:
                direction = 0
            elif corr > 0:
                direction = 1
            else:
                direction = -1
            directions.append(direction)

        # Create DataFrame
        directional_df = pd.DataFrame({
            'Feature':           X_test_data.columns,
            'Importance (SHAP)': mean_abs_shap,
            'Direction':         directions
        }).sort_values(by='Importance (SHAP)', ascending=False)

        # print(f"\nDirectional Feature Importance ({self.model_name}):")
        # print(directional_df.to_string(index=False))

        # Save the dataframe as CSV for the agnostic SHAP explanation
        directional_df.to_csv(self.shap_data_agnostic_path, index=False)

        # Generate the agnostic plot
        print(f"\nGenerating SHAP Visual Plot for {self.model_name}...")
        shap.summary_plot(shap_values, X_test_data, feature_names=X_test_data.columns, show=False)
        plt.title(f"SHAP Summary: {self.model_name}")
        plt.tight_layout()

        # Save the agnostic SHAP plot
        plt.savefig(self.shap_plot_agnostic_path)

    # Dependency Plots (PDP) and Individual Conditional Expectation (ICE)
    def run_pdp_and_ice(self):
        """
        Generates Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots
        for the features in the test dataset.
        """

        # Find the x test data in the data folder
        files_path, _      = self.get_default_test_filenames()
        target_x_test_path = files_path[1]
        X_test_data        = pd.read_csv(target_x_test_path)

        # Generate PDP and ICE plots for each feature
        for feature in X_test_data.columns:
            print(f"Generating PDP and ICE plots for feature: {feature}")
            PartialDependenceDisplay.from_estimator(
                self.model,
                X_test_data,
                [feature],
                kind="both",  # both PDP and ICE

                # Style for individual patient lines (ICE)
                ice_lines_kw={
                    "alpha":    0.04,          # Low opacity so overlapping lines don't block the view
                    "color":    "gray",        # Neutral color for background curves
                    "linewidth": 0.6,
                },
                # Style for the average line (PDP)
                pd_line_kw={
                    "color":    "red",         # Contrasting color
                    "linewidth": 2.5,
                    "linestyle": "-",
                }


            )
            plt.title(f"PDP and ICE for {feature}: {self.model_name}")
            plt.tight_layout()
            plot_path = self.model_path / f"pdp_ice_{feature}.png"
            plt.savefig(plot_path)
            plt.close()

    # If the model does not exist, init completely, which means training, 
    # evaluation, etc...
    def run_full_pipeline(self):
        
        print(f"Starting full pipeline for {self.model_name}...")

        # Make sure that the folder for saving the model exists
        os.makedirs(self.model_path, exist_ok=True)

        # Train
        self.train()

        # Save model the metadata
        metadata = {
            "model_type":        self.model_type,
            "model_name":        self.model_name,
            "train_date":        self.train_date,
            "train_time":        self.train_time,
            "transformation":    self.transformation,
            "min_theoretical_y": self.min_theoretical_y,
            "max_theoretical_y": self.max_theoretical_y
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f)

        # Save the trained model
        self.save()

        # Evaluate the model
        self.evaluate()

        # Run the SHAP explanations
        #self.explain_with_shap_agnostic()
        self.explain_with_shap_specific()

        # Run the PDP and ICE plots
        self.run_pdp_and_ice()

    # Predict the target values for the given input data
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the target values for the given input data.

        This method must automatically bound the predictions within the
        theoretical range.

        Args:
            X (pd.DataFrame): Input features for which predictions are to be made.

        Returns:
            float: Predicted target value.
        """

        # Make predictions on the unseen test data
        raw_predictions = self.model.predict(X)

        # print("Raw predictions:")
        # print(raw_predictions)

        # print("Theoretical bounds:")
        # print(f"min_theoretical_y: {self.min_theoretical_y}")
        # print(f"max_theoretical_y: {self.max_theoretical_y}")


        # Apply Post-Prediction Clipping
        # Some models such as the XGBoost might theoretically predict, for
        # example, -0.01 or 1.05 when the range is 0 to 1.  We force the bounds
        # to [0, 1] if the theoretical y can be out of range
        final_predictions = np.clip(raw_predictions,
                                    self.min_theoretical_y,
                                    self.max_theoretical_y)


        return final_predictions   

    # --------------------------------------------------------------------------
    # Abstract methods
    # --------------------------------------------------------------------------

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def explain_with_shap_specific(self):
        """
        A model-specific SHAP explanation.

        Each model is responsible to evaluate using it own method.

        It will save two files
            - shap_specific.json with the values
            - shap_specific.png with the same information in a plot
        """
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self) -> int:
        """
        Loads the model from its storage.

        Returns:
            int: Status code indicating success (0) or failure (non-zero).
        """
        pass

    @abstractmethod
    def get_model_file(self) -> str:
        """
        Returns the filename of the core model file.
        This allows each specific model implementation to define its own
        storage structure.
        """
        pass