import json
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from pathlib         import Path
from typing          import Self


from NutriRisk.constants import folders

class DataManager():

    """
    This class takes a CSV with data and run the proper Data Preprocessing
    steps, which are:

    Feature Encoding:

        Converting categorical variables into numerical format (e.g., One-Hot
        Encoding, Label Encoding).        

    Data Splitting (Train/Test Split):

        Dividing the dataset into training and test subsets to evaluate how
        well the model generalizes to unseen data.

    Feature Scaling / Normalization:

        Transforming numerical features to a similar scale (e.g., Min-Max
        Normalization to [0,1] or Standardization/Z-score scaling to mean 0 and
        variance 1.

        This is done independently for each feature to ensure that all numerical
        features contribute equally to the model training process.

        This is also done independently for the test set to prevent data
        leakage.

    Data Cleaning:

        Handling missing values (imputation or removal), removing duplicates,
        and addressing outliers.

    """

    # Constructor for the base regression model class
    def __init__(self,
                 source_data_path: str = None,
                 target_folder:    str = None,
                 y_col_name:       str = None
                 ):

        # If no data path is provided, use the default random data file
        if source_data_path is None:
            source_data_path = folders.RANDOM_DATA_FILEPATH

        # Check that the source data path exists
        if not Path(source_data_path).exists():
            raise FileNotFoundError(f"Source data file not found: {source_data_path}")

        # If no target folder is provided, use the default prepared data folder
        if target_folder is None:
            target_folder = folders.PREPARED_DATA_FOLDER

        # Ensure the target folder exists
        if not Path(target_folder).exists():
            Path(target_folder).mkdir(parents = True, exist_ok = True)

        # If the column for the y is not provided, use the last column name
        if y_col_name is None:
            y_col_name = pd.read_csv(source_data_path).columns[-1]

        # Set the attributes
        self.source_data_path = source_data_path
        self.target_folder    = target_folder
        self.y_col_name       = y_col_name

    def run_data_processing(self,
                            split_proportion:   float = 0.2,
                            min_theoreitcal_y:  float = 0.0,
                            max_theoreitcal_y:  float = 100.0,
                            right_skew_columns: list  = None,
                            left_skew_columns:  list  = None
                            ):


        # Load the dataset
        df = pd.read_csv(self.source_data_path)

        # Apply the skewness correction for left and right skewed columns
        #     - Right skewed columns are corrected using log1p transformation.
        if right_skew_columns is not None:
            for col in right_skew_columns:
                df[col] = np.log1p(df[col])

        #     - Left skewed columns are corrected using Yeo-Johnson transformation.
        pt = PowerTransformer(method="yeo-johnson") # Initialize Yeo-Johnson
        if left_skew_columns is not None:
            for col in left_skew_columns:
                df[col] = pt.fit_transform(df[[col]])

        # Separate the input features (X) from the target we want to predict (y)
        X = df.drop(columns=[self.y_col_name])
        y = df[self.y_col_name]

        # Split the data into training and test based on the asked proportion
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=split_proportion, random_state=42)

        # Later on we will also fill these datasets
        X_train_normalized  = None
        X_test_normalized   = None
        y_train_normalized  = None
        y_test_normalized   = None

        X_train_standarized = None
        X_test_standarized  = None
        # y_train_standarized = None  These two does not make sense to standarize
        # y_test_standarized  = None  because the values can then be greater than 1 after standardization


        # Save the following metadata information about the dataset splits
        # We will normalize the data later on, and when the models make predictions
        # they need to transform the raw input into the same normalized format.
        metadata = {
            "split_proportion":   split_proportion,
            "min_theoreitcal_y":  min_theoreitcal_y,
            "max_theoreitcal_y":  max_theoreitcal_y,
            "right_skew_columns": right_skew_columns,
            "left_skew_columns":  left_skew_columns,
            "X_train_shape":      X_train_raw.shape,
            "X_test_shape":       X_test_raw.shape,
            "y_train_shape":      y_train_raw.shape,
            "y_test_shape":       y_test_raw.shape,
            "min_x_test":         X_test_raw.min().to_dict(),
            "max_x_test":         X_test_raw.max().to_dict(),
            "mean_x_test":        X_test_raw.mean().to_dict(),
            "std_x_test":         X_test_raw.std().to_dict(),
            "min_x_train":        X_train_raw.min().to_dict(),
            "max_x_train":        X_train_raw.max().to_dict(),
            "mean_x_train":       X_train_raw.mean().to_dict(),
            "std_x_train":        X_train_raw.std().to_dict(),
            "min_y_test":         y_test_raw.min(),
            "max_y_test":         y_test_raw.max(),
            "mean_y_test":        y_test_raw.mean(),
            "std_y_test":         y_test_raw.std(),
            "min_y_train":        y_train_raw.min(),
            "max_y_train":        y_train_raw.max(),
            "mean_y_train":       y_train_raw.mean(),
            "std_y_train":        y_train_raw.std(),
        }
        # Save into a JSON file
        with open(Path(self.target_folder) / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Standarize the X input features (both training and test sets)
        #
        # Notice that some models later on uses the raw data rather than the
        # standarize data. Here is a quick summary of when to use standardized
        # inputs:
        #
        # ----------------------------------------------------------------------
        # | Algorithm        | Recommended Method |           Why?             |
        # ----------------------------------------------------------------------
        # |  Neural Networks |                    | Helps gradients flow stably
        # |  Deep Learning   |  Standarization    | and prevetns vanishing /
        # |                  |                    | exploding gradients.
        # ----------------------------------------------------------------------
        # |  Linear          |                    | Scale matters for
        # |  Logistic Regr.  |  Standarization    | regularization penalties and
        # |  SVM             |                    | gradient convergence.
        # ----------------------------------------------------------------------
        # |                  |                    | Avoids features with 
        # |  PCA (similar)   |  Standarization    | naturally larger numeric 
        # |                  |                    | scales dominating the
        # |                  |                    | principal components. 
        # ----------------------------------------------------------------------
        # |                  |                    | Equalizes feature importance 
        # |  KNN             |  Standarization    | but Min-Max is ruined if 
        # |  K-means         |  is safer          | outliers exist. 
        # ----------------------------------------------------------------------        
        # |  XGBoost         |                    | Decision trees only care
        # |  Random Forest   |      NEITHER!      | about value order (splits), 
        # |  LightGBM        |                    | not magnitude.
        # ----------------------------------------------------------------------         
        #
        # None of this is true when severe outliers are present. In such case
        # a RobustScaler is recommended (scaling by median and IQR)


        # Standarize the input features (X)
        mean = X_train_raw.mean()
        std  = X_train_raw.std()

        # Apply those same statistics to both sets
        X_train_standarized = (X_train_raw - mean) / std
        X_test_standarized  = (X_test_raw  - mean) / std

        # Normalize the input features (X) to the range [0, 1] using Min-Max scaling
        X_train_normalized = (X_train_raw - X_train_raw.min()) / (X_train_raw.max() - X_train_raw.min())
        X_test_normalized  = (X_test_raw  - X_train_raw.min()) / (X_train_raw.max() - X_train_raw.min())

        # Normalize the input images (y) to the range [0, 1]
        # But take this into account, which is later applied for each model
        #
        # Neural Networks (PyTorch / TensorFlow / Keras): Mandatory:
        #
        # Neural networks train much better when targets are in a small, stable
        # range (like [0,1]). Furthermore, because the targets in the nutrition
        # data are concentrated near 0% and 100% risk, we can place a Sigmoid
        # activation on the output neuron. Sigmoid outputs numbers between 0 and
        # 1 and naturally prefers pushing predictions toward the extreme
        # boundaries (0 and 1).
        #
        # Linear Regression / Ridge / Lasso: Optional.
        # It won't hurt, but it isn't strictly necessary since linear models
        # adapt their weights and bias to match the target's scale.
        #
        # Tree-based models (XGBoost, LightGBM, Random Forest): DONT DO IT
        # Trees split purely on the ordering of values. Scaling y changes
        # nothing mathematically and just creates extra work when interpreting
        # predictions.

        # Best approach: Fixed domain division
        y_train_normalized = y_train_raw / max_theoreitcal_y
        y_test_normalized  = y_test_raw  / max_theoreitcal_y

        # Save each split dataset to the target folder
        #    - RAW
        X_train_raw.to_csv(         Path(self.target_folder) / "X_train_raw.csv",         index = False)
        X_test_raw.to_csv(          Path(self.target_folder) / "X_test_raw.csv",          index = False)
        y_train_raw.to_csv(         Path(self.target_folder) / "y_train_raw.csv",         index = False)
        y_test_raw.to_csv(          Path(self.target_folder) / "y_test_raw.csv",          index = False)        
        #    - STANDARDIZED (no y present, this is correct)
        X_train_standarized.to_csv( Path(self.target_folder) / "X_train_standarized.csv", index = False)
        X_test_standarized.to_csv(  Path(self.target_folder) / "X_test_standarized.csv",  index = False)
        #    - NORMALIZED (both X and y)
        X_train_normalized.to_csv(  Path(self.target_folder) / "X_train_normalized.csv",  index = False)
        X_test_normalized.to_csv(   Path(self.target_folder) / "X_test_normalized.csv",   index = False)
        y_train_normalized.to_csv(  Path(self.target_folder) / "y_train_normalized.csv",  index = False)
        y_test_normalized.to_csv(   Path(self.target_folder) / "y_test_normalized.csv",   index = False)        


        print(f"Data processing complete. Files saved to {self.target_folder}")
        
    def plot_data_balance(self,
                          X: str        = None,
                          y: str        = None,
                          y_name: str   = "Target (y)",
                          bins: int     = 25,
                          filename: str = None):
        """
        Plots a minimal 3-column visual overview:

        Variable Name | Histogram (with min/max) | Balance (Skewness)
        """

        # Get the X and y from the target folder if they are not provided
        if X is None:
            X = pd.read_csv(Path(self.target_folder) / "X_train_raw.csv")
        if y is None:
            y = pd.read_csv(Path(self.target_folder) / "y_train_raw.csv").squeeze()

        # Combine X and y into a single DataFrame
        data = X.copy()
        if y is not None:
            if isinstance(y, pd.Series):
                data[y.name if y.name else y_name] = y
            else:
                data[y_name] = y

        # Only include numeric columns
        # I think this is not needed because up to this point categories are
        # one-shot encoded already. I keep it just in case.
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        n_vars = len(numeric_cols)

        # Setup figure: height scales dynamically with the number of variables
        fig, axes = plt.subplots(
            nrows   = n_vars,
            ncols   = 1,
            figsize = (10, max(2.5, n_vars * 0.9)),
            constrained_layout = True,
        )
        if n_vars == 1:
            axes = [axes]

        # Column Headers (attached above the first row)
        axes[0].text(
            -0.08,
            1.5,
            "Variable Name",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
        axes[0].text(
            0.50,
            1.5,
            "Distribution",
            transform=axes[0].transAxes,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
        axes[0].text(
            1.08,
            1.5,
            "Balance (Skewness)",
            transform=axes[0].transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

        # Plot each row
        for ax, col in zip(axes, numeric_cols):
            series     = data[col].dropna()
            vmin, vmax = series.min(), series.max()
            skew_val   = series.skew()

            # Plot minimal histogram
            ax.hist(
                series, bins=bins, color="#3470a3", edgecolor="white", alpha=0.85
            )

            # Set x-axis to strictly show min and max ticks
            if vmin == vmax:
                ax.set_xlim(vmin - 1, vmax + 1)
                ax.set_xticks([vmin])
            else:
                ax.set_xlim(vmin, vmax)
                ax.set_xticks([vmin, vmax])

            # Formatting ticks to 2 decimal places (or integer if applicable)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x, _: f"{int(x)}" if x.is_integer() else f"{x:.2f}"
                )
            )
            ax.tick_params(axis="x", labelsize=8, length=3)

            # Remove background spines and y-axis ticks for a clean sparkline look
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#cccccc")

            # Left Column: Variable Name (Target highlighted in bold/blue)
            is_target = col in [y_name, getattr(y, "name", None)]
            ax.text(
                -0.08,
                0.4,
                col,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold" if is_target else "normal",
                color="#0b4070" if is_target else "black",
            )

            # Right Column: Skewness value
            # Skew between -0.5 and 0.5 is considered well-balanced
            color = "green" if abs(skew_val) <= 0.5 else "darkorange"
            if abs(skew_val) > 1.0:
                color = "red"  # Heavily imbalanced

            ax.text(
                1.08,
                0.4,
                f"{skew_val:+.2f}",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

        # Save the plot in the target folder

        # If no filename is provided, use the default "plot"
        if filename is None:
            filename = "plot"

        plt.savefig(f"{self.target_folder}/{filename}.png", bbox_inches="tight")

        # Close the plot to free up memory
        plt.close()
        

