from NutriRisk.objects.data_manager.data_manager import DataManager

if __name__ == "__main__":

    # Run the data processing steps without transformations first
    data_manager = DataManager(y_col_name = "risk")
    data_manager.run_data_processing()
    data_manager.plot_data_balance(filename="initial_plot")

    # It looks like the columns "sugar_g" and "exercise_hrs_wk" are right skewed
    # require transformations.
    data_manager.run_data_processing(
        right_skew_columns=["sugar_g", "exercise_hrs_wk"]
    )
    data_manager.plot_data_balance(filename="transformed_plot")
