# %%
import glob
import io
import pathlib
import os
from thesis_commons.constants import PATH_RESULTS_MODELS_OVERALL, PATH_RESULTS_MODELS_SPECIFIC
import pandas as pd
# %%

if __name__ == '__main__':
    config_name = "datasets"
    all_directories = glob.glob(str((PATH_RESULTS_MODELS_OVERALL/ 'overall' /"*.csv").absolute()))
    all_csvs = []
    for directory in all_directories:        
        try:
            dirpath = pathlib.Path(directory)
            print(dirpath.absolute())
            df = pd.read_csv(directory)
            filename = dirpath.name
            experiment_name = dirpath.parent.name
            df["filename"] = filename
            df["experiment_name"] = experiment_name
            all_csvs.append(df)
        except Exception as e:
            print(f"ERROR: Could not open {dirpath} due to Exception: {e}")
    major_df = pd.concat(all_csvs)
    PATH = PATH_RESULTS_MODELS_OVERALL/"overall"
    if not PATH.exists():
        os.makedirs(PATH)
    major_df.to_csv(PATH/ f"experiment_{config_name}_overall.csv")    
    print(major_df)



