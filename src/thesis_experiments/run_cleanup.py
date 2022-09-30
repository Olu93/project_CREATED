import glob
import io
import pathlib
import os
from thesis_commons.constants import PATH_RESULTS_MODELS_SPECIFIC, PATH_RESULTS_MODELS_OVERALL
import pandas as pd
if __name__ == "__main__":
    all_directories = list(glob.glob(str((PATH_RESULTS_MODELS_SPECIFIC/"evolutionary_*/**/*.csv").absolute()), recursive=True))
    all_directories = all_directories + list(glob.glob(str((PATH_RESULTS_MODELS_SPECIFIC/"overall_*/**/*.csv").absolute()), recursive=True))
    
    all_csvs = []
    for directory in all_directories:
        dirpath = pathlib.Path(directory)
        df = pd.read_csv(directory)
        filename = dirpath.name
        wrapper_type = dirpath.parent.name
        experiment_name = dirpath.parent.parent.name 
        df["filename"] = filename
        df["wrapper_type"] = wrapper_type
        df["experiment_name"] = experiment_name
        all_csvs.append(df)
        
    major_df = pd.concat(all_csvs)
    major_df.to_csv("tmp_2.csv")    
    print(major_df)
    # all_specific_csvs =glob.glob(PATH_RESULTS_MODELS_SPECIFIC/"*.csv")