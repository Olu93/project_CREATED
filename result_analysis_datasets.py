# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
# https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/anova/how-to/mixed-effects-model/interpret-the-results/key-results/
# %%
PATH = pathlib.Path('readers/')

new_ds_names = {
    "OutcomeDice4ELReader":"Dice4EL",
    "OutcomeBPIC12ReaderShort": "BPIC12-25",
    "OutcomeBPIC12ReaderMedium": "BPIC12-50",
    "OutcomeBPIC12ReaderFull": "BPIC12-Full",
}
readers = list(new_ds_names.keys())
# %%
list_jsons = []
for idx, rd in enumerate(readers):
    tmp_path = PATH / rd / "stats.json"
    data = json.load(tmp_path.open())
    data["time_preprocess"] = data.get("time").get("preprocess")
    data["time_unit"] = data.get("time").get("time_unit")
    del data["starting_column_stats"]
    del data["length_distribution"]
    del data["orig"]
    del data["time"]
    list_jsons.append(data)
df: pd.DataFrame = pd.json_normalize(list_jsons).set_index("class_name").rename(columns={"outcome_distribution.1": "num_regular", "outcome_distribution.0": "num_deviant"})
df = df.rename(index=new_ds_names)
df.columns = df.columns.map(lambda col: col.replace("_", " ").replace("_", " "))
df.index.names = ["dataset"]
df
# %%
print(df.T.style.format(escape="latex").to_latex())
# %%
