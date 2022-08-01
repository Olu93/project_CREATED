# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import io
from IPython.display import display
from jupyter_constants import PATH_PAPER_TABLES, save_table
# https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/anova/how-to/mixed-effects-model/interpret-the-results/key-results/
# %%
PATH = pathlib.Path('readers/')

new_ds_names = {
    "OutcomeDice4ELReader":"Dice4EL",
    "OutcomeBPIC12Reader25": "BPIC12-25",
    "OutcomeBPIC12Reader50": "BPIC12-50",
    "OutcomeBPIC12Reader75": "BPIC12-75",
    "OutcomeBPIC12Reader100": "BPIC12-100",
    "OutcomeBPIC12ReaderFull": "BPIC12-Full",
    "OutcomeSepsis1Reader": "Sepsis",
    "OutcomeTrafficFineReader": "TraficFines",
}
readers = list(new_ds_names.keys())
# %%
list_jsons = []
for idx, rd in enumerate(readers):
    tmp_path = PATH / rd / "stats.json"
    data = json.load(tmp_path.open())
    data["preprocessing_time"] = data.get("time").get("preprocess")
    data["time_unit"] = data.get("time").get("time_unit")
    data["num_outcome_regular"] = data["outcome_distribution"].get("regular")
    data["num_outcome_deviant"] = data["outcome_distribution"].get("deviant")
    del data["outcome_distribution"]
    del data["starting_column_stats"]
    del data["length_distribution"]
    del data["orig"]
    del data["time"]
    list_jsons.append(data)
df: pd.DataFrame = pd.json_normalize(list_jsons).set_index("class_name").rename(columns={"outcome_distribution.1": "num_regular", "outcome_distribution.0": "num_deviant"})
df = df.rename(index=new_ds_names)
df.columns = df.columns.map(lambda col: " ".join([string.title() for string in col.split("_")]))
df.index.names = ["Dataset"]
df
# %%
caption = "All datasets used within the evaluation. Dice4EL is used for the qualitative evaluation and the remaining are used for quantitative evaluation purposes."
df_styled = df.T.style.format(
        # escape='latex',
        precision=0,
        na_rep='',
        thousands=" ",
    )

df_latex = df_styled.to_latex(
        multicol_align='l',
        # column_format='l',
        caption=caption,
        label=f"tbl:dataset-stats",
        hrules=True,
    )

display(df_styled)
display(df_latex)
# save_table("dataset_stats")
# %%
