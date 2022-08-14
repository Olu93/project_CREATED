# %%
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import io
from jupyter_constants import PATH_PAPER_TABLES, save_table
from IPython.display import display
from jupyter_constants import *
# https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/anova/how-to/mixed-effects-model/interpret-the-results/key-results/
# %%
PATH = pathlib.Path('readers/')


readers = list(map_ds_names.keys())
# %%
all_length_distributions = {}
list_jsons = []
for idx, rd in enumerate(readers):
    tmp_path = PATH / rd / "stats.json"
    data = json.load(tmp_path.open())
    # data["preprocessing_time"] = data.get("time").get("preprocess")
    # data["time_unit"] = data.get("time").get("time_unit")
    data["num_regular"] = data["outcome_distribution"].get("regular")
    data["num_deviant"] = data["outcome_distribution"].get("deviant")
    all_length_distributions[rd] = data["orig"]["length_distribution"]
    del data["outcome_distribution"]
    del data["starting_column_stats"]
    del data["length_distribution"]
    del data["orig"]
    del data["time"]
    list_jsons.append(data)
df: pd.DataFrame = pd.json_normalize(list_jsons).set_index("class_name").rename(columns={"outcome_distribution.1": "num_regular", "outcome_distribution.0": "num_deviant"})
df = df.rename(index=map_ds_names)


def replace_terms(col: str):
    col = col.replace("Num ", "\#").replace("Seq ", "").replace("Events", "Ev.").replace("Features", "Attr")
    col = col.replace("Ratio", "\%").replace("Distinct", "Unique")
    return col


def transform_cols(col: List[str]):
    return replace_terms(" ".join([string.title() for string in col.split("_")]))


df.columns = df.columns.map(transform_cols)
df.index.names = ["Dataset"]
df
# %%
# plt.subplots()
df_length_dist = pd.DataFrame(all_length_distributions).fillna(0)
df_tmp = pd.melt(df_length_dist.reset_index(), id_vars="index", value_vars=all_length_distributions.keys())
# sns.catplot(data=df_tmp.sort_values("index"), x="index", y="value", kind="bar", col_wrap=3, col="variable")
# %%
caption = "All datasets used within the evaluation. Dice4EL is used for the qualitative evaluation and the remaining are used for quantitative evaluation purposes."
df_styled = df.style.format(
    # escape='latex',
    # precision=0,
    na_rep='',
    thousands=" ",
)

df_latex = df_styled.to_latex(
    multicol_align='l',
    # column_format='l',
    # caption=caption,
    # label=f"tbl:dataset-stats",
    hrules=True,
)
# .replace("tabular", "tabularx")

# df_latex = df_latex.replace("\\begin{table}", "\\begin{adjustbox}{center}")
# df_latex = df_latex.replace("\\end{table}", "\\end{adjustbox}")

print(df_latex)
display(df_styled)
save_table(df_latex, "dataset_stats")
# %%
