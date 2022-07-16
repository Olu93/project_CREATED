# %%
from re import S
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from io import StringIO
# https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/anova/how-to/mixed-effects-model/interpret-the-results/key-results/
# %%
PATH = pathlib.Path('results/models_overall/evolutionary_params/experiment_evolutionary_params_results.csv')
original_df = pd.read_csv(PATH)
original_df.head()
# %%


exog = {
    "gen.initiator.type": "initiator",
    "gen.selector.type": "selector",
    "gen.mutator.type": "mutator",
    "gen.recombiner.type": "recombiner",
    "gen.crosser.type": "crosser",
}

edit_types = {
    "gen.mutator.p_delete": "delete",
    "gen.mutator.p_insert": "insert",
    "gen.mutator.p_change": "change",
    "gen.mutator.p_transp": "transp",
    "gen.mutator.p_none": "nochng",
}

edit_rate = {
        "gen.mutator.edit_rate": "editrate",
}

full_name_str = "Hyperparameter Set"
renaming = {
    "run.short_name": "short_name",
    "run.full_name": full_name_str,
    "gen.crosser.crossover_rate": "crate",
    "gen.recombiner.recombination_rate": "rrate",
    "run.duration_sec": "duration",
}

cols_dependent = ["feasibility", "viability"]
cols_editrate = list(edit_rate.values())
cols_edittypes = list(edit_types.values())
cols_config = list(exog.values())

df = original_df.copy()
df = df.rename(columns=exog).rename(columns=edit_types).rename(columns=edit_rate).rename(columns=renaming)

# sm.GLS(original_df["viability"], original_df[exog])
# %%
formular_exog = " + ".join(exog.values())
md1 = smf.mixedlm(f"viability ~ editrate + editrate:delete + editrate:insert + editrate:change + editrate:transp", df, groups=full_name_str)
mdf1 = md1.fit()
mdf1.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md1.score(mdf1.params_object))
mdf1.params
# %%
formular_exog = " + ".join(exog.values())
md2 = smf.mixedlm(f"feasibility ~ editrate + editrate:delete + editrate:insert + editrate:change + editrate:transp", df, groups=full_name_str)
mdf2 = md2.fit()
mdf2.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md2.score(mdf2.params_object))
mdf2.params

# %%
base1, specific1 = mdf1.summary().tables
base2, specific2 = mdf2.summary().tables
# %%
specific1_mod = specific1.iloc[:, :-2].drop('z', axis=1).rename(columns={'P>|z|': 'p-value'})
specific2_mod = specific2.iloc[:, :-2].drop('z', axis=1).rename(columns={'P>|z|': 'p-value'})
fmt = {("Numeric", "Integers"): '\${}', ("Numeric", "Floats"): '{:.6f}', ("Non-Numeric", "Strings"): str.upper}
# %%
caption = StringIO()
caption.write("Table shows the result of the linear mixed model.")
caption.write(" ")
caption.write("It uses viability as dependent variable and the hyperparameters as independent numerical variables.")
caption.write(" ")
caption.write("The model is adjusted for general differences in indivdual hyperparameter settings.")

result_table_latex = specific1_mod.style.format(fmt).to_latex(
    caption=caption.getvalue(),
    label="tbl:params_viability",
)

print(result_table_latex)

# %%
caption = StringIO()
caption.write("Table shows the result of the linear mixed model.")
caption.write(" ")
caption.write("It uses feasibility as dependent variable and the hyperparameters as independent numerical variables.")
caption.write(" ")
caption.write("The model is adjusted for general differences in indivdual hyperparameter settings.")

result_table_latex = specific2_mod.style.format(fmt).to_latex(
    caption=caption.getvalue(),
    label="tbl:params_feasibility",
)

print(result_table_latex)

# %%
fig, ax = plt.subplots((4), figsize=(10, 15))
df_avg_over_editrate = df.groupby(cols_edittypes + ["initiator"]).mean().reset_index()
sns.lineplot(data=df_avg_over_editrate, x='delete', y='viability', ax=ax[0], hue="initiator")
sns.lineplot(data=df_avg_over_editrate, x='insert', y='viability', ax=ax[1], hue="initiator")
sns.lineplot(data=df_avg_over_editrate, x='change', y='viability', ax=ax[2], hue="initiator")
sns.lineplot(data=df_avg_over_editrate, x='transp', y='viability', ax=ax[3], hue="initiator")
# ax.set_xlabel('All hyperparams except editrate')
plt.show()
# %%
fig, ax = plt.subplots(figsize=(10, 15))
df_avg_over_selectors = df.groupby(cols_editrate+["initiator"]).mean().reset_index()
sns.lineplot(data=df_avg_over_selectors, x=cols_editrate[0], y='viability', ax=ax, hue="initiator")

# ax.set_xlabel('All hyperparams except editrate')
plt.show()
# %%

# sm = pd.melt(df, id_vars=[
#     "feasibility",
#     "viability",
#     "rank",
# ], value_vars=["delete", "insert", "change", "transp", "nochng"])
# sm


# # %%
# df_high_feasible = df[df['feasibility'] > 0.1]
# df_high_feasible.groupby(['short_name']).mean()[cols_of_interest].mean()
# # %%
# df.describe()[cols_of_interest]
# # %%
# df[df['feasibility'] > 0.1].describe()[cols_of_interest]
# # %%
# df[df['feasibility'] > 0.2].describe()[cols_of_interest]
# # %%
# best_values = df[df.viability == df.viability.max()]
# best_values
# # %%
# best_values[cols_config + cols_of_interest]
# # %%
# df["is_feasible"] = df['feasibility'] > 0
# most_feasible = df.groupby('short_name').sum()["is_feasible"].sort_values().tail()
# most_feasible
# # %%
# best_configs_to_avoid_zeros = df[df["short_name"].isin(most_feasible.index)]
# best_configs_to_avoid_zeros
# # %%
# best_configs_to_avoid_zeros.groupby(cols_config).mean()[cols_of_interest]
# # %%

# tmp0 = df[df['feasibility'] > 0.0].describe()[cols_of_interest].loc["50%"]
# tmp1 = df[df['feasibility'] > 0.1].describe()[cols_of_interest].loc["50%"]
# tmp2 = df[df['feasibility'] > 0.2].describe()[cols_of_interest].loc["50%"]
# tmpa = best_configs_to_avoid_zeros.describe()[cols_of_interest].loc["50%"]
# # %%
# pd.DataFrame([tmp0, tmp1, tmp2, tmpa]).T
# # %%
# most_feasible
# # %%
# best_values
# # %%
# rename_vals = {
#     "viability": "viability",
#     "feasibility": "feasibility",
#     "delte": "delete-rate",
#     "imrate": "insert-rate",
#     "cmrate": "change-rate",
#     "tmrate": "transp-rate",
#     "nmrate": "nochng-rate",
#     "erate": "edit-rate",
# }
# corr_matrix = df[cols_of_interest].rename(columns=rename_vals)[rename_vals.values()].corr()
# plt.figure(figsize=(10, 7))

# sns.heatmap(corr_matrix, annot=True, fmt=".3f")
# plt.savefig('latex/thesis_phase_2/figures/results/params_heatmap.png')
# plt.show()
# # %%
# # df_agg = df.groupby("rank").mean()
# sns.lineplot(df, x='rank', y="viability", hue="")
# # %%
# # df.groupby("rank")
# melted = pd.melt(
#     df,
#     value_name= "rate",
#     value_vars=["delete", "insert", "change", "transp"],
#     id_vars=["viability", "rank", "iteration.no", "feasibility"],
# )
# melted = melted.groupby(["variable","iteration.no", "rank"]).mean().groupby(["iteration.no", "rank"]).mean()
# melted
# # sns.lineplot(melted, x = 'value',  hue="variable")
# # melted
# # %%
# sns.lineplot(data=melted, x='rank', y="feasibility", hue="variable")

# # %%

# %%
