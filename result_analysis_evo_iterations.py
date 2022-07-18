# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
# https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/anova/how-to/mixed-effects-model/interpret-the-results/key-results/
# %%
PATH = pathlib.Path('results/models_overall/evolutionary_iterations/experiment_evolutionary_iterations_results.csv')
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

renaming = {
    "run.short_name": "short_name",
    "run.full_name": "full_name",
    "gen.max_iter":"num_iter",
    "iteration.no":"instance",
    "row.no":"cf_id",
}

cols_operators = list(exog.values())

df = original_df.rename(columns=exog)
df = df.rename(columns=renaming)
df["group"] = df["full_name"] + "_" + df["num_iter"].map(str) + "_" + df["instance"].map(str)
df["cfg_set"] = df[cols_operators].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
# sm.GLS(original_df["viability"], original_df[exog])
df
# %%
formular_exog = " + ".join(exog.values())
md = smf.mixedlm(f"viability ~ C(num_iter)", df, groups="group")
mdf = md.fit()
mdf.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md.score(mdf.params_object))
mdf.params
# %%
formular_exog = " + ".join(exog.values())
md = smf.mixedlm(f"feasibility ~ C(num_iter) ", df, groups="group")
mdf = md.fit(method="cg")
mdf.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md.score(mdf.params_object))
mdf.params# %%

# %%
fig, ax = plt.subplots(1,1, figsize=(15,15))
sns.lineplot(data=df, x="num_iter", y="viability", hue="cfg_set", ax=ax)
# %%
