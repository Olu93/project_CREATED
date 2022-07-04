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
PATH = pathlib.Path('results/models_overall/evolutionary_configs/experiment_evolutionary_configs_results.csv')
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
}

df = original_df.rename(columns=exog)
df = df.rename(columns=renaming)

# sm.GLS(original_df["viability"], original_df[exog])
# %%
formular_exog = " + ".join(exog.values())
md = smf.mixedlm(f"viability ~ {formular_exog}", df, groups="full_name")
mdf = md.fit()
mdf.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md.score(mdf.params_object))
mdf.params
# %%
formular_exog = " + ".join(exog.values())
md = smf.mixedlm(f"dllh ~ {formular_exog} ", df, groups="full_name")
mdf = md.fit(method="cg")
mdf.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md.score(mdf.params_object))
mdf.params# %%

# %%
