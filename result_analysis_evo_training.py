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
PATH = pathlib.Path('results/models_specific/specific_experiment_results.csv')
original_df = pd.read_csv(PATH)
original_df.head()
# %%
exog = {
    "iteration.initiator": "initiator",
    "iteration.selector": "selector",
    "iteration.mutator": "mutator",
    "iteration.recombiner": "recombiner",
    "iteration.crosser": "crosser",
}

renaming = {
    "row.avg_viability": "mean_viability",
    "row.no": "cycle",
    "row.avg_zeros": "mean_num_zeros",
    "row.mutsum.DELETE": "mutsum_DELETE",
    "row.mutsum.INSERT": "mutsum_INSERT",
    "row.mutsum.CHANGE": "mutsum_CHANGE",
    "row.mutsum.TRANSP": "mutsum_TRANSP",
    "row.mutsum.NONE": "mutsum_NONE",
}

df = original_df
df = df.rename(columns=exog)
df = df.rename(columns=renaming)
df["full_name"] = df[exog.values()].agg('-'.join, axis=1)

# sm.GLS(original_df["viability"], original_df[exog])
# %%
formular_exog = " + ".join(exog.values())
md = smf.mixedlm(f"mean_viability ~ {formular_exog}", df, groups="filename")
mdf = md.fit()
mdf.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md.score(mdf.params_object))
mdf.params
# %%
formular_exog = " + ".join(exog.values())
# https://github.com/statsmodels/statsmodels/blob/35b803767bd7803ca5f9fc35d4546aa8cb7be844/statsmodels/regression/tests/test_lme.py#L316
vcf = {"cycle": "1 + C(cycle)"}
md = smf.mixedlm(f"mean_num_zeros ~ {formular_exog} ", df, groups="full_name")
mdf = md.fit()
mdf.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md.score(mdf.params_object))
mdf.params  # %%

# %%
