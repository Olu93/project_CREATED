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

renaming = {
    "run.short_name": "short_name",
    "run.full_name": "full_name",
    "gen.mutator.p_delete":"dmrate",
    "gen.mutator.p_insert":"imrate",
    "gen.mutator.p_change":"cmrate",
    "gen.mutator.p_transp":"tmrate",
    "gen.mutator.p_none":"nmrate",
    "gen.mutator.edit_rate":"erate",
    "gen.crosser.crossover_rate":"crate",
    "gen.recombiner.recombination_rate":"rrate",
    "run.duration_sec":"duration",
}

cols_of_interest = ["dllh", "viability", "erate", "dmrate", "imrate", "cmrate", "tmrate", "nmrate"]
cols_config = ["initiator", "selector", "mutator", "crosser", "recombiner"]

df = original_df.copy()
df = df.rename(columns=exog)
df = df.rename(columns=renaming)

# sm.GLS(original_df["viability"], original_df[exog])
# %%
formular_exog = " + ".join(exog.values())
md = smf.mixedlm(f"viability ~ erate + dmrate + imrate + cmrate + tmrate + {formular_exog}", df, groups="short_name")
mdf = md.fit()
mdf.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md.score(mdf.params_object))
mdf.params
# %%
formular_exog = " + ".join(exog.values())
md = smf.mixedlm(f"dllh ~ erate + dmrate + imrate + cmrate + tmrate + {formular_exog}", df, groups="short_name")
mdf = md.fit()
mdf.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md.score(mdf.params_object))
mdf.params
# %%
df_only_feasible = df[df['dllh'] > 0]
formular_exog = " + ".join(exog.values())
md = smf.mixedlm(f"viability ~ erate + dmrate + imrate + cmrate + tmrate + {formular_exog}", df_only_feasible, groups="short_name")
mdf = md.fit()
mdf.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md.score(mdf.params_object))
mdf.params# %%

# %%
df_only_feasible = df[df['dllh'] > 0]
formular_exog = " + ".join(exog.values())
md = smf.mixedlm(f"dllh ~ erate + dmrate + imrate + cmrate + tmrate + {formular_exog}", df_only_feasible, groups="short_name")
mdf = md.fit()
mdf.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md.score(mdf.params_object))
mdf.params# %%

# %%
df_high_feasible = df[df['dllh'] > 0.1]
df_high_feasible.groupby(['short_name']).mean()[cols_of_interest].mean()
# %%
df.describe()[cols_of_interest]
# %%
df[df['dllh'] > 0.1].describe()[cols_of_interest]
# %%
df[df['dllh'] > 0.2].describe()[cols_of_interest]
# %%
best_values = df[df.viability == df.viability.max()]
best_values
# %%
best_values[cols_config + cols_of_interest]
# %%
df["is_feasible"] = df['dllh'] > 0
most_feasible = df.groupby('short_name').sum()["is_feasible"].sort_values().tail()
most_feasible
# %%
best_configs_to_avoid_zeros = df[df["short_name"].isin(most_feasible.index)]
best_configs_to_avoid_zeros
# %%
best_configs_to_avoid_zeros.groupby(cols_config).mean()[cols_of_interest]
# %%


tmp0=df[df['dllh'] > 0.0].describe()[cols_of_interest].loc["50%"]
tmp1=df[df['dllh'] > 0.1].describe()[cols_of_interest].loc["50%"]
tmp2=df[df['dllh'] > 0.2].describe()[cols_of_interest].loc["50%"]
tmpa=best_configs_to_avoid_zeros.describe()[cols_of_interest].loc["50%"]
# %%
pd.DataFrame([tmp0, tmp1, tmp2, tmpa]).T
# %%
most_feasible
# %%
best_values