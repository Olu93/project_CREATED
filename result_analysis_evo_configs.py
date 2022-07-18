# %%
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
PATH = pathlib.Path('results/models_overall/evolutionary_configs/experiment_evolutionary_configs_results.csv')
original_df = pd.read_csv(PATH)
original_df.head()
# %%
full_name_str = "Configuration Set"

exog = {
    "gen.initiator.type": "initiator",
    "gen.selector.type": "selector",
    "gen.mutator.type": "mutator",
    "gen.recombiner.type": "recombiner",
    "gen.crosser.type": "crosser",
}

renaming = {
    "run.short_name": "short_name",
    "run.full_name": full_name_str,
}

df = original_df.rename(columns=exog)
df = df.rename(columns=renaming)

# sm.GLS(original_df["viability"], original_df[exog])
# %%
formular_exog = " + ".join(exog.values())
md1 = smf.mixedlm(f"viability ~ {formular_exog}", df, groups=full_name_str)
mdf1 = md1.fit()
mdf1.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md1.score(mdf1.params_object))
mdf1.params
# %%
formular_exog = " + ".join(exog.values())
md2 = smf.mixedlm(f"feasibility ~ {formular_exog} ", df, groups=full_name_str)
mdf2 = md2.fit(method="cg")
mdf2.summary()
# %%
# https://github.com/statsmodels/statsmodels/issues/6157
print(md2.score(mdf2.params_object))
mdf2.params  # %%

# %%
base1, specific1 = mdf1.summary().tables
base2, specific2 = mdf2.summary().tables
# %%
specific1_mod = specific1.iloc[:, :-2].drop('z', axis=1).rename(columns={'P>|z|': 'p-value'})
specific2_mod = specific2.iloc[:, :-2].drop('z', axis=1).rename(columns={'P>|z|': 'p-value'})
fmt = {
   ("Numeric", "Integers"): '\${}',
   ("Numeric", "Floats"): '{:.6f}',
   ("Non-Numeric", "Strings"): str.upper
}

# %%
caption = StringIO()
caption.write("Table shows the result of the linear mixed model.")
caption.write(" ")
caption.write("It uses viability as dependent variable and evolutionary operators as independent categorical variable.")
caption.write(" ")
caption.write("The model is adjusted for general differences in combinations.")
caption.write(" ")
caption.write("The coefficient explains the effect of the model, while p-value explains the statistical significance.")
caption.write(" ")
caption.write("Only significant values are considerable effects.")

result_table_latex = specific1_mod.style.format(fmt).to_latex(
    caption=caption.getvalue(),
    label="tbl:configs_viability",
)

print(result_table_latex)

# %%
caption = StringIO()
caption.write("Table shows the result of the linear mixed model.")
caption.write(" ")
caption.write("It uses feasibility as dependent variable and evolutionary operators as independent categorical variable.")
caption.write(" ")
caption.write("The model is adjusted for general differences in combinations.")
caption.write(" ")
caption.write("The coefficient explains the effect of the model, while the p-value explains the statistical significance.")
caption.write(" ")
caption.write("Only significant values are considerable effects.")

result_table_latex = specific2_mod.style.format(escape='latex').to_latex(
    caption=caption.getvalue(),
    label="tbl:configs_feasibility",
)

print(result_table_latex)

# %%
# plt.figure(figsize=(10, 10))
# sns.lineplot(data=df.groupby('short_name'), x="rank", y="avg_viability", hue="short_name",)

# %%
df["Name"] = df[full_name_str].str.replace('EvoGeneratorWrapper', '')
df["Name"] = df["Name"].str.replace('EvolutionaryStrategy', '')
df["Name"] = df["Name"].str.replace('_ImprovementMeasure', '')
df["Name"] = df["Name"].str.replace('___', '')
df["Name"] = df["Name"].str.replace('_', '-')


# %%
fig, axes = plt.subplots(2, 3, figsize=(15,10), sharey=True)
faxes = axes.flatten()
# df_no_fi = df[df['initiator'] != 'FactualInitiator']
df_no_fi = df
y_of_interest = "feasibility"
for col, ax in zip(['initiator', 'selector', 'crosser', 'mutator', 'recombiner'], faxes): 
    df_agg = df_no_fi.groupby([col, "rank"]).mean().reset_index()#.replace()
    df_agg = df_agg.rename(columns={col: col.upper()})
    sns.lineplot(data=df_agg, x="rank", y=y_of_interest, hue=col.upper(), ax=ax)
    ax.invert_xaxis()
    ax.set_xlabel("Rank of Counterfactual")


# %%
fig, axes = plt.subplots(2, 3, figsize=(15,10), sharey=True)
faxes = axes.flatten()
# df_no_fi = df[df['initiator'] != 'FactualInitiator']
df_no_fi = df
y_of_interest = "viability"
x_of_interest = "rank"
for col, ax in zip(['initiator', 'selector', 'crosser', 'mutator', 'recombiner'], faxes): 
    df_agg = df_no_fi.groupby([col, x_of_interest]).mean().reset_index()#.replace()
    df_agg = df_agg.rename(columns={col: col.upper()})
    sns.lineplot(data=df_agg, x=x_of_interest, y=y_of_interest, hue=col.upper(), ax=ax)
    ax.invert_xaxis()
    ax.set_xlabel(f"{x_of_interest.title()} of Counterfactual")

# ax = faxes[-1]
# sns.lineplot(data=df_no_fi, x="rank", y=y_of_interest, ax=ax)
# sns.lineplot(data=df[df['initiator'] != 'FactualInitiator'], x="rank", y=y_of_interest, ax=ax)

# 