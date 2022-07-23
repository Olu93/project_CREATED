# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
from jupyter_constants import C_VIABILITY_COMPONENT, COLS_VIAB_COMPONENTS,map_parts_overall, map_mrates, map_parts, map_operators, map_operator_short2long, map_viability_specifics, map_erate, save_figure

# %%
# PATH = pathlib.Path('results/models_overall/overall_sidequest/experiment_overall_sidequest_results.csv')
PATH = pathlib.Path('results/models_overall/overall/experiment_overall_results.csv')
original_df = pd.read_csv(PATH)
original_df.head()
# %%
plt.figure(figsize=(10, 10))
sns.scatterplot(data=original_df, x="likelihood", y="viability", hue="run.short_name")
# %%
data = original_df.copy()
all_type_cols = [col for col in data.columns if ("type" in col) and col != "viab.type"]
data["is_degenerate"] = data["cf"].str.split(">").apply(lambda x: np.sum([int(i) for i in x])) == 0
# data["id"] = data["model_name"] + data[all_type_cols].fillna("NA").agg('_'.join, axis=1)
# data["id"].str.extract(r"((([A-Z])+)_)+", expand=True)
# data["shortid"] = data["id"].str.replace(pat=r"([a-z])+", repl="", regex=True)
data["is_correct"] = data["result_outcome"] == data["target_outcome"]
data
# %%
top_10 = data[(data["rank"] < 11)]
top_10_means = top_10.groupby(["run.short_name", "iteration.no"]).mean()
top_10_means["is_correct"] = (top_10_means["is_correct"] == 1)
# %%
plt.figure(figsize=(10, 10))
sns.scatterplot(data=top_10_means, x="likelihood", y="viability", hue="run.short_name", style="iteration.no")
# %%
plt.figure(figsize=(10, 10))
# sns.scatterplot(data=top_10_means, x="sparcity", y="similarity", hue="run.short_name", size="viability")
sns.scatterplot(data=top_10, x="sparcity", y="similarity", hue="run.short_name", size="viability", style="is_degenerate", sizes=(1, 100), alpha=0.5)
# %%
plt.figure(figsize=(10, 10))
col_selector = (data.model_name != "SimpleGeneratorModel") & (data.model_name != "RandomGenerator") & (data.is_degenerate != True)
sns.scatterplot(data=top_10[col_selector], x="sparcity", y="similarity", hue="run.short_name", size="viability", sizes=(40, 400), alpha=0.5)

# %%
# df_tmp = top_10.copy()
# df_intermediate = df_tmp[df_tmp.columns[df_tmp.isnull().sum() == 0]]

# df_melt = df_intermediate.melt(id_vars=set(df_intermediate.columns) - set(COLS_VIAB_COMPONENTS), value_vars=COLS_VIAB_COMPONENTS, var_name=C_VIABILITY_COMPONENT, value_name="Value")
# sns.relplot(data=df_melt, x="Value", y="viability", hue=C_VIABILITY_COMPONENT)
# %%
# col_selector = (data.model_name != "SimpleGeneratorModel") & (data.model_name != "RandomGenerator") & (data.is_degenerate != True)
# df_tmp = top_10[col_selector].copy()
df_tmp = data.copy().rename(columns=map_parts_overall)
df_intermediate = df_tmp[df_tmp.columns[df_tmp.isnull().sum() == 0]]

df_melt = df_intermediate.melt(id_vars=set(df_intermediate.columns) - set(COLS_VIAB_COMPONENTS), value_vars=COLS_VIAB_COMPONENTS, var_name=C_VIABILITY_COMPONENT, value_name="Value")
sns.pairplot(data=df_intermediate, vars=COLS_VIAB_COMPONENTS+["viability"], hue="run.short_name")
# Second pairplot with different lower triangle. Maybe target outcome
save_figure("exp4_all_vs_all")
# %%
# plt.figure(figsize=(10, 10))
# all_common_cols = [col for col in data.columns if ("gen." not in col)]
# sns.heatmap(data[all_common_cols].corr())

# %%
# fig, ax = plt.subplots(figsize=(10, 10))
# sns.lineplot(data=data[all_common_cols], x="rank", y="viability", hue="model_name")
# ax.invert_xaxis()
# %%
# # EVO_PATH = pathlib.Path('results/models_specific/overall/EvoGeneratorWrapper/EGW_ES_EGW_CBI_RWS_TPC_DDM_BBR_IM.csv')
# # EVO_PATH = pathlib.Path('results/models_specific/overall/EvoGeneratorWrapper/EGW_ES_EGW_CBI_TS_TPC_DDM_BBR_IM.csv')
# # EVO_PATH = pathlib.Path('results/models_specific/overall/EvoGeneratorWrapper/EGW_ES_EGW_DDSI_RWS_TPC_DDM_BBR_IM.csv')
# EVO_PATH = pathlib.Path('results/models_specific/overall/EvoGeneratorWrapper/EGW_ES_EGW_DBI_RWS_TPC_DDM_IM.csv')
# evo_df = pd.read_csv(EVO_PATH)
# evo_df.head(10)
# # %%
# evo_df_means = evo_df.groupby(['instance.no', 'iteration.no']).mean().reset_index()
# evo_df_means

# # # %%
# plt.figure(figsize=(10, 10))
# sns.lineplot(data=evo_df_means, x="row.no", y="instance.duration_s", hue="iteration.no")
# # %%
# plt.figure(figsize=(10, 10))
# sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_sparcity", hue="instance.no")
# # %%
# plt.figure(figsize=(10, 10))
# sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_similarity", hue="instance.no")
# # %%
# plt.figure(figsize=(10, 10))
# sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_feasibility", hue="instance.no")
# # %%
# plt.figure(figsize=(10, 10))
# sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_delta", hue="instance.no")
# # %%
# plt.figure(figsize=(10, 10))
# sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_viability", hue="instance.no")
# # %%
# # sns.ecdfplot(data=evo_df_means, x="viability")
# # %%
