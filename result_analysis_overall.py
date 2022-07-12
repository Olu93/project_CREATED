# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
# %%
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
plt.figure(figsize=(10, 10))
all_common_cols = [col for col in data.columns if ("gen." not in col)]
sns.heatmap(data[all_common_cols].corr())
# %%
# EVO_PATH = pathlib.Path('results/models_specific/overall/EvoGeneratorWrapper/EGW_ES_EGW_CBI_RWS_TPC_DDM_BBR_IM.csv')
# EVO_PATH = pathlib.Path('results/models_specific/overall/EvoGeneratorWrapper/EGW_ES_EGW_CBI_TS_TPC_DDM_BBR_IM.csv')
EVO_PATH = pathlib.Path('results/models_specific/overall/EvoGeneratorWrapper/EGW_ES_EGW_DDSI_RWS_TPC_DDM_BBR_IM.csv')
# EVO_PATH = pathlib.Path('results/models_specific/overall/EvoGeneratorWrapper/EGW_ES_EGW_DDSI_TS_TPC_DDM_BBR_IM.csv')
evo_df = pd.read_csv(EVO_PATH)
evo_df.head(10)
# %%
evo_df_means = evo_df.groupby(['iteration.no', 'row.no']).mean().reset_index()
evo_df_means

# # %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="row.no", y="iteration.duration_s", hue="iteration.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="row.no", y="iteration.mean_sparcity", hue="iteration.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="row.no", y="iteration.mean_similarity", hue="iteration.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="row.no", y="iteration.mean_feasibility", hue="iteration.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="row.no", y="iteration.mean_delta", hue="iteration.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="row.no", y="iteration.mean_viability", hue="iteration.no")
# %%
sns.ecdfplot(data=evo_df_means, x="viability")
# %%
