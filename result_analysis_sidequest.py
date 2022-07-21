# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
# %%

# %%
EVO_PATH = pathlib.Path('results/models_specific/evolutionary_sidequest/EvoGeneratorWrapper/EGW_ES_EGW_DBI_TKS_TPC_DDM_PR_IM.csv')
EVO_PATH = pathlib.Path('results/models_specific/evolutionary_sidequest/EvoGeneratorWrapper/EGW_ES_EGW_DBI_USS_TPC_DDM_PR_IM.csv')
# EVO_PATH = pathlib.Path('results/models_specific/evolutionary_sidequest/EvoGeneratorWrapper/EGW_ES_EGW_DBI_ES_TPC_DDM_PR_IM.csv')
evo_df = pd.read_csv(EVO_PATH)
evo_df.head(10)
# %%
evo_df_means = evo_df.groupby(['instance.no', 'iteration.no']).mean().reset_index()
evo_df_means


# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_sparcity", hue="instance.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_similarity", hue="instance.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_feasibility", hue="instance.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_delta", hue="instance.no")
# %%
plt.figure(figsize=(10, 10))
sns.lineplot(data=evo_df_means, x="iteration.no", y="iteration.mean_viability", hue="instance.no")
# %%
# sns.ecdfplot(data=evo_df_means, x="viability")
# %%
