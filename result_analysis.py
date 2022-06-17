# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
# %%
PATH = pathlib.Path('results/models_overall/cf_generation_results_test.csv')
original_df = pd.read_csv(PATH)
original_df.head()
# %%
plt.figure(figsize=(10, 10))
sns.scatterplot(data=original_df, x="likelihood", y="viability", hue="wrapper")
# %%
data = original_df.copy()
all_type_cols = [col for col in data.columns if ("type" in col) and col != "viab.type"]
data["id"] = data["model_name"] + data[all_type_cols].fillna("NA").agg('_'.join, axis=1)
# data["id"].str.extract(r"((([A-Z])+)_)+", expand=True)
data["shortid"] = data["id"].str.replace(pat=r"([a-z])+", repl="", regex=True)
data["is_correct"] = data["outcome"] == data["target_outcome"]
data
# %%
top_10 = data[(data["rank"]<11)]
top_10_means = top_10.groupby(["shortid", "iteration"]).mean()
top_10_means["is_correct"] = (top_10_means["is_correct"] == 1)
# %%
plt.figure(figsize=(10, 10))
sns.scatterplot(data=top_10_means, x="likelihood", y="viability", hue="shortid", style="iteration")

# %%
plt.figure(figsize=(10, 10))
all_common_cols = [col for col in data.columns if ("gen." not in col)]
sns.heatmap(data[all_common_cols].corr())
# %%
