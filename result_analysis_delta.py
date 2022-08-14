# %%
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
from sklearn import metrics
from IPython.display import display
from scipy import stats
from scipy import spatial
from jupyter_constants import *
# %%
positive_points = np.array([
    [0.4, 0.2],
    [0.53, 0.75],
    [0.7, 0.4],
    [0.2, 0.6],
    [0.1, 0.9],
    [0.9, 0.1],
    [0.2, 0.1],
    [0.7, 0.9],
    [0.3, 0.8],
    [0.6, 0.1],
])
negative_points = np.array([
    [0.3, 0.4],
    [0.65, 0.55],
    [0.1, 0.3],
    [0.05, 0.45],
    [0.9, 0.7],
])
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot([(0, 0), (1, 1)], '-.', color="black")

# x0 = np.arange(0.0, 0.5, 0.01)
# x1 = np.arange(0.5, 1, 0.01)
# x1 = np.arange(0.5, 1, 0.01)
# y1 = x0 * 1
# y2 = x0 * 0
# y3 = x1 * 1
# y4 = np.zeros_like(x0)
# ax.fill_between(x0, y1, y2, alpha=0.5)
# ax.fill_between(x1, y3, y4, alpha=0.5)
# ax.fill_between(x0, y1, y2, alpha=0.5)
# ax.fill_between(x0, y1, y2, alpha=0.5)

ax.scatter(positive_points[:, 0], positive_points[:, 1], s=50, color='green', label='Positive Delta')
ax.scatter(negative_points[:, 0], negative_points[:, 1], s=50, color='red', label='Negative Delta')
ax.axhline(y=0.5, color='k')
ax.axvline(x=0.5, color='k')
# ax.invert_yaxis()
for i, (x, y) in enumerate(positive_points):
    ax.text(positive_points[i, 0] + 0.01, positive_points[i, 1] + 0.01, f"{positive_points[i]}".replace(" ", ", ").replace("[", "(").replace("]", ")"))
for i, (x, y) in enumerate(negative_points):
    ax.text(negative_points[i, 0] + 0.01, negative_points[i, 1] + 0.01, f"{negative_points[i]}".replace(" ", ", ").replace("[", "(").replace("]", ")"))

ax.legend()

# ax.text(0.68, 0.23, f"Positive Region", c="green", size=15)
# ax.text(0.13, 0.8 , f"Positive Region", c="green", size=15)
ax.set_xlabel("Factual Prediction Score", fontsize=12, labelpad=10)
ax.set_ylabel("Counterfactual Prediction Score", rotation=0, fontsize=12, labelpad=100, va="top")
ax.yaxis.set_label_position("right")
ax.xaxis.set_label_position("top")
fig.tight_layout()
save_figure("delta-space")
plt.show()
# %%
