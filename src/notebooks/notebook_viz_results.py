# %%
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from helper.constants import NUMBER_OF_INSTANCES, SEQUENCE_LENGTH
from IPython.display import display

# %%
METRIC = "damerau_levenshtein"


# %%
def extract_plot_data(METRIC, results):
    t_res = results.groupby("input_x_seq_len").agg(["mean", "std", "median"])[METRIC]
    t_res = t_res.fillna(0.001)
    t_res["min"] = t_res["mean"] - (2 * t_res["std"])
    t_res["max"] = t_res["mean"] + (2 * t_res["std"])
    return t_res


def plot_curve(ax, res, title, color=None):
    curve = ax.plot(res.index, res["mean"], label=title, color=color)
    if color is not None:
        ax.fill_between(res.index, res["min"], res["max"], color=color, alpha=0.1)
    return ax


# %%
full_lstm_one_way = pd.read_csv('results/result_full_lstm_model_one_way.csv').set_index("trace")
simple_lstm_one_way = pd.read_csv('results/result_simple_lstm_model_one_way.csv').set_index("trace")
transformer_one_way = pd.read_csv('results/result_transformer_model_one_way.csv').set_index("trace")
transformer_two_way = pd.read_csv('results/result_transformer_model_two_way.csv').set_index("trace")

data = {
    "Full Vector LSTM (One Way)": extract_plot_data(METRIC, simple_lstm_one_way),
    "Simple LSTM (One Way)": extract_plot_data(METRIC, full_lstm_one_way),
    "Transformer (One Way)": extract_plot_data(METRIC, transformer_one_way),
    "Transformer (Two Way)": extract_plot_data(METRIC, transformer_two_way)
}


def plot_all(data):
    fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(15, 7))
    faxes = axes.flatten()
    for idx, (ax, (name, df)) in enumerate(zip(faxes, data.items())):
        cl = list(colors.TABLEAU_COLORS)[idx]
        plot_curve(ax, df, title=name, color=cl)
        ax.legend()
        ax.set_title(f'Mean {METRIC} over {SEQUENCE_LENGTH}')
        ax.set_ylabel(METRIC)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel(SEQUENCE_LENGTH)
    fig.tight_layout()
    plt.show()
    
def plot_all_in_one(data):
    fig, ax = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(15, 7))
    for name, df in data.items():
        plot_curve(ax, df, title=name)
        ax.legend()
        ax.set_title(f'Mean {METRIC} over {SEQUENCE_LENGTH}')
        ax.set_ylabel(METRIC)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel(SEQUENCE_LENGTH)
    fig.tight_layout()
    plt.show()

plot_all(data)
plot_all_in_one(data)

# %%