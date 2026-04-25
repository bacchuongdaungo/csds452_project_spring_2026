"""
plot_noise.py
=============
Visualizes mean metric score vs noise level across KNN, Forest, and BART.
X axis: noise level (std or columns dropped)
Y axis: mean metric score across replicas
Dropdown: select noise type (gaussian, drop, both)

Set METRIC below to change which metric is plotted.
"""

import re
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
METRIC = "pehe"

DIR          = Path(__file__).resolve().parent
_FOREST_DIR  = DIR / "Forest"
_BART_DIR    = DIR / "BART"
_KNN_DIR     = DIR / "KNN"


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_gaussian():
    forest = pd.read_csv(_FOREST_DIR / "forest_gaussianSTD.csv")
    bart   = pd.read_csv(_BART_DIR   / "bart_results_gaussian.csv")
    knn    = pd.read_csv(_KNN_DIR    / "knn_results_gaussian.csv")
    return forest, bart, knn

def load_drop():
    forest = pd.read_csv(_FOREST_DIR / "forest_drop_repeat.csv")
    bart   = pd.read_csv(_BART_DIR   / "bart_results_drop.csv")
    knn    = pd.read_csv(_KNN_DIR    / "knn_results_drop.csv")
    return forest, bart, knn

def load_both():
    forest = pd.read_csv(_FOREST_DIR / "forest_both.csv")
    bart   = pd.read_csv(_BART_DIR   / "bart_results_both.csv")
    knn    = pd.read_csv(_KNN_DIR    / "knn_results_both.csv")
    return forest, bart, knn


# ── Cleaners ──────────────────────────────────────────────────────────────────

def mean_gaussian(df, metric):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # keep only valid rows
    df = df[df["replica"].str.contains(r"std_\d+(?:\.\d+)?", regex=True, na=False)]

    df["noise_std"] = (
        df["replica"]
        .str.extract(r"std_(\d+(?:\.\d+)?)")[0]
        .astype(float)
    )

    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return (
        df.groupby("noise_std")[metric]
        .mean()
        .reset_index()
        .rename(columns={"noise_std": "x", metric: "y"})
    )


def mean_drop(df, metric):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # keep only valid rows
    df = df[df["replica"].str.contains(r"drop_\d+", regex=True, na=False)]

    df["drop_level"] = (
        df["replica"]
        .str.extract(r"drop_(\d+)")[0]
        .astype(float)
    )

    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return (
        df.groupby("drop_level")[metric]
        .mean()
        .reset_index()
        .rename(columns={"drop_level": "x", metric: "y"})
    )


def clean_knn(df, mode, metric):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        "counterfactual_rmse":         "cf_rmse",
        "control_counterfactual_rmse": "control_cf_rmse",
        "treated_counterfactual_rmse": "treated_cf_rmse",
        "ate_abs_error":               "ate_error",
    })

    if metric not in df.columns:
        df[metric] = pd.NA

    if mode == "gaussian":
        df["x"] = [0.2, 0.4, 0.6, 0.8, 1.0]
    elif mode in ("drop", "both"):
        df["x"] = [3, 6, 9, 12, 15]

    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return df[["x", metric]].rename(columns={metric: "y"})


# ── Build figure ──────────────────────────────────────────────────────────────

def build_figure(metric):
    fig = go.Figure()

    conditions = [
        ("gaussian", "Gaussian Noise (std)", load_gaussian),
        ("drop",     "Feature Drop (# cols)", load_drop),
        ("both",     "Both (# cols dropped)", load_both),
    ]

    traces_per_condition = 3
    total_traces = len(conditions) * traces_per_condition

    for cond_idx, (mode, xlabel, loader) in enumerate(conditions):
        forest_df, bart_df, knn_df = loader()

        if mode == "gaussian":
            f_data = mean_gaussian(forest_df, metric)
            b_data = mean_gaussian(bart_df,   metric)
        else:
            f_data = mean_drop(forest_df, metric)
            b_data = mean_drop(bart_df,   metric)

        k_data = clean_knn(knn_df, mode, metric)

        visible = (cond_idx == 0)

        fig.add_trace(go.Scatter(
            x=f_data["x"], y=f_data["y"],
            mode="lines+markers",
            name="Causal Forest",
            line=dict(color="steelblue"),
            visible=visible,
        ))
        fig.add_trace(go.Scatter(
            x=b_data["x"], y=b_data["y"],
            mode="lines+markers",
            name="BART",
            line=dict(color="darkorange", dash="dash"),
            visible=visible,
        ))
        fig.add_trace(go.Scatter(
            x=k_data["x"], y=k_data["y"],
            mode="lines+markers",
            name="KNN",
            line=dict(color="seagreen", dash="dot"),
            visible=visible,
        ))

    buttons = []
    for cond_idx, (mode, xlabel, _) in enumerate(conditions):
        vis = [False] * total_traces
        base = cond_idx * traces_per_condition
        vis[base] = vis[base + 1] = vis[base + 2] = True

        buttons.append(dict(
            label=xlabel,
            method="update",
            args=[
                {"visible": vis},
                {"xaxis": {"title": xlabel}},
            ],
        ))

    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} vs Noise Level (mean across replicas)",
        xaxis_title=conditions[0][1],
        yaxis_title=metric.replace("_", " ").title(),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top",
        )],
        legend=dict(x=1.0, xanchor="right"),
    )

    return fig


if __name__ == "__main__":
    fig = build_figure(METRIC)
    fig.show()
