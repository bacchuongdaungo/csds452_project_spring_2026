import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import re
import numpy as np

DIR = Path(__file__).resolve()
PROJECT_ROOT = DIR.parent
_FOREST_DIR = PROJECT_ROOT / "Forest"
_BART_DIR = PROJECT_ROOT / "BART"
_KNN_DIR = PROJECT_ROOT / "KNN"


df_Forest = pd.read_csv(_FOREST_DIR / "forest_results.csv")
df_bart = pd.read_csv(_BART_DIR / "bart_results_base.csv")
df_knn = pd.read_csv(_KNN_DIR / "knn_results_base.csv")

def clean_df(df):
    df = df[~df["replica"].isin(["MEAN", "STD", "MEDIAN"])].copy()
    df["replica_id"] = df["replica"].str.extract(r'(\d+)').astype(int)
    return df.sort_values("replica_id")
def clean_knn_df(df, key):
    df = df.copy()

    # normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # KNN has no replica → create a stable index
    match key:
        case "original":
            df["replica_id"] = range(1, len(df) + 1)
        case "gaussianSTD":
            df["replica"] = [0.2, 0.4, 0.6, 0.8, 1.0]
        case "drop":
            df["replica"] = [3,6,9,12,15]

    # Map KNN-specific metrics into your standard schema
    # rename_map = {
    #     "counterfactual_rmse": "cf_rmse",
    #     "control_counterfactual_rmse": "control_cf_rmse",
    #     "treated_counterfactual_rmse": "treated_cf_rmse",
    #     "ate_abs_error": "ate_error"
    # }

    # df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure missing expected columns exist (prevents KeyErrors in plotting)
    required_cols = [
        "pehe",
        "ate_error",
        "att_error",
        "policy_value",
        "cf_rmse",
        "control_cf_rmse",
        "treated_cf_rmse"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA  # safe placeholder for Plotly

    return df
def clean_noise_df(df):
    df = df.copy()

    if "replica" in df.columns:
        df = df[~df["replica"].isin(["MEAN", "STD", "MEDIAN"])]

    # extract noise level
    df["noise_std"] = (
        df["replica"]
        .str.replace(".csv", "", regex=False)
        .str.split("std_")
        .str[-1]
        .astype(float)
)

    return df.sort_values("noise_std")

def extract_drop(x):
    return float(re.search(r"drop_(\d+)", x).group(1))

def clean_drop_df(df):
    df = df.copy()

    df.columns = df.columns.str.strip().str.lower()

    # remove summary rows
    df = df[~df["replica"].isin(["MEAN", "STD", "MEDIAN"])]

    # extract drop level
    df["drop_level"] = df["replica"].apply(extract_drop)

    return df.sort_values("drop_level")

def plot (df_Forest, df_bart, df_knn, plotNum):

    match plotNum:
        case 1:
            title_suffix = "Base"
        case 2:
            title_suffix = "Gaussian"
        case 3:
            title_suffix = "Drop"
        case 4:
            title_suffix = "Both"
    df_Forest.to_latex(PROJECT_ROOT / "Latex_Tables" / f"forest_results_{title_suffix}.tex", index=False)
    df_bart.to_latex(PROJECT_ROOT / "Latex_Tables" / f"bart_results_{title_suffix}.tex", index=False)
    df_knn.to_latex(PROJECT_ROOT / "Latex_Tables" / f"knn_results_{title_suffix}.tex", index=False)
    match plotNum:
        case 1:
            df_Forest = clean_df(df_Forest)
            df_bart = clean_df(df_bart)
            df_knn = clean_knn_df(df_knn, "original")
            graph("replica_id", df_Forest, df_bart, df_knn)
        case 2:
            df_Forest = clean_noise_df(df_Forest)
            df_bart = clean_noise_df(df_bart)
            df_knn = clean_noise_df(df_knn)
            graph("noise_std", df_Forest, df_bart, df_knn)
        case 3:
            df_Forest = clean_drop_df(df_Forest)
            df_bart = clean_drop_df(df_bart)
            df_knn = clean_knn_df(df_knn, "drop")
            graph("drop_level", df_Forest, df_bart, df_knn)
        case 4:
            df_Forest = clean_drop_df(df_Forest)
            df_bart = clean_drop_df(df_bart)
            df_knn = clean_knn_df(df_knn, "drop")
            graph("drop_level", df_Forest, df_bart, df_knn)

        

def graph(xaxis, df_Forest, df_bart, df_knn):
        # Metrics to visualize
    metrics = [
        "pehe",
        "ate_error",
        "att_error",
        "policy_value",
        "cf_rmse",
        "control_cf_rmse",
        "treated_cf_rmse"
    ]
    fig = go.Figure()

    # Add traces for BOTH datasets
    # Each metric gets two traces: mean + std
    for i, metric in enumerate(metrics):

        # Forest results
        fig.add_trace(
            go.Scatter(
                x=df_Forest[xaxis],
                y=df_Forest[metric],
                mode="lines+markers",
                name=f"{metric} (mean)",
                visible=(i == 0)
            )
        )

        # add bart results
        fig.add_trace(
            go.Scatter(
                x=df_bart[xaxis],
                y=df_bart[metric],
                mode="lines+markers",
                name=f"{metric} (BART)",
                line=dict(dash="dash"),
                visible=(i == 0)
            )
        )

        # add knn results
        fig.add_trace(
            go.Scatter(
                x=df_knn["replica"],
                y=df_knn[metric],
                mode="lines+markers",
                name=f"{metric} (KNN)",
                line=dict(dash="dot"),
                visible=(i == 0)
            )
        )

    # Dropdown logic

    buttons = []

    traces_per_metric = 3
    total_traces = len(metrics) * traces_per_metric

    for i, metric in enumerate(metrics):
        visibility = [False] * total_traces

        base = i * traces_per_metric
        visibility[base] = True       # Forest
        visibility[base + 1] = True   # BART
        visibility[base + 2] = True   # KNN

        buttons.append(dict(
            label=metric,
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"{metric} Across Replicas"},
            ]
        ))

    # =========================
    # Layout
    # =========================
    fig.update_layout(
        title=f"Dynamic Forest Metrics Viewer - {xaxis}",
        xaxis_title=xaxis.capitalize().split("_")[0] +" " + xaxis.capitalize().split("_")[1] if "_" in xaxis else xaxis.capitalize(),
        yaxis_title="Metric Value",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True
            )
        ]
    )

    fig.show()


if __name__ == "__main__":
    plotNum = 2 # is base 2 is noisy 3 is drop 4 is both
    if plotNum == 1:
        df_Forest = pd.read_csv(_FOREST_DIR / "forest_results.csv")
        df_bart = pd.read_csv(_BART_DIR / "bart_results_base.csv")
        df_knn = pd.read_csv(_KNN_DIR / "knn_results_base.csv")
        plot(df_Forest, df_bart, df_knn, plotNum)
    elif plotNum == 2:
        df_Forest = pd.read_csv(_FOREST_DIR / "forest_gaussianSTD.csv")
        df_bart = pd.read_csv(_BART_DIR / "bart_results_gaussian.csv")
        df_knn = pd.read_csv(_KNN_DIR / "knn_results_gaussian.csv")
        plot(df_Forest, df_bart, df_knn, plotNum)
    elif plotNum == 3:
        df_Forest = pd.read_csv(_FOREST_DIR / "forest_drop_repeat.csv")
        df_bart = pd.read_csv(_BART_DIR / "bart_results_drop.csv")
        df_knn = pd.read_csv(_KNN_DIR / "knn_results_drop.csv")
        plot(df_Forest, df_bart, df_knn, plotNum)
    elif plotNum == 4:
        df_Forest = pd.read_csv(_FOREST_DIR / "forest_both.csv")
        df_bart = pd.read_csv(_BART_DIR / "bart_results_both.csv")
        df_knn = pd.read_csv(_KNN_DIR / "knn_results_both.csv")
        plot(df_Forest, df_bart, df_knn, plotNum)
