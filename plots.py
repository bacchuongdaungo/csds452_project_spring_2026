import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

DIR = Path(__file__).resolve()
PROJECT_ROOT = DIR.parent
_FOREST_DIR = PROJECT_ROOT / "Forest"
_BART_DIR = PROJECT_ROOT / "BART"
_KNN_DIR = PROJECT_ROOT / "KNN"


df_Forest = pd.read_csv(_FOREST_DIR / "forest_results.csv")
df_bart = pd.read_csv(_BART_DIR / "bart_results_base.csv")
df_knn = pd.read_csv(_KNN_DIR / "results" / "original" / "k_18" / "metrics.csv")

print("Forest columns:", df_Forest.columns)
print("BART columns:", df_bart.columns)
print("KNN columns:", df_knn.columns)

def clean_df(df):
    df = df[~df["replica"].isin(["MEAN", "STD", "MEDIAN"])].copy()
    df["replica_id"] = df["replica"].str.extract(r'(\d+)').astype(int)
    return df.sort_values("replica_id")
def clean_knn(df):
    df = df.copy()

    # Create replica_id (since none exists)
    df["replica_id"] = range(1, len(df) + 1)

    # Rename columns
    df = df.rename(columns={
        "ate_abs_error": "ate_error",
        "counterfactual_rmse": "cf_rmse",
        "control_counterfactual_rmse": "control_cf_rmse",
        "treated_counterfactual_rmse": "treated_cf_rmse"
    })

    # Add missing columns (set to NaN so Plotly skips them)
    df["att_error"] = None
    df["policy_value"] = None

    return df
def plot (df_Forest, df_bart, df_knn):
    df_Forest = clean_df(df_Forest)
    df_bart = clean_df(df_bart)
    df_knn = clean_knn(df_knn)


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


    # Build interactive figure

    fig = go.Figure()

    # Add traces for BOTH datasets
    # Each metric gets two traces: mean + std
    for i, metric in enumerate(metrics):

        # Forest results
        fig.add_trace(
            go.Scatter(
                x=df_Forest["replica_id"],
                y=df_Forest[metric],
                mode="lines+markers",
                name=f"{metric} (mean)",
                visible=(i == 0)
            )
        )

        # add bart results
        fig.add_trace(
            go.Scatter(
                x=df_bart["replica_id"],
                y=df_bart[metric],
                mode="lines+markers",
                name=f"{metric} (BART)",
                line=dict(dash="dash"),
                visible=(i == 0)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_knn["replica_id"],
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
                {"title": f"{metric} Across Replicas"}
            ]
        ))

    # =========================
    # Layout
    # =========================
    fig.update_layout(
        title="Dynamic Forest Metrics Viewer",
        xaxis_title="Replica ID",
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
    fig.write_html("plots_saved/base_comparison.html")

if __name__ == "__main__":
    plotNum = 1 # is base 2 is noisy 3 is drop 4 is both
    if plotNum == 1:
        df_Forest = pd.read_csv(_FOREST_DIR / "forest_results.csv")
        df_bart = pd.read_csv(_BART_DIR / "bart_results_base.csv")
        df_knn = pd.read_csv(_KNN_DIR / "results" / "original" / "k_18" / "metrics.csv")
        plot(df_Forest, df_bart, df_knn)
    elif plotNum == 2:
        df_Forest = pd.read_csv(_FOREST_DIR / "forest_gaussianSTD.csv")
        df_bart = pd.read_csv(_BART_DIR / "bart_results_gaussian.csv")
        df_knn = pd.read_csv(_KNN_DIR / "results" / "gaussianSTD" / "k_18" / "metrics.csv")
        plot(df_Forest, df_bart, df_knn)
    elif plotNum == 3:
        df_Forest = pd.read_csv(_FOREST_DIR / "forest_drop_repeat.csv")
        df_bart = pd.read_csv(_BART_DIR / "bart_results_drop.csv")
        df_knn = pd.read_csv(_KNN_DIR / "results" / "drop" / "k_18" / "metrics.csv")
        plot(df_Forest, df_bart, df_knn)
    elif plotNum == 4:
        df_Forest = pd.read_csv(_FOREST_DIR / "forest_both.csv")
        df_bart = pd.read_csv(_BART_DIR / "bart_results_both.csv")
        df_knn = pd.read_csv(_KNN_DIR / "results" / "noisy_drop" / "k_18" / "metrics.csv")
        plot(df_Forest, df_bart, df_knn)
