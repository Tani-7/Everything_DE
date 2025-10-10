"""
src/stats_viz.py
Produce JSON-serializable datasets for visualizations. Uses plotly-ready structures
(i.e., lists of records) so the dashboard can easily convert to plotly figures.
"""
import numpy as np
import pandas as pd
import plotly.express as px
from src.utils import to_json_serializable

def get_score_distribution_data(df: pd.DataFrame, bins: int = 30):
    series = df["total_score"].dropna().astype(float)
    if series.empty:
        return []
    counts, edges = np.histogram(series, bins=bins)
    mids = 0.5 * (edges[:-1] + edges[1:])
    out = pd.DataFrame({"midpoint": mids.tolist(), "count": counts.tolist()})
    return to_json_serializable(out)

def get_graduation_by_track_data(df: pd.DataFrame):
    tmp = df.groupby("track_name").agg(
        total_count=("graduation_status", lambda s: int(s.notna().sum())),
        graduated_count=("graduation_status", lambda s: int(s.eq(True).sum()))
    ).reset_index()
    tmp["graduation_rate"] = tmp.apply(lambda r: (r["graduated_count"] / r["total_count"]) if r["total_count"] else None, axis=1)
    return to_json_serializable(tmp.sort_values("graduation_rate", ascending=False))

def get_graduation_by_country_data(df: pd.DataFrame):
    tmp = df.groupby("country_name").agg(
        total_count=("graduation_status", lambda s: int(s.notna().sum())),
        graduated_count=("graduation_status", lambda s: int(s.eq(True).sum()))
    ).reset_index()
    tmp["graduation_rate"] = tmp.apply(lambda r: (r["graduated_count"] / r["total_count"]) if r["total_count"] else None, axis=1)
    return to_json_serializable(tmp.sort_values("graduation_rate", ascending=False))

def get_graduation_by_gender_data(df: pd.DataFrame):
    tmp = df.groupby("gender").agg(
        total_count=("graduation_status", lambda s: int(s.notna().sum())),
        graduated_count=("graduation_status", lambda s: int(s.eq(True).sum()))
    ).reset_index()
    tmp["graduation_rate"] = tmp.apply(lambda r: (r["graduated_count"] / r["total_count"]) if r["total_count"] else None, axis=1)
    return to_json_serializable(tmp)

def get_score_by_track_data(df: pd.DataFrame):
    tmp = df.groupby(["track_name", "graduation_status"])["total_score"].mean().reset_index().rename(columns={"total_score":"avg_score"})
    return to_json_serializable(tmp)

def get_experience_vs_score_data(df: pd.DataFrame, sample: int = 1000):
    if df.empty:
        return []
    sample_df = df.sample(min(sample, len(df)), random_state=42)
    keep = ["hours_per_week", "years_experience", "total_score", "graduation_status"]
    return to_json_serializable(sample_df[keep])

def get_correlation_data(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return []
    corr = numeric.corr(method="spearman").round(4)
    corr_long = corr.stack().reset_index().rename(columns={0: "correlation"})
    corr_long.columns = ["feature1", "feature2", "correlation"]
    return to_json_serializable(corr_long)

def get_feature_importance_data(model, feature_names):
    import numpy as _np
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = _np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        imp = _np.abs(_np.array(model.coef_)).ravel()
    else:
        return []
    if feature_names and len(feature_names) == len(imp):
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
    else:
        df = pd.DataFrame({"feature": [f"f{i}" for i in range(len(imp))], "importance": imp})
    df = df.sort_values("importance", ascending=False)
    return to_json_serializable(df)

def get_histogram(df, column):
    fig = px.histogram(df, x=column, title=f"Distribution of {column}")
    return fig

def get_pie_chart(df, column):
    fig = px.pie(df, names=column, title=f"Pie Chart of {column}")
    return fig

def get_3d_scatter(df, x_col, y_col, z_col, color_col=None):
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                        title=f"3D Scatter of {x_col}, {y_col}, {z_col}")
    return fig

def get_choropleth(df, location_col, value_col, title):
    fig = px.choropleth(df, locations=location_col, color=value_col,
                        locationmode="country names", title=title)
    return fig