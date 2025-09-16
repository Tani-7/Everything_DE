import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import numpy as np

# Graduation Rate by Track
def plot_graduation_rate_by_track(df):
    track_grad = df.groupby("track_name")["graduation_status"].mean().reset_index()
    fig = px.bar(
        track_grad,
        x="track_name",
        y="graduation_status",
        title="Graduation Rate by Track",
        color="graduation_status",
        text="graduation_status",
        labels={"graduation_status": "Graduation Rate"},
    )
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig.update_yaxes(tickformat=".0%")
    return fig

# Graduation Rate by Country
def plot_graduation_rate_by_country(df):
    country_grad = df.groupby("country_name")["graduation_status"].mean().reset_index()
    fig = px.bar(
        country_grad,
        x="country_name",
        y="graduation_status",
        title="Graduation Rate by Country",
        color="graduation_status",
        text="graduation_status",
        labels={"graduation_status": "Graduation Rate"},
    )
    fig.update_traces(texttemplate='%{text:.1%}', textposition="outside")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    return fig


# Score Distribution

def plot_score_distribution(df):
    fig = px.histogram(
        df,
        x="total_score",
        color="graduation_status",
        nbins=30,
        barmode="overlay",
        title="Score Distribution by Graduation Status",
    )
    fig.update_traces(opacity=0.7)
    return fig

# Average Score by Track

def plot_score_by_track(df):
    fig = px.box(
        df,
        x="track_name",
        y="total_score",
        color="graduation_status",
        points="all",
        title="Score Distribution by Track & Graduation Status",
    )
    return fig

# Feature Importance (from model)

def plot_feature_importance(model, feature_names):
    importance = model.coef_[0]
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)
    
    fig = px.bar(
        imp_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance (Logistic Regression)",
        color="importance",
        color_continuous_scale="Viridis"
    )
    return fig


# Correlation Heatmap

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r"
    )
    return fig


# Graduation vs Gender

def plot_graduation_by_gender(df):
    gender_grad = df.groupby("gender")["graduation_status"].mean().reset_index()
    fig = px.bar(
        gender_grad,
        x="gender",
        y="graduation_status",
        title="Graduation Rate by Gender",
        text="graduation_status",
        color="graduation_status"
    )
    fig.update_traces(texttemplate='%{text:.1%}', textposition="outside")
    fig.update_yaxes(tickformat=".0%")
    return fig

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", colorbar=False)
    fig = px.imshow(cm.confusion_matrix, text_auto=True, labels=dict(x="Predicted", y="Actual"))
    fig.update_layout(title="Confusion Matrix")
    return fig

def plot_roc_curve(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.3f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig

def plot_logistic_coefficients(model, feature_names):
    # extract coefficients
    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]
    odds_ratios = np.exp(coefs)
    fig = px.bar(
        x=feature_names,
        y=odds_ratios,
        labels={"x": "Feature", "y": "Odds Ratio"},
        title="Logistic Regression Feature Odds Ratios"
    )
    fig.update_layout(yaxis_type="log")  # easier to interpret wide-ranging ORs
    return fig
