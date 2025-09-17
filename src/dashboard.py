import streamlit as st
import pandas as pd
from src.data_utils import get_engine, load_participants
from src.stats_models import run_logistic_regression
from src.stats_inf import run_hypothesis_tests
import stats_viz as viz
from pathlib import Path

st.set_page_config(page_title="Graduation Insights Dashboard", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #FAFAFA; }
        .stPlotlyChart { border: 1px solid #CCC; border-radius: 10px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)


# header

st.title("🎓 Graduation Insights Dashboard")
st.markdown("""
Play with this Dashboard to explore student graduation patterns, model predictions, and key metrics interactively.  
Use the sidebar to filter data and inspect results.
""")


# load data

df = load_and_clean_data()


# sidebar filters

st.sidebar.header("🔍 Filters")
tracks = df["track_name"].unique() if "track_name" in df.columns else []
countries = df["country_name"].unique() if "country_name" in df.columns else []
genders = df["gender"].unique() if "gender" in df.columns else []

selected_tracks = st.sidebar.multiselect("Select Track(s)", tracks, default=tracks)
selected_countries = st.sidebar.multiselect("Select Country(s)", countries, default=countries)
selected_genders = st.sidebar.multiselect("Select Gender(s)", genders, default=genders)

df_filtered = df[
    (df["track_name"].isin(selected_tracks)) &
    (df["country_name"].isin(selected_countries)) &
    (df["gender"].isin(selected_genders))
]


# KPI cards

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("👥 Total Participants", len(df_filtered))
with col2:
    st.metric("🎓 Graduation Rate", f"{df_filtered['graduation_status'].mean()*100:.1f}%")
with col3:
    st.metric("📊 Avg Total Score", f"{df_filtered['total_score'].mean():.2f}")


# exploratory plots

st.subheader("📈 Graduation Patterns")
st.plotly_chart(viz.plot_graduation_rate_by_track(df_filtered), use_container_width=True)
st.plotly_chart(viz.plot_graduation_rate_by_country(df_filtered), use_container_width=True)
st.plotly_chart(viz.plot_graduation_by_gender(df_filtered), use_container_width=True)

st.subheader("📊 Score Distributions")
st.plotly_chart(viz.plot_score_distribution(df_filtered), use_container_width=True)
st.plotly_chart(viz.plot_score_by_track(df_filtered), use_container_width=True)


# correlation heatmap

st.subheader("🧮 Correlations")
st.plotly_chart(viz.plot_correlation_heatmap(df_filtered), use_container_width=True)


# statistical inference

st.subheader("📋 Chi-Square & Hypothesis Tests")
chi_results = run_hypothesis_tests(df_filtered)
st.json(chi_results)


# logistic regression

st.subheader("🧮 Logistic Regression")
X, y, model, X_train, X_test, y_train, y_test = run_logistic_regression(df_filtered)
metrics = evaluate_model(model, X_test, y_test)

st.write("**Model Evaluation Metrics:**")
st.json(metrics)

st.plotly_chart(viz.plot_confusion_matrix(model, X_test, y_test), use_container_width=True)
st.plotly_chart(viz.plot_roc_curve(model, X_test, y_test), use_container_width=True)
st.plotly_chart(viz.plot_logistic_coefficients(model, X.columns), use_container_width=True)


# download report

st.subheader("📄 Download Full Report")

BASE_DIR = Path(__file__).resolve().parent
pdf_path = BASE_DIR / ".." / "private_files" / "Omwansa_C._Omwansa - Graduation Outcomes Analysis Report.pdf"
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

st.download_button(
    label="Download PDF Report",
    data=pdf_bytes,
    file_name="Graduation_Outcomes_Report.pdf",
    mime="application/pdf"
)
