import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Graduation Insights", layout="wide")
API = st.sidebar.text_input("API URL", value="http://localhost:8000").rstrip("/")
st.title("🎓 Graduation Insights Dashboard (API-driven)")

def api_get(path, params=None):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None

def api_post(path, json_payload=None):
    try:
        r = requests.post(f"{API}{path}", json=json_payload, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None

tab1, tab2, tab3 = st.tabs(["Data", "Visuals", "Predict"])

with tab1:
    st.header("Sample Data & Summary")
    rows = st.slider("Sample rows", 5, 200, 20)
    data = api_get(f"/sample-data?limit={rows}")
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    summary = api_get("/eda/summary")
    if summary:
        st.write("### Summary")
        st.json(summary)

with tab2:
    st.header("Visualizations")
    viz = st.selectbox("Choose visualization", [
        "distribution", "graduation_by_track", "graduation_by_country",
        "graduation_by_gender", "score_by_track", "experience_vs_score", "correlations"
    ])
    if viz == "distribution":
        bins = st.slider("Bins", 10, 80, 30)
        data = api_get(f"/viz/distribution?bins={bins}")
        if data:
            df_v = pd.DataFrame(data)
            fig = px.bar(df_v, x="midpoint", y="count", title="Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
    elif viz == "graduation_by_track":
        data = api_get("/viz/graduation_by_track")
        if data:
            df_v = pd.DataFrame(data)
            fig = px.bar(df_v, x="track_name", y="graduation_rate", title="Graduation Rate by Track")
            st.plotly_chart(fig, use_container_width=True)
            top = st.slider("Top N tracks for pie", 3, 20, 6)
            top_df = df_v.sort_values("graduation_rate", ascending=False).head(top)
            fig2 = px.pie(top_df, values="graduated_count", names="track_name", title="Graduated Count (Top Tracks)")
            st.plotly_chart(fig2, use_container_width=True)
    elif viz == "graduation_by_country":
        data = api_get("/viz/graduation_by_country")
        if data:
            df_v = pd.DataFrame(data)
            try:
                fig = px.choropleth(df_v, locations="country_name", locationmode="country names",
                                    color="graduation_rate", hover_name="country_name", title="Graduation Rate by Country")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                fig = px.bar(df_v, x="country_name", y="graduation_rate", title="Graduation Rate by Country")
                st.plotly_chart(fig, use_container_width=True)
    elif viz == "graduation_by_gender":
        data = api_get("/viz/graduation_by_gender")
        if data:
            df_v = pd.DataFrame(data)
            fig = px.pie(df_v, values="graduated_count", names="gender", title="Graduated Count by Gender")
            st.plotly_chart(fig, use_container_width=True)
    elif viz == "score_by_track":
        data = api_get("/viz/score_by_track")
        if data:
            df_v = pd.DataFrame(data)
            st.dataframe(df_v, use_container_width=True)
            fig = px.box(df_v, x="track_name", y="avg_score", title="Average Score by Track")
            st.plotly_chart(fig, use_container_width=True)
    elif viz == "experience_vs_score":
        sample = st.number_input("Max sample rows", 100, 5000, 1000)
        data = api_get(f"/viz/experience_vs_score?sample={sample}")
        if data:
            df_v = pd.DataFrame(data)
            try:
                fig = px.scatter_3d(df_v, x="hours_per_week", y="years_experience", z="total_score",
                                    color=df_v["graduation_status"].astype(str), title="Hours vs Experience vs Score (3D)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(df_v)
    elif viz == "correlations":
        data = api_get("/viz/correlations")
        if data:
            df_v = pd.DataFrame(data)
            try:
                pivot = df_v.pivot(index="feature1", columns="feature2", values="correlation")
                fig = px.imshow(pivot, text_auto=True, aspect="auto", title="Spearman Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(df_v)

with tab3:
    st.header("Predict Graduation (via API)")
    models = api_get("/models")
    if not models:
        st.info("No models available. Train models via API first.")
        st.stop()
    model_choice = st.selectbox("Model", models["available_models"])

    with st.form("pred"):
        total_score = st.number_input("Total score", 0.0, 100.0, 70.0)
        hours = st.number_input("Hours / week", 0.0, 100.0, 15.0)
        years = st.number_input("Years experience", 0.0, 40.0, 1.0)
        skill = st.number_input("Skill level (0-10)", 0.0, 10.0, 5.0)
        track = st.text_input("Track name", "Data Science")
        country = st.text_input("Country name", "Kenya")
        gender = st.selectbox("Gender", ["male", "female", "other"])
        age_range = st.text_input("Age range", "18-24")
        heard_about = st.text_input("Heard about", "Friend")
        submit = st.form_submit_button("Predict")

    if submit:
        payload = {
            "total_score": float(total_score),
            "hours_per_week": float(hours),
            "years_experience": float(years),
            "skill_level": float(skill),
            "track_name": str(track),
            "country_name": str(country),
            "gender": str(gender),
            "age_range": str(age_range),
            "heard_about": str(heard_about),
        }
        res = api_post(f"/predict/{model_choice}", json_payload=payload)
        if res:
            prob = res.get("probability")
            predicted = res.get("prediction")
            st.success(f"Prediction: {'Graduated' if predicted else 'Not graduated'} — probability: {prob}")
