import streamlit as st
import pandas as pd
import plotly.express as px
import os
import google.generativeai as genai
import streamlit.components.v1 as components

st.set_page_config(page_title="AI BI Assistant", layout="wide")

st.title("LLM Powered Business Intelligence Assistant")

# -------------------------------
# Gemini Setup
# -------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------------
# Upload Dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV dataset")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ Smart demo message
    st.success("Dataset loaded successfully into AI system")
    st.info("Power BI dashboards can be refreshed for updated insights")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Chart Section
    # -------------------------------
    st.subheader("Create Chart")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) >= 1:
        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter"])
        x = st.selectbox("X Axis", all_cols)
        y = st.selectbox("Y Axis", numeric_cols)

        if chart_type == "Bar":
            fig = px.bar(df, x=x, y=y)
        elif chart_type == "Line":
            fig = px.line(df, x=x, y=y)
        else:
            fig = px.scatter(df, x=x, y=y)

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # AI Insights Section
    # -------------------------------
    st.subheader("Ask Questions About Your Data")

    user_question = st.text_input("Ask something about the dataset")

    if user_question:

        summary_stats = df.describe(include='all').fillna("").to_string()
        columns_info = ", ".join(df.columns)
        sample_data = df.head(5).to_string()

        correlation_info = ""
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().to_string()
            correlation_info = f"\nCorrelation Matrix:\n{corr}"

        prompt = f"""
You are an expert business intelligence analyst.

Dataset columns:
{columns_info}

Sample data:
{sample_data}

Statistical summary:
{summary_stats}

{correlation_info}

User question:
{user_question}

Instructions:
- Give clear business insights
- Explain trends and possible causes
- Highlight anomalies
- Suggest actionable recommendations
"""

        with st.spinner("Analyzing..."):
            response = model.generate_content(prompt)
            answer = response.text

        st.subheader("AI Insight")
        st.write(answer)

# -------------------------------
# Power BI Section
# -------------------------------
st.subheader("Power BI Dashboard")

POWERBI_URL = "https://app.powerbi.com/reportEmbed?reportId=5103c909-2686-4b16-b1f1-57d566052411&autoAuth=true&ctid=27282fdd-4c0b-4dfb-ba91-228cd83fdf71"

components.iframe(POWERBI_URL, height=650, scrolling=True)

if powerbi_url:
    st.components.v1.iframe(powerbi_url, height=600)
