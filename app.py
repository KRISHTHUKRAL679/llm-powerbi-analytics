import streamlit as st
import pandas as pd
import plotly.express as px
import os
from openai import OpenAI

st.set_page_config(page_title="AI BI Assistant", layout="wide")

st.title("LLM Powered Business Intelligence Assistant")

# Load dataset
uploaded_file = st.file_uploader("Upload CSV dataset")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Simple chart
    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) >= 2:
        x = st.selectbox("X Axis", df.columns)
        y = st.selectbox("Y Axis", numeric_cols)

        fig = px.bar(df, x=x, y=y)
        st.plotly_chart(fig)

    # LLM Chat Section
    st.subheader("Ask Questions About Your Data")

    user_question = st.text_input("Ask something about the dataset")

    if user_question:

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        summary = df.describe().to_string()

        prompt = f"""
        You are a business data analyst.

        Dataset statistics:
        {summary}

        User question:
        {user_question}

        Provide insights explaining trends, possible causes and recommendations.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )

        answer = response.choices[0].message.content

        st.subheader("AI Insight")
        st.write(answer)

# Power BI embed
st.subheader("Power BI Dashboard")

powerbi_url = st.text_input("Paste Power BI Embed URL")

if powerbi_url:
    st.components.v1.iframe(powerbi_url, height=600)
