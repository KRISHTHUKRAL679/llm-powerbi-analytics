import streamlit as st
import pandas as pd
import plotly.express as px
import os
import google.generativeai as genai
import boto3
from io import StringIO

st.set_page_config(page_title="AI BI Assistant", layout="wide")
st.title("LLM Powered Business Intelligence Assistant")

# -------------------------------
# Gemini Setup
# -------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------------
# AWS S3 Setup
# -------------------------------
s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION")
)
BUCKET = os.getenv("S3_BUCKET")

# -------------------------------
# Upload Dataset to Cloud
# -------------------------------
st.subheader("Upload Dataset (Cloud Storage Enabled)")

uploaded_file = st.file_uploader("Upload CSV dataset")

df = None

if uploaded_file:
    file_name = uploaded_file.name

    # Read file into memory (IMPORTANT)
    file_bytes = uploaded_file.read()

    # Upload to S3
    s3.put_object(
        Bucket=BUCKET,
        Key=file_name,
        Body=file_bytes
    )

    st.success(f"{file_name} uploaded to cloud (S3)")

    # Read into pandas
    from io import BytesIO
    df = pd.read_csv(BytesIO(file_bytes))

# -------------------------------
# Load from Cloud
# -------------------------------
st.subheader("Load Dataset from Cloud")

objects = s3.list_objects_v2(Bucket=BUCKET)

file_options = []
if "Contents" in objects:
    file_options = [obj["Key"] for obj in objects["Contents"]]

selected_file = st.selectbox("Select dataset from S3", file_options)

if selected_file:
    obj = s3.get_object(Bucket=BUCKET, Key=selected_file)
    df = pd.read_csv(obj["Body"])

    st.success(f"Loaded {selected_file} from cloud")

# -------------------------------
# If dataset available
# -------------------------------
if df is not None:

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Charts
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
    # AI Insights
    # -------------------------------
    st.subheader("Ask Questions About Your Data")

    user_question = st.text_input("Ask something about the dataset")

    if user_question:

        summary_stats = df.describe(include='all').fillna("").to_string()
        columns_info = ", ".join(df.columns)
        sample_data = df.head(5).to_string()

        prompt = f"""
You are an expert business intelligence analyst.

Columns: {columns_info}

Sample:
{sample_data}

Stats:
{summary_stats}

Question:
{user_question}

Give business insights, trends, and recommendations.
"""

        with st.spinner("Analyzing..."):
            response = model.generate_content(prompt)
            st.write(response.text)

# -------------------------------
# Cloud Info (for demo)
# -------------------------------
st.subheader("Cloud Integration")

st.info("Datasets are stored in AWS S3 for persistent and scalable access.")
