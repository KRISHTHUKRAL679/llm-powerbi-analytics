import os
import re
import json
import time
import sqlite3
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import boto3
import google.generativeai as genai

# =========================================================
# Page setup
# =========================================================
st.set_page_config(
    page_title="AI BI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("LLM-Powered Business Intelligence Assistant")
st.caption("Upload data, store it in S3, analyze it with SQL and Gemini, generate charts, and forecast trends.")

# =========================================================
# Helpers
# =========================================================
def get_secret(name: str, default=None):
    if name in st.secrets:
        return st.secrets.get(name, default)
    return os.getenv(name, default)

def init_gemini():
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def init_s3():
    region = get_secret("AWS_REGION")
    bucket = get_secret("S3_BUCKET")
    if not bucket:
        return None, None
    try:
        client = boto3.client("s3", region_name=region)
        return client, bucket
    except Exception:
        return None, bucket

def safe_read_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean column names
    df.columns = [
        re.sub(r"\s+", "_", str(c).strip()).lower()
        for c in df.columns
    ]

    # Drop duplicates
    df = df.drop_duplicates()

    # Normalize empty strings
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    # Fill missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna("Unknown")

    return df

def upload_to_s3(s3, bucket, file_name, file_bytes):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    key = f"{timestamp}_{file_name}"
    s3.put_object(Bucket=bucket, Key=key, Body=file_bytes)
    return key

def list_s3_files(s3, bucket):
    try:
        objects = s3.list_objects_v2(Bucket=bucket)
        if "Contents" not in objects:
            return []
        return [obj["Key"] for obj in objects["Contents"]]
    except Exception:
        return []

def load_csv_from_s3(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])

def dataframe_sqlite_connection(df: pd.DataFrame):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql("data", conn, index=False, if_exists="replace")
    return conn

def extract_json_from_text(text: str):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

def ask_gemini(model, prompt: str):
    if model is None:
        return "Gemini is not configured. Set GEMINI_API_KEY to enable AI responses."
    try:
        response = model.generate_content(prompt)
        return getattr(response, "text", "") or "No response returned."
    except Exception as e:
        return f"Gemini error: {e}"

def build_dataset_context(df: pd.DataFrame) -> str:
    sample = df.head(5).to_string(index=False)
    stats = df.describe(include="all").transpose().fillna("").to_string()
    dtypes = "\n".join([f"{c}: {t}" for c, t in df.dtypes.items()])
    return f"""
Columns:
{', '.join(df.columns)}

Data types:
{dtypes}

Sample rows:
{sample}

Summary stats:
{stats}
"""

def suggest_chart_config(model, user_query: str, df: pd.DataFrame):
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    prompt = f"""
You are a BI assistant. Convert the user request into a chart configuration.

Dataset columns: {columns}
Numeric columns: {numeric_cols}

User request:
{user_query}

Return ONLY valid JSON with these keys:
chart_type: one of ["bar", "line", "scatter", "histogram"]
x_column: a column name or null
y_column: a column name or null
color_column: a column name or null

Rules:
- Use a time/date-like column for line charts if the query mentions trend, over time, daily, monthly, yearly.
- For comparisons, use bar charts.
- For relationships/correlation, use scatter charts.
- For distributions, use histogram.
- Pick only columns that exist in the dataset.
"""

    raw = ask_gemini(model, prompt)
    cfg = extract_json_from_text(raw)
    if cfg:
        return cfg

    # Heuristic fallback
    lower = user_query.lower()
    date_like = next((c for c in columns if any(k in c.lower() for k in ["date", "time", "day", "month", "year"])), None)
    num = numeric_cols[0] if numeric_cols else None
    cat = next((c for c in columns if c not in numeric_cols), columns[0] if columns else None)

    if any(k in lower for k in ["trend", "over time", "growth", "daily", "monthly", "yearly"]):
        return {"chart_type": "line", "x_column": date_like or cat, "y_column": num, "color_column": None}
    if any(k in lower for k in ["correlation", "relationship", "scatter", "compare two"]):
        return {"chart_type": "scatter", "x_column": num, "y_column": numeric_cols[1] if len(numeric_cols) > 1 else num, "color_column": None}
    if any(k in lower for k in ["distribution", "histogram", "spread"]):
        return {"chart_type": "histogram", "x_column": num, "y_column": None, "color_column": None}
    return {"chart_type": "bar", "x_column": cat, "y_column": num, "color_column": None}

def create_chart(df: pd.DataFrame, cfg: dict):
    chart_type = (cfg or {}).get("chart_type", "bar")
    x_col = (cfg or {}).get("x_column")
    y_col = (cfg or {}).get("y_column")
    color_col = (cfg or {}).get("color_column")

    if chart_type == "histogram":
        if not y_col:
            return None, "No numeric column available for histogram."
        fig = px.histogram(df, x=y_col, color=color_col)
        return fig, None

    if not x_col and len(df.columns) > 0:
        x_col = df.columns[0]
    if not y_col:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        y_col = num_cols[0] if num_cols else None
    if not y_col:
        return None, "No numeric column available for charting."

    if chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, color=color_col)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
    else:
        fig = px.bar(df, x=x_col, y=y_col, color=color_col)

    return fig, None

def generate_insights(model, df: pd.DataFrame):
    context = build_dataset_context(df)
    prompt = f"""
You are an expert business intelligence analyst.
Analyze the dataset and provide:
1. Key trends
2. Possible anomalies
3. Business recommendations
4. Any data quality issues

Keep the response concise but useful.

Dataset:
{context}
"""
    return ask_gemini(model, prompt)

def answer_question(model, df: pd.DataFrame, question: str):
    context = build_dataset_context(df)
    prompt = f"""
You are an expert business intelligence analyst.

Dataset:
{context}

Question:
{question}

Answer using business language.
Include insights, trends, and recommendations where relevant.
"""
    return ask_gemini(model, prompt)

def forecast_numeric_series(df: pd.DataFrame, target_col: str, steps: int = 5):
    series = pd.to_numeric(df[target_col], errors="coerce").dropna().reset_index(drop=True)
    if len(series) < 3:
        return None, "Not enough numeric data to forecast."

    x = np.arange(len(series))
    y = series.values.astype(float)

    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(series), len(series) + steps)
    future_y = slope * future_x + intercept

    forecast_df = pd.DataFrame({
        "step": np.arange(1, steps + 1),
        "predicted_value": future_y
    })
    return forecast_df, None

# =========================================================
# State
# =========================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "df" not in st.session_state:
    st.session_state.df = None

if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

# =========================================================
# Clients
# =========================================================
model = init_gemini()
s3, BUCKET = init_s3()

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Controls")

use_preprocessing = st.sidebar.checkbox("Enable preprocessing", value=True)
show_missing_report = st.sidebar.checkbox("Show missing-value report", value=True)

if st.sidebar.button("Reset current session"):
    st.session_state.df = None
    st.session_state.clean_df = None
    st.session_state.chat_history = []
    st.session_state.selected_file = None
    st.rerun()

st.sidebar.subheader("Environment")
st.sidebar.write(f"AWS bucket: `{BUCKET or 'Not set'}`")
st.sidebar.write(f"Gemini: `{'Enabled' if model else 'Not configured'}`")

# =========================================================
# Upload section
# =========================================================
st.subheader("Upload Dataset")

uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        raw_df = safe_read_csv(file_bytes)

        if use_preprocessing:
            clean_df = preprocess_data(raw_df)
        else:
            clean_df = raw_df.copy()

        st.session_state.df = raw_df
        st.session_state.clean_df = clean_df

        if s3 and BUCKET:
            key = upload_to_s3(s3, BUCKET, uploaded_file.name, file_bytes)
            st.success(f"Uploaded to S3: {key}")
            st.session_state.selected_file = key
        else:
            st.warning("S3 is not configured. Dataset is loaded locally only.")

    except Exception as e:
        st.error(f"Failed to load file: {e}")

# =========================================================
# Load from cloud
# =========================================================
st.subheader("Load Dataset from Cloud")

if s3 and BUCKET:
    files = list_s3_files(s3, BUCKET)
    selected_file = st.selectbox("Select dataset from S3", [""] + files)

    if selected_file:
        try:
            cloud_df = load_csv_from_s3(s3, BUCKET, selected_file)
            if use_preprocessing:
                cloud_df = preprocess_data(cloud_df)
            st.session_state.df = cloud_df
            st.session_state.clean_df = cloud_df
            st.session_state.selected_file = selected_file
            st.success(f"Loaded {selected_file} from S3")
        except Exception as e:
            st.error(f"Could not load dataset from S3: {e}")
else:
    st.info("Set AWS_REGION and S3_BUCKET to enable cloud loading.")

df = st.session_state.clean_df

# =========================================================
# Main content
# =========================================================
if df is not None:
    row_count, col_count = df.shape
    missing_cells = int(df.isna().sum().sum())
    numeric_count = len(df.select_dtypes(include="number").columns)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{row_count:,}")
    c2.metric("Columns", f"{col_count:,}")
    c3.metric("Numeric Columns", f"{numeric_count:,}")
    c4.metric("Missing Cells", f"{missing_cells:,}")

    tabs = st.tabs(["Data", "Visualization", "SQL Explorer", "AI Insights", "Forecast", "Cloud Info"])

    # -------------------- Data tab --------------------
    with tabs[0]:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Column Types**")
            dtype_df = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes.values]})
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        with col2:
            if show_missing_report:
                st.markdown("**Missing Values**")
                missing_df = pd.DataFrame({
                    "column": df.columns,
                    "missing_count": df.isna().sum().values,
                    "missing_percent": (df.isna().sum().values / len(df) * 100).round(2)
                }).sort_values("missing_count", ascending=False)
                st.dataframe(missing_df, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download cleaned CSV",
            data=csv_bytes,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    # -------------------- Visualization tab --------------------
    with tabs[1]:
        st.subheader("Manual Chart Builder")

        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        chart_type = st.selectbox("Chart type", ["Bar", "Line", "Scatter", "Histogram"])
        x_col = st.selectbox("X axis", all_cols, index=0 if all_cols else None)
        y_col = st.selectbox("Y axis", numeric_cols, index=0 if numeric_cols else None)

        if chart_type == "Histogram":
            if y_col:
                fig = px.histogram(df, x=y_col)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric column available for histogram.")
        else:
            if x_col and y_col:
                if chart_type == "Bar":
                    fig = px.bar(df, x=x_col, y=y_col)
                elif chart_type == "Line":
                    fig = px.line(df, x=x_col, y=y_col)
                else:
                    fig = px.scatter(df, x=x_col, y=y_col)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Select valid columns for plotting.")

        st.divider()
        st.subheader("AI Chart Generator")
        chart_question = st.text_input("Describe the chart you want", placeholder="Example: Show monthly sales trend")
        if chart_question:
            cfg = suggest_chart_config(model, chart_question, df)
            fig, err = create_chart(df, cfg)
            st.write("Generated configuration:", cfg)

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(err or "Could not generate chart.")

    # -------------------- SQL tab --------------------
    with tabs[2]:
        st.subheader("SQL Explorer")
        st.caption("Query the dataset using SQL syntax. The table name is `data`.")

        sql_query = st.text_area(
            "Enter SQL query",
            value="SELECT * FROM data LIMIT 10",
            height=120
        )

        if st.button("Run SQL Query"):
            try:
                conn = dataframe_sqlite_connection(df)
                result = pd.read_sql_query(sql_query, conn)
                st.dataframe(result, use_container_width=True)
            except Exception as e:
                st.error(f"SQL error: {e}")

    # -------------------- AI Insights tab --------------------
    with tabs[3]:
        st.subheader("Auto Insights")
        if st.button("Generate AI Insights"):
            with st.spinner("Analyzing dataset..."):
                insights = generate_insights(model, df)
            st.write(insights)

        st.divider()
        st.subheader("Ask Questions About the Data")
        user_question = st.text_input("Ask a question", placeholder="Example: Which category contributes most to revenue?")
        if user_question:
            with st.spinner("Thinking..."):
                answer = answer_question(model, df, user_question)

            st.session_state.chat_history.append(("User", user_question))
            st.session_state.chat_history.append(("Assistant", answer))

            st.write(answer)

        if st.session_state.chat_history:
            st.markdown("**Conversation History**")
            for role, msg in st.session_state.chat_history[-10:]:
                if role == "User":
                    st.markdown(f"**You:** {msg}")
                else:
                    st.markdown(f"**Assistant:** {msg}")

    # -------------------- Forecast tab --------------------
    with tabs[4]:
        st.subheader("Forecasting")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            target_col = st.selectbox("Select numeric column to forecast", numeric_cols)
            horizon = st.slider("Forecast steps", min_value=1, max_value=30, value=5)

            if st.button("Run Forecast"):
                forecast_df, err = forecast_numeric_series(df, target_col, horizon)
                if forecast_df is not None:
                    st.dataframe(forecast_df, use_container_width=True)

                    fig = px.line(forecast_df, x="step", y="predicted_value", markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(err)
        else:
            st.info("No numeric column found for forecasting.")

    # -------------------- Cloud info tab --------------------
    with tabs[5]:
        st.subheader("Cloud Integration")
        if s3 and BUCKET:
            st.success("AWS S3 storage is enabled.")
            st.write(f"Bucket: `{BUCKET}`")
            if st.session_state.selected_file:
                st.write(f"Current file: `{st.session_state.selected_file}`")

            files = list_s3_files(s3, BUCKET)
            if files:
                cloud_df = pd.DataFrame({"S3 Objects": files})
                st.dataframe(cloud_df, use_container_width=True, hide_index=True)
            else:
                st.info("No files found in S3.")
        else:
            st.warning("S3 is not configured in the environment.")

else:
    st.info("Upload a CSV file or load one from S3 to begin.")