import streamlit as st
import pandas as pd
import plotly.express as px
import io
import base64
import requests
import os  # ✅ Added to load environment variables

# ===========================
# ExcelDash Prototype
# ===========================

st.set_page_config(page_title="ExcelDash", layout="wide")

st.title("📊 ExcelDash – Generate Dashboards from Excel")
st.markdown("Upload your Excel or CSV file and generate a data dashboard instantly using AI or heuristics.")

# Sidebar options
st.sidebar.header("Options")
use_gemini = st.sidebar.checkbox("Use Gemini LLM (if API key available)", value=False)

# ✅ Securely load API key from Streamlit Secrets
api_key = os.getenv("GEMINI_API_KEY")

# If not found in secrets, you can still test manually by pasting temporarily:
# api_key = "AIzaYourKey"  # ⚠️ Remove this before committing

max_charts = st.sidebar.slider("Number of charts to generate", 1, 10, 5)

# File uploader
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])

def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

def heuristic_chart_plan(df):
    """
    Basic heuristic: generate chart suggestions based on column types.
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    non_numeric = df.select_dtypes(exclude='number').columns.tolist()
    charts = []

    # Simple logic: pair first non-numeric with each numeric
    for cat in non_numeric[:2]:
        for num in numeric_cols[:3]:
            charts.append({
                "type": "bar",
                "x": cat,
                "y": num,
                "title": f"{num} by {cat}"
            })
    return charts[:max_charts]

def gemini_chart_plan(df, api_key):
    """Generate chart suggestions using Gemini API."""
    import json

    url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}

    prompt = f"""
    You are a data visualization assistant.
    Based on this dataset with columns: {list(df.columns)},
    suggest up to 5 useful charts in JSON format.
    Use this format:
    [
      {{"type": "bar", "x": "column_name", "y": "column_name", "title": "Chart title"}},
      ...
    ]
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(f"{url}?key={api_key}", headers=headers, json=payload)
        result = response.json()

        if "candidates" in result:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        else:
            st.warning(f"Gemini plan failed, fallback to heuristic. Raw response: {result}")
            return heuristic_chart_plan(df)

    except Exception as e:
        st.warning(f"Gemini plan failed, fallback to heuristic. Error: {e}")
        return heuristic_chart_plan(df)

def generate_dashboard(df, chart_plan):
    """
    Generate Plotly charts based on plan.
    """
    charts = []
    for ch in chart_plan:
        chart_type = ch.get("type", "bar")
        x, y, title = ch.get("x"), ch.get("y"), ch.get("title", "Chart")

        if x not in df.columns or y not in df.columns:
            continue

        if chart_type == "line":
            fig = px.line(df, x=x, y=y, title=title)
        elif chart_type == "pie":
            fig = px.pie(df, names=x, values=y, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, title=title)
        else:
            fig = px.bar(df, x=x, y=y, title=title)

        charts.append(fig)
    return charts

def export_dashboard(charts):
    """
    Export charts to standalone HTML file.
    """
    html_parts = ["<html><head><title>ExcelDash Dashboard</title></head><body>"]
    for fig in charts:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    html_parts.append("</body></html>")
    html_content = "\n".join(html_parts)
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="exceldash_dashboard.html">📥 Download Dashboard (HTML)</a>'
    return href

if uploaded_file:
    df = load_data(uploaded_file)
    st.success(f"Data loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns.")
    st.dataframe(df.head())

    if st.button("🚀 Generate Dashboard"):
        with st.spinner("Generating dashboard..."):
            if use_gemini and api_key:
                charts = gemini_chart_plan(df, api_key)
            else:
                charts = heuristic_chart_plan(df)

            # Render the charts
            for chart in charts[:max_charts]:
                st.subheader(chart["title"])
                fig = None
                if chart["type"] == "bar":
                    fig = px.bar(df, x=chart["x"], y=chart["y"], title=chart["title"])
                elif chart["type"] == "line":
                    fig = px.line(df, x=chart["x"], y=chart["y"], title=chart["title"])
                elif chart["type"] == "pie":
                    fig = px.pie(df, names=chart["x"], values=chart["y"], title=chart["title"])
                elif chart["type"] == "scatter":
                    fig = px.scatter(df, x=chart["x"], y=chart["y"], title=chart["title"])

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown(export_dashboard(charts), unsafe_allow_html=True)

else:
    st.info("👆 Upload an Excel or CSV file to begin.")
