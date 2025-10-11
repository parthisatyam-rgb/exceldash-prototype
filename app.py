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
    """
    Call Gemini LLM to suggest chart types and fields.
    """
    try:
        import requests, json, re

        # Use a small sample for free-tier Gemini
        data_sample = df.head(5).to_csv(index=False)

        prompt = f"""
        You are a data visualization assistant.
        Analyze the dataset sample below and return a JSON list of chart suggestions.
        Each suggestion should include: chart type (bar, line, pie, scatter), x column, y column, and title.

        Example:
        [
          {{"type": "bar", "x": "Category", "y": "Sales", "title": "Sales by Category"}}
        ]

        Dataset sample:
        {data_sample}
        """

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            params={"key": api_key},
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ]
            },
            timeout=30
        )

        # Debug log for development
        st.write("Raw Gemini response:", response.text)

        if not response.text.strip():
            raise Exception("Empty response from Gemini API")

        data = response.json()

        # Extract text safely
        text = ""
        if "candidates" in data:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        elif "output" in data:
            text = data["output"][0]["content"]
        elif "contents" in data:
            parts = data["contents"][0].get("parts", [])
            if parts and "text" in parts[0]:
                text = parts[0]["text"]

        text = re.sub(r"```json|```", "", text)
        charts = json.loads(text)
        return charts

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
        with st.spinner("Analyzing and generating dashboard..."):
          if use_gemini and api_key:
                chart_plan = gemini_chart_plan(df, GEMINI_API_KEY)
            else:
                chart_plan = heuristic_chart_plan(df)

            charts = generate_dashboard(df, chart_plan)
            for fig in charts:
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(export_dashboard(charts), unsafe_allow_html=True)
else:
    st.info("👆 Upload an Excel or CSV file to begin.")
