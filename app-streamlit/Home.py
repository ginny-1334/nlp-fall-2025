import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import base64

from utils import initialize_workspace
from functions.database import SELECTED_COLUMNS
from functions import database as db
from components.header import render_header

# Initialize workspace path and imports
initialize_workspace()

st.set_page_config(
    layout="wide",
    page_title="Job Intelligence Platform",
    initial_sidebar_state="expanded"
)

# Header & footer are rendered by the router (streamlit_app.py) when using st.navigation
# CSS is loaded globally from styles/app.css in the router
# Fallback: load CSS here if page is run directly
try:
    css_path = os.path.join(os.path.dirname(__file__), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# CSS is loaded from styles/app.css

def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


sps_logo = ""
main_logo = ""
sg_logo = ""

# Override with requested hard-coded benchmark numbers
total_transactions = 1646539
unique_companies = 8706
job_titles = 10767


# Build metrics list using available values, fallback to defaults when missing
metrics_data = [
    {
        "label": "Job Postings Analyzed",
        "value": (f"{total_transactions:,}" if total_transactions not in (None, 0) else "0"),
        "delta": "",
        "context": "",
    },
    {
        "label": "Unique Companies",
        "value": (f"{unique_companies:,}" if unique_companies not in (None, 0) else "0"),
        "delta": "",
        "context": "",
    },
    {
        "label": "Unique Job Titles",
        "value": (f"{job_titles:,}" if job_titles not in (None, 0) else "0"),
        "delta": "",
        "context": "",
    },
]

# persist to session for smoother navigation
st.session_state['home_metrics'] = {'overview': metrics_data}


session_metrics = st.session_state.get("home_metrics") if "home_metrics" in st.session_state else None
if isinstance(session_metrics, list) and session_metrics:
    metrics_data = session_metrics
elif isinstance(session_metrics, dict):
    overview_metrics = session_metrics.get("overview")
    if isinstance(overview_metrics, list) and overview_metrics:
        metrics_data = overview_metrics

metric_cards = []
for metric in metrics_data:
    label = str(metric.get("label", "")).strip()
    value = str(metric.get("value", "")).strip()
    delta_text = str(metric.get("delta", "")).strip()
    context_text = str(metric.get("context", "")).strip()
    
    trend_html = ""
    if delta_text:
        trend_class = "trend-up" if delta_text.startswith("+") else "trend-down"
        trend_icon = "↑" if delta_text.startswith("+") else "↓"
        trend_html = f'<span class="{trend_class}">{trend_icon} {delta_text}</span>'
    
    metric_cards.append(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        # f'<div class="metric-trend">{trend_html} <span style="opacity: 0.7">{context_text}</span></div>'
        f'</div>'
    )

hero_html = f"""
<div class="hero-container">
    <div class="hero-bg-pattern"></div>
    <div class="hero-content">
        <div>
            <h1 class="hero-title">AI-Powered Job Intelligence</h1>
            <p class="hero-description">
                Transform how you analyze job markets and match candidates. Our advanced AI platform uses cutting-edge natural language processing to uncover insights from millions of job postings.
            </p>
        </div>
        <div class="metrics-grid" style="display: flex; flex-direction: column; gap: 1rem; align-items: stretch;">
            {''.join(metric_cards)}
        </div>
    </div>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)

navigation_cards = [
    {
        "title": "Data Analysis",
        "description": "Visualize job market trends, company insights, and skill demand patterns across industries.",
        "button": "Explore Data",
        "page": "pages/1_EDA.py",
        "key": "nav_eda_btn",
    },
    {
        "title": "AI Analytics",
        "description": "Extract skills, topics, and insights from job descriptions using advanced machine learning.",
        "button": "Run Analytics",
        "page": "pages/3_NLP_Analytics.py",
        "key": "nav_analytics_btn",
    },
    {
        "title": "Resume Matching",
        "description": "Find your perfect job matches using AI-powered resume analysis and job compatibility scoring.",
        "button": "Match Resume",
        "page": "pages/7_Resume_Matching.py",
        "key": "nav_resume_btn",
    },
]

nav_columns = st.columns(len(navigation_cards))
for column, card in zip(nav_columns, navigation_cards):
    with column:
        st.markdown(
            f"""
            <div class="nav-card">
                <h4 class="nav-title">{card['title']}</h4>
                <p class="nav-desc">{card['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Button outside the card div to work with Streamlit's native button
        if st.button(card["button"], key=card["key"]):
            st.switch_page(card["page"])

