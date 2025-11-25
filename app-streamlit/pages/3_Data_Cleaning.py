import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
import unicodedata
import re
from datetime import datetime

from utils import initialize_workspace

# Initialize workspace path and imports
initialize_workspace()

# Page configuration
st.set_page_config(
    page_title="Data Cleaning - Job Analysis",
    page_icon="🧹",
    layout="wide"
)

# Load global CSS
try:
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.title("Data Cleaning")

# Initialize session state
if 'raw_jobs_df' not in st.session_state:
    st.session_state.raw_jobs_df = None
if 'cleaned_jobs_df' not in st.session_state:
    st.session_state.cleaned_jobs_df = None

def build_job_text(row):
    """Build combined job text from multiple columns"""
    parts = []

    if pd.notnull(row.get("Job Title")):
        parts.append("Job Title:\n" +str(row["Job Title"]))

    if pd.notnull(row.get("Responsibilities")):
        parts.append("Responsibilities:\n" +str(row["Responsibilities"]))

    if pd.notnull(row.get("Job Description")):
        parts.append("Job Description:\n" +str(row["Job Description"]))

    if pd.notnull(row.get("skills")):
        parts.append("Skills:\n" +str(row["skills"]))

    if pd.notnull(row.get("Experience")):
        parts.append("Experience:\n" + str(row["Experience"]))

    return "\n".join(parts)

def clean_job_text(text):
    """
    Cleans job posting text:
    - Removes emails, phone numbers, URLs
    - Removes HTML tags & entities
    - Normalizes bullets and whitespace
    - Collapses extra blank lines
    - Preserves actual content like skills, responsibilities, requirements
    """

    if not isinstance(text, str):
        return ""

    # 1) Unicode normalize + remove zero-width characters
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)

    # 2) Remove emails, phone numbers, URLs
    text = re.sub(r"\S+@\S+", " ", text)                          # emails
    text = re.sub(r"\+?\d[\d\-\s\(\)]{7,}\d", " ", text)          # phone numbers
    text = re.sub(r"(https?:\/\/\S+|www\.\S+)", " ", text)        # URLs

    # Remove names like: linkedin jobs, glassdoor jobs, etc.
    text = re.sub(r"(linkedin|glassdoor|indeed|monster|career|company)\S*",
                  " ", text, flags=re.IGNORECASE)

    # 3) Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)

    # 4) Normalize bullet points
    text = re.sub(r"[•●▪■◆▶►▸⦿⦾]", "- ", text)
    text = re.sub(r"^-(\S)", r"- \1", text, flags=re.MULTILINE)

    # 5) Normalize dashes
    text = text.replace("–", "-").replace("—", "-")

    # 6) Compact spaces
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)

    # 7) Collapse multiple blank lines (allow max 1)
    lines = [line.strip() for line in text.split("\n")]
    final_lines = []
    blank_seen = False

    for line in lines:
        if line == "":
            if not blank_seen:
                final_lines.append("")
            blank_seen = True
        else:
            final_lines.append(line)
            blank_seen = False

    text = "\n".join(final_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def strip_experience(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"Experience required:\s*\d+\s*(to\s*\d+)?\s*Years", "", text, flags=re.IGNORECASE).strip()

def load_raw_job_data():
    """Load raw job data from workspace"""
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        # Try different possible locations
        possible_paths = [
            os.path.join(workspace_path, "Data", "Jobs_data.csv"),
            os.path.join(workspace_path, "Data_Cleaning", "Jobs_data.csv"),
            os.path.join(workspace_path, "scraps", "combined_data.json"),
            os.path.join(workspace_path, "combined_data.json")
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    if path.endswith('.csv'):
                        df = pd.read_csv(path)
                    elif path.endswith('.json'):
                        with open(path, 'r') as f:
                            data = json.load(f)
                        df = pd.DataFrame(data)
                    st.success(f"✅ Loaded raw data from {path}")
                    return df
                except Exception as e:
                    st.warning(f"Failed to load {path}: {e}")
                    continue

    st.error("❌ Could not find raw job data files")
    return None

# Main interface
st.markdown("### Clean and Process Job Posting Data")
st.markdown("""
This page allows you to:
- Load raw job posting data
- Apply comprehensive text cleaning
- Remove duplicates
- Save cleaned data to JSON format
""")

# Load raw data
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Load Raw Job Data", type="primary"):
        with st.spinner("Loading raw job data..."):
            df = load_raw_job_data()
            if df is not None:
                st.session_state.raw_jobs_df = df
                st.session_state.cleaned_jobs_df = None  # Reset cleaned data
                st.rerun()

with col2:
    if st.session_state.raw_jobs_df is not None:
        if st.button("Clear Loaded Data"):
            st.session_state.raw_jobs_df = None
            st.session_state.cleaned_jobs_df = None
            st.rerun()

# Display raw data info
if st.session_state.raw_jobs_df is not None:
    df = st.session_state.raw_jobs_df
    st.success(f"✅ Raw data loaded: {len(df):,} job postings")

    # Data overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Jobs", f"{len(df):,}")
    with col2:
        duplicate_count = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicate_count)
    with col3:
        missing_text = df.get('Description', df.get('Job Description', pd.Series())).isnull().sum()
        st.metric("Missing Descriptions", missing_text)

    # Cleaning options
    st.markdown("### Cleaning Options")

    col1, col2 = st.columns(2)
    with col1:
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        combine_text = st.checkbox("Combine multiple text columns", value=True)
        strip_exp = st.checkbox("Strip experience requirements", value=False)

    with col2:
        clean_text = st.checkbox("Apply text cleaning", value=True)
        add_job_id = st.checkbox("Add job_id column", value=True)

    # Clean data button
    if st.button("Apply Cleaning", type="primary"):
        with st.spinner("Cleaning data..."):
            cleaned_df = df.copy()

            # Remove duplicates
            if remove_duplicates:
                initial_count = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                st.info(f"Removed {initial_count - len(cleaned_df)} duplicate rows")

            # Combine text columns
            if combine_text:
                cleaned_df['job_text_raw'] = cleaned_df.apply(build_job_text, axis=1)
                st.info("Combined text from multiple columns into 'job_text_raw'")

            # Apply text cleaning
            if clean_text:
                text_col = 'job_text_raw' if 'job_text_raw' in cleaned_df.columns else 'Description'
                if text_col in cleaned_df.columns:
                    cleaned_df['job_text_cleaned'] = cleaned_df[text_col].apply(clean_job_text)
                    st.info("Applied text cleaning to create 'job_text_cleaned'")
                else:
                    st.warning("No text column found for cleaning")

            # Add job ID
            if add_job_id:
                cleaned_df['job_id'] = range(len(cleaned_df))
                st.info("Added job_id column")

            # Strip experience and deduplicate
            if strip_exp:
                if 'job_text_cleaned' in cleaned_df.columns:
                    initial_count = len(cleaned_df)
                    cleaned_df['job_text_cleaned'] = cleaned_df['job_text_cleaned'].apply(strip_experience)
                    cleaned_df = cleaned_df.drop_duplicates(subset=['job_text_cleaned'], keep='first')
                    st.info(f"Stripped experience and deduplicated based on cleaned text: reduced from {initial_count} to {len(cleaned_df)} jobs")
                else:
                    st.warning("Cannot strip experience: 'job_text_cleaned' column not found. Enable 'Apply text cleaning' first.")

            st.session_state.cleaned_jobs_df = cleaned_df
            st.success(f"✅ Cleaning complete! Processed {len(cleaned_df)} jobs")
            st.rerun()

# Display cleaned data
if st.session_state.cleaned_jobs_df is not None:
    cleaned_df = st.session_state.cleaned_jobs_df

    st.markdown("### Cleaned Data Preview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cleaned Jobs", f"{len(cleaned_df):,}")
    with col2:
        if 'job_text_cleaned' in cleaned_df.columns:
            avg_length = cleaned_df['job_text_cleaned'].str.len().mean()
            st.metric("Avg Text Length", f"{avg_length:.0f} chars")
    with col3:
        columns_added = len(cleaned_df.columns) - len(st.session_state.raw_jobs_df.columns)
        st.metric("Columns Added", columns_added)

    # Show sample
    st.dataframe(cleaned_df.head(10), use_container_width=True)

    # Save options
    st.markdown("### Save Cleaned Data")

    col1, col2 = st.columns(2)
    with col1:
        filename = st.text_input("Filename", value="cleaned_data.json", help="Filename for saved data")

    with col2:
        save_format = st.selectbox("Format", ["JSON", "CSV"], index=0)

    if st.button("Save Cleaned Data", type="primary"):
        workspace_path = st.session_state.get('workspace_path')
        if workspace_path:
            if save_format == "JSON":
                save_path = os.path.join(workspace_path, "Data", filename)
                # Convert to records format for JSON
                json_data = cleaned_df.to_dict('records')
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)

                save_path_10 = os.path.join(workspace_path, "Data", "cleaned_data_sample.json")
                json_data_10 = cleaned_df.head(10).to_dict('records')
                with open(save_path_10, 'w', encoding='utf-8') as f:
                    json.dump(json_data_10, f, indent=2, ensure_ascii=False) 
            else:  # CSV
                save_path = os.path.join(workspace_path, "Data", filename)
                cleaned_df.to_csv(save_path, index=False)

            st.success(f"✅ Saved {len(cleaned_df)} cleaned jobs to {save_path}")

            # Also update combined_data.json if that's the target
            if filename == "combined_data.json":
                st.info("Updated combined_data.json - this will be used by other pages")
        else:
            st.error("Workspace path not found")

    # Populate Database with Embeddings
    st.markdown("### Populate Database with Embeddings")
    if st.button("🚀 Populate DB with Embeddings (50k batch)", type="primary", help="Compute embeddings for cleaned data and insert into database in batches of 50,000"):
        if st.session_state.cleaned_jobs_df is None:
            st.error("❌ No cleaned data available. Please clean data first.")
        else:
            with st.spinner("Computing embeddings and populating database... This may take several minutes."):
                try:
                    from functions.database import populate_job_embeddings_from_df
                    
                    success = populate_job_embeddings_from_df(st.session_state.cleaned_jobs_df)
                    if success:
                        st.success("✅ Database populated with embeddings from cleaned data!")
                    else:
                        st.error("❌ Failed to populate database. Check logs for details.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**Cleaning Process:**
1. **Load Raw Data**: Import job postings from CSV or JSON files
2. **Remove Duplicates**: Eliminate duplicate entries
3. **Combine Text**: Merge relevant columns into comprehensive job descriptions
4. **Text Cleaning**: Remove noise, normalize formatting, clean HTML/entities
5. **Save Results**: Export cleaned data in JSON format for use by other analysis pages
""")