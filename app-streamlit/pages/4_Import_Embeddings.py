import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime

from components.header import render_header
from utils import initialize_workspace

# Initialize workspace path and imports
initialize_workspace()

# Page configuration
st.set_page_config(
    page_title="Import Embeddings - Database",
    page_icon="📊",
    layout="wide"
)

# Load global CSS
try:
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.title("Import Embeddings to Database")

# Try to import NLP libraries
try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# Try to import SBERT utilities
try:
    from functions.nlp_models import load_sbert_model, compute_job_embeddings_sbert
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# Try to import Word2Vec utilities
try:
    from functions.nlp_models import load_trained_word2vec_model
    W2V_LOADER_AVAILABLE = True
except ImportError:
    W2V_LOADER_AVAILABLE = False

# Helper functions
def simple_tokenize(text):
    """Simple tokenization"""
    if pd.isna(text):
        return []
    return str(text).split()

def get_doc_embedding_w2v(tokens, model):
    """Get document embedding using Word2Vec"""
    if not GENSIM_AVAILABLE or model is None:
        # Get vector size from model if available, otherwise default to 300
        vector_size = getattr(model, 'vector_size', 300) if model else 300
        return np.zeros(vector_size, dtype="float32")
    
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        # Use model's vector_size, default to 300 for Word2Vec
        vector_size = getattr(model, 'vector_size', 300)
        return np.zeros(vector_size, dtype="float32")
    return np.mean(vectors, axis=0)

@st.cache_data
def compute_job_embeddings_w2v(job_texts, _model):
    """Compute embeddings for jobs using Word2Vec"""
    embeddings = []
    for text in job_texts:
        tokens = simple_tokenize(text)
        emb = get_doc_embedding_w2v(tokens, _model)
        embeddings.append(emb)
    return np.array(embeddings)

def load_job_data():
    """Load job data from workspace"""
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        # Try loading from cleaned_data.json first (default for NLP)
        json_path = os.path.join(workspace_path, "Data", "cleaned_data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            return df
        
        # Fallback to combined_data.json (has job links)
        json_path = os.path.join(workspace_path, "Data", "combined_data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            return df
        
        # Fallback to CSV
        csv_path = os.path.join(workspace_path, "Data_Cleaning", "cleaned_job_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
    
    # Try loading from current directory
    csv_path = "workspace/Data_Cleaning/cleaned_job_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    
    return None

# Initialize session state
if 'cleaned_jobs_df' not in st.session_state:
    st.session_state.cleaned_jobs_df = None

if 'w2v_model' not in st.session_state:
    st.session_state.w2v_model = None

if 'sbert_model' not in st.session_state:
    st.session_state.sbert_model = None

if 'auto_load_attempted' not in st.session_state:
    st.session_state.auto_load_attempted = False

if 'precomputed_w2v_embeddings' not in st.session_state:
    st.session_state.precomputed_w2v_embeddings = None

if 'precomputed_sbert_embeddings' not in st.session_state:
    st.session_state.precomputed_sbert_embeddings = None

if 'job_data_auto_load_attempted' not in st.session_state:
    st.session_state.job_data_auto_load_attempted = False

# Auto-load cleaned job data if not already loaded
if st.session_state.cleaned_jobs_df is None and not st.session_state.job_data_auto_load_attempted:
    with st.spinner("🔄 Auto-loading cleaned job data..."):
        df = load_job_data()
        if df is not None:
            st.session_state.cleaned_jobs_df = df
            st.success(f"✅ Auto-loaded {len(df):,} cleaned job postings")
            st.session_state.job_data_auto_load_attempted = True
            st.rerun()
        else:
            st.info("ℹ️ No cleaned job data found. Please clean job data in the Data Cleaning page first.")
            st.session_state.job_data_auto_load_attempted = True

# Functions to find and load pre-computed embeddings
def find_precomputed_embeddings():
    """Find available pre-computed embeddings in models directory"""
    workspace_path = st.session_state.get('workspace_path')
    if not workspace_path:
        return None, None
    
    models_dir = os.path.join(workspace_path, "models")
    if not os.path.exists(models_dir):
        return None, None
    
    w2v_embeddings = None
    sbert_embeddings = None
    w2v_metadata = None
    sbert_metadata = None
    
    try:
        import joblib
        all_files = os.listdir(models_dir)
        
        # Find Word2Vec embeddings - look for job_embeddings_w2v_*.npy
        w2v_files = [f for f in all_files if f.startswith('job_embeddings_w2v_') and f.endswith('.npy')]
        if w2v_files:
            # Get the most recent one (largest number of jobs)
            try:
                w2v_files_sorted = sorted(w2v_files, key=lambda x: int(x.split('_')[3].replace('jobs.npy', '')) if len(x.split('_')) > 3 else 0, reverse=True)
            except (ValueError, IndexError):
                # Fallback: just use the first file if sorting fails
                w2v_files_sorted = w2v_files
            
            w2v_file = w2v_files_sorted[0]
            w2v_path = os.path.join(models_dir, w2v_file)
            
            # Directly load the embeddings file
            try:
                w2v_embeddings = np.load(w2v_path, allow_pickle=False)
                # Verify it's a valid numpy array
                if not isinstance(w2v_embeddings, np.ndarray):
                    w2v_embeddings = None
            except Exception as e:
                # Show error in console for debugging
                print(f"Error loading Word2Vec embeddings from {w2v_path}: {e}")
                import traceback
                traceback.print_exc()
                w2v_embeddings = None

        
        # Find SBERT embeddings
        sbert_files = [f for f in all_files if f.startswith('job_embeddings_sbert_') and f.endswith('.npy')]
        if sbert_files:
            # Get the most recent one (largest number of jobs)
            try:
                sbert_files_sorted = sorted(sbert_files, key=lambda x: int(x.split('_')[3].replace('jobs.npy', '')) if len(x.split('_')) > 3 else 0, reverse=True)
            except (ValueError, IndexError):
                # Fallback: just use the first file if sorting fails
                sbert_files_sorted = sbert_files
            
            sbert_file = sbert_files_sorted[0]
            sbert_path = os.path.join(models_dir, sbert_file)
            
            try:
                sbert_embeddings = np.load(sbert_path)
            except Exception as e:
                # Log error but continue
                print(f"Error loading SBERT embeddings from {sbert_path}: {e}")
            
            # Try to load metadata
            if sbert_embeddings is not None:
                metadata_file = sbert_file.replace('.npy', '_metadata.joblib')
                metadata_path = os.path.join(models_dir, metadata_file)
                if os.path.exists(metadata_path):
                    try:
                        sbert_metadata = joblib.load(metadata_path)
                    except Exception:
                        pass
    except Exception as e:
        # Log the error for debugging
        print(f"Error in find_precomputed_embeddings: {e}")
        import traceback
        traceback.print_exc()
    
    return (w2v_embeddings, w2v_metadata), (sbert_embeddings, sbert_metadata)

# Auto-load functions
def auto_load_word2vec_model(use_pretrained=False):
    """Automatically load Word2Vec model if available"""
    if st.session_state.w2v_model is not None:
        return True  # Already loaded
    
    if not W2V_LOADER_AVAILABLE:
        return False
    
    try:
        w2v_model = load_trained_word2vec_model(use_pretrained=use_pretrained)
        if w2v_model is not None:
            st.session_state.w2v_model = w2v_model
            return True
        return False
    except Exception:
        return False

def auto_load_sbert_model():
    """Automatically load SBERT model if available"""
    if st.session_state.sbert_model is not None:
        return True  # Already loaded
    
    if not SBERT_AVAILABLE:
        return False
    
    try:
        sbert_model = load_sbert_model()
        if sbert_model is not None:
            st.session_state.sbert_model = sbert_model
            return True
        return False
    except Exception:
        return False

# Direct load of pre-computed embeddings if not already loaded
workspace_path = st.session_state.get('workspace_path')
if workspace_path:
    models_dir = os.path.join(workspace_path, "models")
    if os.path.exists(models_dir):
        try:
            all_files = os.listdir(models_dir)
            
            # Load Word2Vec embeddings
            if st.session_state.get('precomputed_w2v_embeddings') is None:
                w2v_files = [f for f in all_files if f.startswith('job_embeddings_w2v_') and f.endswith('.npy')]
                if w2v_files:
                    try:
                        w2v_files_sorted = sorted(w2v_files, key=lambda x: int(x.split('_')[3].replace('jobs.npy', '')) if len(x.split('_')) > 3 else 0, reverse=True)
                    except (ValueError, IndexError):
                        w2v_files_sorted = w2v_files
                    
                    w2v_file = w2v_files_sorted[0]
                    w2v_path = os.path.join(models_dir, w2v_file)
                    embeddings = np.load(w2v_path, allow_pickle=False)
                    if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
                        st.session_state.precomputed_w2v_embeddings = embeddings
            
            # Load SBERT embeddings
            if st.session_state.get('precomputed_sbert_embeddings') is None:
                sbert_files = [f for f in all_files if f.startswith('job_embeddings_sbert_') and f.endswith('.npy')]
                if sbert_files:
                    try:
                        sbert_files_sorted = sorted(sbert_files, key=lambda x: int(x.split('_')[3].replace('jobs.npy', '')) if len(x.split('_')) > 3 else 0, reverse=True)
                    except (ValueError, IndexError):
                        sbert_files_sorted = sbert_files
                    
                    sbert_file = sbert_files_sorted[0]
                    sbert_path = os.path.join(models_dir, sbert_file)
                    embeddings = np.load(sbert_path, allow_pickle=False)
                    if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
                        st.session_state.precomputed_sbert_embeddings = embeddings
        except Exception as e:
            # Store error for debugging
            st.session_state.embedding_load_error = str(e)

# Auto-load models and pre-computed embeddings on first page load
if not st.session_state.auto_load_attempted:
    with st.spinner("🔄 Auto-loading models and checking for pre-computed embeddings..."):
        w2v_loaded = auto_load_word2vec_model()
        sbert_loaded = auto_load_sbert_model()
        
        # Check for pre-computed embeddings
        workspace_path = st.session_state.get('workspace_path')
        models_dir = os.path.join(workspace_path, "models") if workspace_path else None
        
        # Debug: Show what we're looking for
        if models_dir and os.path.exists(models_dir):
            all_files = os.listdir(models_dir)
            w2v_embedding_files = [f for f in all_files if f.startswith('job_embeddings_w2v_') and f.endswith('.npy')]
            if w2v_embedding_files:
                st.session_state.debug_w2v_files_found = w2v_embedding_files
        
        (w2v_emb, w2v_meta), (sbert_emb, sbert_meta) = find_precomputed_embeddings()
        
        # Store embeddings in session state
        if w2v_emb is not None:
            st.session_state.precomputed_w2v_embeddings = w2v_emb
            st.session_state.precomputed_w2v_metadata = w2v_meta
        if sbert_emb is not None:
            st.session_state.precomputed_sbert_embeddings = sbert_emb
            st.session_state.precomputed_sbert_metadata = sbert_meta
        
        st.session_state.auto_load_attempted = True
        
        # Show auto-load results
        results = []
        warnings = []
        
        if w2v_loaded:
            results.append("✅ Word2Vec model")
        else:
            # Check if models directory exists and has files
            workspace_path = st.session_state.get('workspace_path')
            if workspace_path:
                models_dir = os.path.join(workspace_path, "models")
                if os.path.exists(models_dir):
                    all_files = os.listdir(models_dir)
                    joblib_files = [f for f in all_files if f.endswith('.joblib')]
                    if joblib_files:
                        w2v_files = [f for f in joblib_files if 'word2vec' in f.lower() or 'w2v' in f.lower()]
                        if not w2v_files:
                            w2v_files = joblib_files
                        if w2v_files:
                            warnings.append(f"⚠️ Found model file(s) but couldn't auto-load. Try manual load below.")
                    else:
                        warnings.append(f"ℹ️ No Word2Vec model found in {models_dir}. Train one in NLP Analytics page.")
                else:
                    warnings.append(f"ℹ️ Models directory not found. Train a Word2Vec model in NLP Analytics page first.")
            else:
                warnings.append(f"ℹ️ Workspace path not set. Train a Word2Vec model in NLP Analytics page first.")
        
        if sbert_loaded:
            results.append("✅ SBERT model")
        
        if w2v_emb is not None:
            num_jobs = w2v_emb.shape[0] if hasattr(w2v_emb, 'shape') else 'unknown'
            results.append(f"✅ Pre-computed Word2Vec embeddings ({num_jobs} jobs)")
        elif st.session_state.auto_load_attempted:
            # Check if embedding files exist but couldn't be loaded
            workspace_path = st.session_state.get('workspace_path')
            if workspace_path:
                models_dir = os.path.join(workspace_path, "models")
                if os.path.exists(models_dir):
                    all_files = os.listdir(models_dir)
                    w2v_embedding_files = [f for f in all_files if f.startswith('job_embeddings_w2v_') and f.endswith('.npy')]
                    if w2v_embedding_files:
                        warnings.append(f"⚠️ Found Word2Vec embedding file(s): {', '.join(w2v_embedding_files)} but couldn't load them. Check file permissions or file format.")
        
        if sbert_emb is not None:
            num_jobs = sbert_emb.shape[0] if hasattr(sbert_emb, 'shape') else 'unknown'
            results.append(f"✅ Pre-computed SBERT embeddings ({num_jobs} jobs)")
        
        if results:
            st.success(f"Auto-loaded: {', '.join(results)}")
        if warnings:
            for warning in warnings:
                st.info(warning)
        
        # Debug info: Show what was checked
        with st.expander("🔍 Debug: Auto-load Details", expanded=False):
            workspace_path = st.session_state.get('workspace_path')
            st.write(f"**Workspace Path**: {workspace_path if workspace_path else 'Not set'}")
            if workspace_path:
                models_dir = os.path.join(workspace_path, "models")
                st.write(f"**Models Directory**: {models_dir}")
                st.write(f"**Directory Exists**: {os.path.exists(models_dir) if models_dir else False}")
                if os.path.exists(models_dir):
                    all_files = os.listdir(models_dir)
                    w2v_embedding_files = [f for f in all_files if f.startswith('job_embeddings_w2v_') and f.endswith('.npy')]
                    sbert_embedding_files = [f for f in all_files if f.startswith('job_embeddings_sbert_') and f.endswith('.npy')]
                    st.write(f"**Word2Vec embedding files found**: {w2v_embedding_files if w2v_embedding_files else 'None'}")
                    st.write(f"**SBERT embedding files found**: {sbert_embedding_files if sbert_embedding_files else 'None'}")
                    st.write(f"**Word2Vec embeddings loaded**: {w2v_emb is not None}")
                    st.write(f"**SBERT embeddings loaded**: {sbert_emb is not None}")
                    if w2v_emb is not None:
                        st.write(f"**Word2Vec embeddings shape**: {w2v_emb.shape if hasattr(w2v_emb, 'shape') else 'N/A'}")
                    if sbert_emb is not None:
                        st.write(f"**SBERT embeddings shape**: {sbert_emb.shape if hasattr(sbert_emb, 'shape') else 'N/A'}")
        
        # Only rerun if we actually loaded something
        if w2v_loaded or sbert_loaded or w2v_emb is not None or sbert_emb is not None:
            st.rerun()

# Main content
st.markdown("""
### Overview
Import computed embeddings to PostgreSQL database with pgvector extension:
- **Word2Vec embeddings**  - Word2Vec model 
- **SBERT embeddings**  - SBERT model 
- **Pre-computed embeddings** - Automatically detects and uses pre-computed embeddings from `workspace/models/` if available (much faster!)
- Stores embeddings in `embedding` (SBERT, 384 dimensions) and `word2vec_embedding` (Word2Vec, 300 dimensions)
- Enables fast vector similarity search using pgvector indexes

**Note**: You only need Word2Vec to import embeddings. SBERT is optional and will be added automatically if available. Pre-computed embeddings (e.g., `job_embeddings_w2v_*.npy`, `job_embeddings_sbert_*.npy`) will be automatically detected and used if the number of jobs matches.
""")

# Check database availability
try:
    from functions.database import (
        create_db_engine, setup_jobs_table,
        insert_job_with_multiple_embeddings,
        batch_insert_jobs_with_embeddings,
        drop_jobs_table,
        reset_jobs_table,
        backup_jobs_to_sql,
        restore_jobs_from_latest_sql_backup,
        execute_query,
    )
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    st.error(f"Database functions not available: {e}")
    st.info("Please ensure database connection is configured in `.env` or `docker-compose.yml`")

if not DB_AVAILABLE:
    st.stop()

# Database Setup Section
st.markdown("---")
st.markdown("### Database Setup")

col_setup, col_backup, col_reset = st.columns(3)

with col_setup:
    if st.button("Setup / Ensure Jobs Table", key="setup_db_table", type="primary"):
        with st.spinner("Setting up database table..."):
            if setup_jobs_table():
                # After ensuring schema, try to restore from latest SQL backup if available
                restored = restore_jobs_from_latest_sql_backup()
                if restored:
                    st.success("✅ Database table created/updated and restored from latest SQL backup!")
                else:
                    st.success("✅ Database table created/updated! (No SQL backup restored; either none found or restore failed — check logs/backups.)")
            else:
                st.error("❌ Failed to setup database table. Check database connection.")

with col_backup:

    if st.button("Backup Jobs to SQL", key="backup_jobs_sql"):
        with st.spinner("Exporting jobs table as SQL..."):
            backup_sql_path = backup_jobs_to_sql()
            if backup_sql_path:
                st.success(f"✅ Jobs exported as SQL to `{backup_sql_path}`")
                st.info("You can restore data by running this SQL file against a database where the `jobs` table already exists.")
            else:
                st.error("❌ Failed to export jobs to SQL. Check logs and permissions.")

with col_reset:
    if st.button("Delete & Recreate Jobs Table", key="reset_db_table"):
        with st.spinner("Dropping and recreating jobs table..."):
            if reset_jobs_table():
                st.success("✅ Jobs table dropped and recreated successfully!")
            else:
                st.error("❌ Failed to reset jobs table. Check database connection and permissions.")

# Database Sample Data Section
st.markdown("---")
st.markdown("### Sample Data from Database")

col_view, col_refresh = st.columns([3, 1])
with col_view:
    st.markdown("View sample records from the `jobs` table in the database.")
with col_refresh:
    if st.button("🔄 Refresh Data", key="refresh_db_sample"):
        st.rerun()

try:
    # Query sample data from database with embedding values as strings
    query = """
    SELECT 
        id,
        title,
        company,
        LEFT(text, 200) as text_preview,
        CASE 
            WHEN embedding IS NOT NULL THEN embedding::text
            ELSE NULL 
        END as sbert_embedding_str,
        CASE 
            WHEN word2vec_embedding IS NOT NULL THEN word2vec_embedding::text
            ELSE NULL 
        END as w2v_embedding_str,
        CASE 
            WHEN embedding IS NOT NULL THEN 'Yes' 
            ELSE 'No' 
        END as has_sbert_embedding,
        CASE 
            WHEN word2vec_embedding IS NOT NULL THEN 'Yes' 
            ELSE 'No' 
        END as has_w2v_embedding,
        created_at
    FROM jobs
    ORDER BY created_at DESC NULLS LAST
    LIMIT 10
    """
    
    df_sample = execute_query(query, use_host=False)
    
    if df_sample is not None and len(df_sample) > 0:
        st.success(f"✅ Found {len(df_sample)} sample records (showing latest 10)")
        
        # Get total count
        count_query = "SELECT COUNT(*) as total FROM jobs"
        count_df = execute_query(count_query, use_host=False)
        total_count = count_df.iloc[0]['total'] if count_df is not None and len(count_df) > 0 else 0
        
        st.info(f"📊 Total jobs in database: **{total_count:,}**")
        
        # Create preview columns for embeddings (first 100 chars)
        df_display = df_sample.copy()
        if 'sbert_embedding_str' in df_display.columns:
            df_display['sbert_preview'] = df_display['sbert_embedding_str'].apply(
                lambda x: (x[:100] + '...') if x and len(str(x)) > 100 else (x if x else 'N/A')
            )
        if 'w2v_embedding_str' in df_display.columns:
            df_display['w2v_preview'] = df_display['w2v_embedding_str'].apply(
                lambda x: (x[:100] + '...') if x and len(str(x)) > 100 else (x if x else 'N/A')
            )
        
        # Display the sample data
        display_cols = ['id', 'title', 'company', 'text_preview', 'has_sbert_embedding', 'has_w2v_embedding', 'created_at']
        if 'sbert_preview' in df_display.columns:
            display_cols.insert(display_cols.index('has_sbert_embedding'), 'sbert_preview')
        if 'w2v_preview' in df_display.columns:
            display_cols.insert(display_cols.index('has_w2v_embedding'), 'w2v_preview')
        
        st.dataframe(
            df_display[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.TextColumn("Job ID", width="small"),
                "title": st.column_config.TextColumn("Title", width="medium"),
                "company": st.column_config.TextColumn("Company", width="medium"),
                "text_preview": st.column_config.TextColumn("Text", width="large"),
                "sbert_preview": st.column_config.TextColumn("SBERT Embedding", width="medium"),
                # "has_sbert_embedding": st.column_config.TextColumn("Has SBERT", width="small"),
                "w2v_preview": st.column_config.TextColumn("Word2Vec Embedding", width="medium"),
                # "has_w2v_embedding": st.column_config.TextColumn("Has W2V", width="small"),
                "created_at": st.column_config.DatetimeColumn("Created At", width="small"),
            }
        )
        
except Exception as e:
    st.error(f"❌ Error querying database: {str(e)}")
    st.info("💡 Make sure the database is running and the `jobs` table exists. Click 'Setup / Ensure Jobs Table' above if needed.")

# Import Embeddings Section
st.markdown("---")
st.markdown("### Import Embeddings to Database")

# Show requirements info
precomputed_w2v_available = st.session_state.get('precomputed_w2v_embeddings') is not None
precomputed_sbert_available = st.session_state.get('precomputed_sbert_embeddings') is not None

if st.session_state.cleaned_jobs_df is None:
    st.warning("⚠️ Please load job data first.")
    col_load1, col_load2 = st.columns([1, 3])
    with col_load1:
        if st.button("🔄 Load Cleaned Job Data", type="primary", key="manual_load_jobs"):
            with st.spinner("Loading cleaned job data..."):
                df = load_job_data()
                if df is not None:
                    st.session_state.cleaned_jobs_df = df
                    st.success(f"✅ Loaded {len(df):,} cleaned job postings")
                    st.rerun()
                else:
                    st.error("❌ Could not load cleaned job data. Please ensure cleaned data exists in workspace/Data/cleaned_data.json, workspace/Data/combined_data.json, or workspace/Data_Cleaning/cleaned_job_data.csv")
    with col_load2:
        st.info("💡 Tip: Clean job data in the **Data Cleaning** page first, then return here to import embeddings.")
elif st.session_state.w2v_model is None and st.session_state.get('precomputed_w2v_embeddings') is None:
    # Only show warning if auto-load was attempted (to avoid showing it during auto-load)
    # Check if we have pre-computed embeddings as an alternative
    if st.session_state.auto_load_attempted:
        st.warning("⚠️ Word2Vec model not loaded and no pre-computed embeddings found. Please load a model manually below, train one in the NLP Analytics page, or ensure pre-computed embeddings are available.")
else:
    df = st.session_state.cleaned_jobs_df
        # Import options
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.number_input(
            "Batch Size for Import",
            min_value=10,
            max_value=10000,
            value=1000,
            step=10,
            help="Number of jobs to process in each batch"
        )
    
    with col2:
        # Get available text columns
        text_columns = [col for col in df.columns if 'text' in col.lower() or 'description' in col.lower() or 'job' in col.lower()]
        if not text_columns:
            text_columns = df.columns.tolist()
        
        text_column = st.selectbox(
            "Text Column",
            options=text_columns,
            index=5 if 'job_text_cleaned' in text_columns else 0,
            help="Column to use for computing embeddings"
        )
    
    # Check for pre-computed embeddings
    precomputed_w2v = st.session_state.get('precomputed_w2v_embeddings')
    precomputed_sbert = st.session_state.get('precomputed_sbert_embeddings')
    
    # Show pre-computed embeddings info if available
    if precomputed_w2v is not None or precomputed_sbert is not None:
        st.markdown("#### Pre-computed Embeddings Available")
        col1, col2 = st.columns(2)
        with col1:
            if precomputed_w2v is not None:
                w2v_shape = precomputed_w2v.shape if hasattr(precomputed_w2v, 'shape') else None
                num_w2v = w2v_shape[0] if w2v_shape else 'unknown'
                st.success(f"✅ Word2Vec embeddings: {num_w2v} jobs")
            else:
                st.info("ℹ️ No pre-computed Word2Vec embeddings")
        with col2:
            if precomputed_sbert is not None:
                sbert_shape = precomputed_sbert.shape if hasattr(precomputed_sbert, 'shape') else None
                num_sbert = sbert_shape[0] if sbert_shape else 'unknown'
                st.success(f"✅ SBERT embeddings: {num_sbert} jobs")
            else:
                st.info("ℹ️ No pre-computed SBERT embeddings")
        
        use_precomputed = st.checkbox(
            "Use pre-computed embeddings if available (faster)",
            value=True,
            help="If the number of jobs matches, use pre-computed embeddings instead of recomputing"
        )
    else:
        use_precomputed = False
    
    # Button text based on SBERT availability
    button_text = "Import Embeddings to Database"
    if SBERT_AVAILABLE:
        button_text = "Import Embeddings (Word2Vec + SBERT)"
    else:
        button_text = "Import Embeddings (Word2Vec only)"
    
    if st.button(button_text, type="primary", key="import_w2v_to_db"):
        # Get text column
        if text_column not in df.columns:
            st.error(f"Column '{text_column}' not found in dataframe. Available columns: {', '.join(df.columns)}")
        else:
            job_texts = df[text_column].dropna().tolist()
            num_jobs = len(job_texts)
            
            if num_jobs == 0:
                st.error("No job texts found in the selected column")
            else:
                progress_container = st.container()
                progress_bar = progress_container.progress(0)
                status_text = progress_container.empty()
                
                # Use pre-computed Word2Vec embeddings if available and matching
                w2v_embeddings = None
                if use_precomputed and precomputed_w2v is not None:
                    w2v_shape = precomputed_w2v.shape if hasattr(precomputed_w2v, 'shape') else None
                    if w2v_shape and w2v_shape[0] == num_jobs:
                        status_text.text(f"Using pre-computed Word2Vec embeddings for {num_jobs} jobs...")
                        progress_bar.progress(0.1)
                        w2v_embeddings = precomputed_w2v
                        st.info("✅ Using pre-computed Word2Vec embeddings (no computation needed)")
                    else:
                        st.warning(f"⚠️ Pre-computed Word2Vec embeddings have {w2v_shape[0] if w2v_shape else 'unknown'} jobs, but you have {num_jobs} jobs. Will recompute.")
                
                # Compute Word2Vec embeddings if not using pre-computed
                if w2v_embeddings is None:
                    # Check if we have a model to compute with
                    if st.session_state.w2v_model is None:
                        st.error("❌ Cannot compute Word2Vec embeddings: No model loaded and pre-computed embeddings don't match. Please load a Word2Vec model or ensure pre-computed embeddings match the number of jobs.")
                        st.stop()
                    
                    status_text.text(f"Computing Word2Vec embeddings for {num_jobs} jobs...")
                    progress_bar.progress(0.1)
                    w2v_embeddings = compute_job_embeddings_w2v(job_texts, st.session_state.w2v_model)
                
                if w2v_embeddings is None or len(w2v_embeddings) == 0:
                    st.error("Failed to compute Word2Vec embeddings")
                    st.stop()

                # Check Word2Vec embedding dimension
                w2v_dim = len(w2v_embeddings[0]) if len(w2v_embeddings) > 0 else 0
                if w2v_dim != 300:
                    st.warning(f"⚠️ **Dimension Mismatch**: Word2Vec embeddings have {w2v_dim} dimensions, but database expects 300.")
                    st.info("The import may fail. Consider updating the database schema or retraining the model.")
                    
                    if not st.checkbox("Continue anyway (may fail)", key="continue_dim_mismatch"):
                        st.stop()

                # Optionally compute SBERT embeddings
                sbert_embeddings = None
                if SBERT_AVAILABLE:
                    # Use pre-computed SBERT embeddings if available and matching (no model needed!)
                    if use_precomputed and precomputed_sbert is not None:
                        sbert_shape = precomputed_sbert.shape if hasattr(precomputed_sbert, 'shape') else None
                        if sbert_shape and sbert_shape[0] == num_jobs:
                            status_text.text(f"Using pre-computed SBERT embeddings for {num_jobs} jobs...")
                            progress_bar.progress(0.2)
                            sbert_embeddings = precomputed_sbert
                            st.info("✅ Using pre-computed SBERT embeddings (no model or computation needed)")
                        else:
                            st.warning(f"⚠️ Pre-computed SBERT embeddings have {sbert_shape[0] if sbert_shape else 'unknown'} jobs, but you have {num_jobs} jobs. Will recompute using model.")
                    
                    # Compute SBERT embeddings if not using pre-computed (model required)
                    if sbert_embeddings is None:
                        status_text.text(f"Computing SBERT embeddings for {num_jobs} jobs...")
                        progress_bar.progress(0.2)
                        try:
                            # Use pre-loaded SBERT model from session state if available
                            sbert_model = st.session_state.get('sbert_model')
                            if sbert_model is None:
                                # Fallback: try to load it now
                                sbert_model = load_sbert_model()
                                if sbert_model:
                                    st.session_state.sbert_model = sbert_model
                            
                            if sbert_model:
                                sbert_embeddings = compute_job_embeddings_sbert(job_texts, sbert_model)
                            else:
                                st.warning("SBERT model could not be loaded; proceeding with Word2Vec only.")
                        except Exception as e:
                            st.warning(f"SBERT embedding computation failed: {e}. Proceeding with Word2Vec only.")
                else:
                    st.info("SBERT utilities not available. Only Word2Vec embeddings will be imported.")

                # If SBERT embeddings exist, validate dimension
                if sbert_embeddings is not None and len(sbert_embeddings) > 0:
                    sbert_dim = len(sbert_embeddings[0])
                    if sbert_dim != 384:
                        st.warning(f"⚠️ **Dimension Mismatch**: SBERT embeddings have {sbert_dim} dimensions, but database expects 384.")
                        if not st.checkbox("Continue anyway with SBERT (may fail)", key="continue_sbert_dim_mismatch"):
                            st.stop()
                
                progress_bar.progress(0.3)
                status_text.text(f"Preparing data for database import...")
                
                # Prepare batch data
                batch_data = []
                valid_indices = df[text_column].notna()
                valid_df = df[valid_indices].reset_index(drop=True)
                
                for idx, (_, row) in enumerate(valid_df.iterrows()):
                    if idx < len(w2v_embeddings):
                        job_id = str(row.get('id', row.get('Job Id', row.get('job_id', idx))))
                        job_text = row.get(text_column, '')
                        company = row.get('Company', row.get('company', None))
                        title = row.get('Job Title', row.get('job_title', row.get('title', None)))
                        
                        # Convert numpy arrays to lists
                        w2v_embedding = w2v_embeddings[idx].tolist() if hasattr(w2v_embeddings[idx], 'tolist') else list(w2v_embeddings[idx])
                        
                        job_record = {
                            'id': job_id,
                            'title': title,
                            'company': company,
                            'text': job_text,
                            'word2vec_embedding': w2v_embedding
                        }

                        if sbert_embeddings is not None and idx < len(sbert_embeddings):
                            sbert_embedding = sbert_embeddings[idx].tolist() if hasattr(sbert_embeddings[idx], 'tolist') else list(sbert_embeddings[idx])
                            job_record['embedding'] = sbert_embedding

                        batch_data.append(job_record)
                
                progress_bar.progress(0.5)
                status_text.text(f"Importing {len(batch_data)} jobs to database in batches of {batch_size}...")
                
                # Import in batches
                success_count = 0
                total_batches = (len(batch_data) + batch_size - 1) // batch_size
                
                for batch_idx in range(0, len(batch_data), batch_size):
                    batch = batch_data[batch_idx:batch_idx + batch_size]
                    current_batch = (batch_idx // batch_size) + 1
                    
                    progress = 0.5 + (current_batch / total_batches) * 0.5
                    progress_bar.progress(progress)
                    status_text.text(f"Importing batch {current_batch}/{total_batches} ({len(batch)} jobs)...")
                    
                    if batch_insert_jobs_with_embeddings(batch):
                        success_count += len(batch)
                    else:
                        st.warning(f"Failed to import batch {current_batch}")
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Successfully imported {success_count}/{len(batch_data)} jobs with embeddings!")
                
                # Determine what was imported
                if sbert_embeddings is not None and len(sbert_embeddings) > 0:
                    embeddings_info = "Word2Vec + SBERT"
                else:
                    embeddings_info = "Word2Vec"
                
                st.success(f"✅ Import complete! {success_count} jobs with {embeddings_info} embeddings are now in the database.")
                
                # Show summary
                with st.expander("📊 Import Summary", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Jobs Processed", f"{num_jobs:,}")
                    with col2:
                        st.metric("Successfully Imported", f"{success_count:,}")
                    with col3:
                        w2v_dim_display = len(w2v_embeddings[0]) if len(w2v_embeddings) > 0 else 'N/A'
                        st.metric("Word2Vec Embedding Dim", w2v_dim_display)
                        if w2v_dim_display != 300 and w2v_dim_display != 'N/A':
                            st.caption(f"⚠️ Expected 300, got {w2v_dim_display}")
                    
                    st.markdown("**Details:**")
                    if sbert_embeddings is not None and len(sbert_embeddings) > 0:
                        st.write(f"- **Embeddings imported**: Word2Vec (300-dim) + SBERT (384-dim)")
                        st.write(f"- **Database columns**: `embedding` (SBERT), `word2vec_embedding` (Word2Vec)")
                    else:
                        st.write(f"- **Embeddings imported**: Word2Vec (300-dim) only")
                        st.write(f"- **Database columns**: `word2vec_embedding` (Word2Vec)")
                    st.write(f"- **Vector search**: Enabled with pgvector index")
                    st.write(f"- **Batch size used**: {batch_size}")
                    st.write(f"- **Text column**: `{text_column}`")

