import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import unicodedata
from datetime import datetime

# Add joblib import for model saving
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from utils import initialize_workspace

# Initialize workspace path and imports
initialize_workspace()

# Try to import NLP libraries
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Master skill list from NER notebook
MASTER_SKILL_LIST = [
    # Technical / Data Skills
    "python", "r", "java", "javascript", "typescript",
    "c++", "c#", "scala", "go", "matlab",
    "bash", "shell scripting",
    "sql", "nosql", "postgresql", "mysql", "oracle", "sqlite",
    "mongodb", "snowflake", "redshift", "bigquery", "azure sql",
    "data analysis", "data analytics", "statistical analysis",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "plotly", "pyspark", "spark", "hadoop", "hive", "mapreduce",
    "machine learning", "deep learning", "neural networks",
    "logistic regression", "linear regression", "random forest",
    "xgboost", "lightgbm", "catboost",
    "svm", "knn", "decision trees", "pca", "kmeans",
    "gradient boosting", "model tuning", "feature engineering",
    "nlp", "natural language processing", "topic modeling",
    "lda", "lsa", "keyword extraction",
    "named entity recognition", "text classification",
    "sentiment analysis", "embeddings", "bert", "word2vec",
    "aws", "azure", "gcp", "docker", "kubernetes",
    "lambda", "ec2", "s3", "athena", "dynamodb",
    "databricks", "airflow", "cloud functions",
    "tableau", "power bi", "metabase", "looker", "qlik",
    "data visualization", "dashboard development",
    "etl", "elt", "data pipeline", "data ingestion",
    "data cleaning", "data transformation", "data integration",
    "git", "github", "gitlab", "bitbucket",
    "ci/cd", "jenkins",
    "sap", "sap erp", "salesforce", "salesforce crm",
    "hubspot", "hubspot crm", "airtable", "jira", "confluence", "notion",
    # Business & Analytics Skills
    "business analysis", "requirements gathering",
    "market research", "competitive analysis",
    "financial analysis", "risk analysis", "cost analysis",
    "forecasting", "trend analysis", "variance analysis",
    "p&l management", "strategic planning",
    "business modeling", "stakeholder management",
    "reporting", "presentation development",
    "process improvement", "process optimization",
    "root cause analysis", "gap analysis",
    "workflow automation", "operational efficiency",
    "kpi analysis", "performance analysis",
    "customer segmentation", "persona development",
    "data-driven decision making",
    "problem solving", "insights synthesis",
    "client communication", "proposal writing",
    "project scoping", "roadmap planning",
    "change management", "cross-functional collaboration",
    # Marketing / Sales / RevOps Skills
    "crm management", "lead generation", "pipeline management",
    "sales operations", "sales strategy", "sales forecasting",
    "revenue operations", "revops", "gtm strategy",
    "go-to-market", "account management",
    "client success", "customer retention",
    "digital marketing", "content marketing",
    "seo", "sem", "ppc", "email marketing",
    "campaign optimization", "social media analytics",
    "marketing automation", "google analytics",
    "google ads", "mailchimp", "marketo",
    "outreach", "gong", "zoominfo",
    "validation rules", "crm integrations",
    "funnel analysis", "data stamping",
    # Product Skills
    "product management", "product analytics",
    "a/b testing", "experiment design",
    "feature prioritization", "user research", "ux research",
    "user stories", "agile", "scrum", "kanban",
    "roadmap development", "user journey mapping",
    "requirements documentation",
    "market sizing", "competitive positioning",
    # Finance & Operations Skills
    "fp&a", "financial modeling", "budgeting",
    "scenario analysis", "invoice processing",
    "billing operations", "revenue analysis",
    "cost optimization",
    "supply chain management", "inventory management",
    "logistics", "procurement", "vendor management",
    "operations management", "kpi reporting",
    # Soft Skills
    "communication", "leadership", "teamwork",
    "collaboration", "critical thinking", "problem solving",
    "adaptability", "time management",
    "presentation skills", "negotiation",
    "public speaking", "project management",
    "detail oriented", "strategic thinking",
    "multitasking", "analytical thinking",
    "decision making", "organization skills"
]

MASTER_SKILL_LIST = list(set(MASTER_SKILL_LIST))

# NER functions from notebook
@st.cache_resource
def load_spacy_model():
    """Load spaCy model"""
    if SPACY_AVAILABLE:
        try:
            return spacy.load("en_core_web_sm")
        except:
            return None
    return None

@st.cache_resource
def build_skill_ner(skill_list):
    """Builds a spaCy PhraseMatcher for custom skill extraction."""
    if not SPACY_AVAILABLE:
        return None
    nlp = load_spacy_model()
    if nlp is None:
        return None
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SKILL", patterns)
    return matcher

def extract_skill_entities(text, skill_matcher):
    """Extracts skill entities from text using the SKILL PhraseMatcher."""
    if not SPACY_AVAILABLE or skill_matcher is None:
        return []
    nlp = load_spacy_model()
    if nlp is None:
        return []
    doc = nlp(text)
    matches = skill_matcher(doc)
    skills_found = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        skills_found.add(span.text.lower())
    return sorted(list(skills_found))

def extract_spacy_entities(text):
    """Extract spaCy NER entities"""
    if not SPACY_AVAILABLE:
        return []
    nlp = load_spacy_model()
    if nlp is None:
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def strip_experience(text):
    """Strip experience requirements from text"""
    if not isinstance(text, str):
        return ""
    import re
    return re.sub(r"Experience required:\s*\d+\s*(to\s*\d+)?\s*Years", "", text, flags=re.IGNORECASE).strip()

@st.cache_data
def run_topic_modeling(texts, method='LDA', n_topics=10, n_words=10, save_model=True):
    """Run topic modeling on texts"""
    if not SKLEARN_AVAILABLE:
        return None
    
    # Try to import cuML for GPU acceleration
    try:
        from cuml.decomposition import LatentDirichletAllocation as cuLDA
        from cuml.decomposition import TruncatedSVD as cuSVD
        CUM_AVAILABLE = True
    except ImportError:
        CUM_AVAILABLE = False
    
    # Create models directory if it doesn't exist
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        models_dir = os.path.join(workspace_path, "models")
        os.makedirs(models_dir, exist_ok=True)
    else:
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
    
    model_filename = f"topic_model_{method.lower()}_{n_topics}topics.joblib"
    model_path = os.path.join(models_dir, model_filename)
    
    # Check if model already exists
    if os.path.exists(model_path) and JOBLIB_AVAILABLE:
        st.info(f"Loading existing {method} model from {model_path}")
        try:
            saved_data = joblib.load(model_path)
            # Return the saved results
            return saved_data['results']
        except Exception as e:
            st.warning(f"Could not load saved model: {e}. Retraining...")
    
    # Preprocess texts
    processed_texts = [strip_experience(text) for text in texts if isinstance(text, str)]
    
    if method == 'LDA':
        # LDA configuration
        vectorizer = CountVectorizer(
            strip_accents="unicode",
            stop_words="english",
            lowercase=True,
            max_features=5000,
            token_pattern=r"\b[a-zA-Z]{3,}\b",
            max_df=0.75,
            min_df=5,
            ngram_range=(1, 3),
        )
        dtm = vectorizer.fit_transform(processed_texts)
        
        if CUM_AVAILABLE:
            st.info("Using GPU-accelerated cuML for LDA")
            lda = cuLDA(
                n_components=n_topics,
                max_iter=100,
                random_state=44,
            )
        else:
            st.info("Using CPU-based scikit-learn for LDA (multi-core enabled)")
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=100,
                learning_method="batch",
                random_state=44,
                n_jobs=-1,  # Use all available CPU cores
            )
        topics = lda.fit_transform(dtm)
        
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top words for each topic
        topic_words = []
        for topic_idx in range(n_topics):
            top_words_idx = lda.components_[topic_idx].argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_words.append(top_words)
            
        results = {
            'method': 'LDA',
            'n_topics': n_topics,
            'topic_words': topic_words,
            'vocab_size': len(feature_names)
        }
        
        # Save model if requested
        if save_model and JOBLIB_AVAILABLE:
            model_data = {
                'vectorizer': vectorizer,
                'model': lda,
                'results': results,
                'method': method,
                'n_topics': n_topics,
                'trained_at': datetime.now().isoformat()
            }
            joblib.dump(model_data, model_path)
            st.success(f"✅ LDA model saved to {model_path}")
        
        return results
    
    elif method == 'LSA':
        # LSA configuration
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
            max_features=None,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.6,
            dtype=np.float32,
        )
        tfidf = vectorizer.fit_transform(processed_texts)
        
        if CUM_AVAILABLE:
            st.info("Using GPU-accelerated cuML for LSA")
            svd = cuSVD(n_components=min(n_topics, tfidf.shape[1]-1), random_state=42)
        else:
            st.info("Using CPU-based scikit-learn for LSA")
            svd = TruncatedSVD(n_components=min(n_topics, tfidf.shape[1]-1), random_state=42)
        topics = svd.fit_transform(tfidf)
        
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top words for each topic
        topic_words = []
        for topic_idx in range(min(n_topics, svd.components_.shape[0])):
            top_words_idx = svd.components_[topic_idx].argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_words.append(top_words)
            
        results = {
            'method': 'LSA',
            'n_topics': len(topic_words),
            'topic_words': topic_words,
            'explained_variance': svd.explained_variance_ratio_.sum()
        }
        
        # Save model if requested
        if save_model and JOBLIB_AVAILABLE:
            model_data = {
                'vectorizer': vectorizer,
                'model': svd,
                'results': results,
                'method': method,
                'n_topics': n_topics,
                'trained_at': datetime.now().isoformat()
            }
            joblib.dump(model_data, model_path)
            st.success(f"✅ LSA model saved to {model_path}")
        
        return results
    
    return None

@st.cache_data
def simple_tokenize(text):
    """Simple tokenization"""
    if pd.isna(text):
        return []
    return str(text).split()

@st.cache_resource
def train_word2vec_model(job_texts, resume_texts, save_model=True):
    """Train Word2Vec model on combined job and resume texts"""
    if not GENSIM_AVAILABLE:
        return None
    
    # Create models directory if it doesn't exist
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        models_dir = os.path.join(workspace_path, "models")
        os.makedirs(models_dir, exist_ok=True)
    else:
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
    
    model_filename = "word2vec_model.joblib"
    model_path = os.path.join(models_dir, model_filename)
    
    # Check if model already exists
    if os.path.exists(model_path) and JOBLIB_AVAILABLE:
        st.info(f"Loading existing Word2Vec model from {model_path}")
        try:
            saved_data = joblib.load(model_path)
            return saved_data['model']
        except Exception as e:
            st.warning(f"Could not load saved model: {e}. Retraining...")
    
    # Combine and tokenize
    training_corpus = [simple_tokenize(text) for text in job_texts + resume_texts if isinstance(text, str)]
    
    # Train model
    model = Word2Vec(
        sentences=training_corpus,
        vector_size=300,
        window=5,
        min_count=10,
        workers=4,
        sg=1,
        epochs=10
    )
    
    # Save model if requested
    if save_model and JOBLIB_AVAILABLE:
        model_data = {
            'model': model,
            'trained_at': datetime.now().isoformat(),
            'vector_size': 300,
            'window': 5,
            'min_count': 10,
            'sg': 1,
            'epochs': 10
        }
        joblib.dump(model_data, model_path)
        st.success(f"✅ Word2Vec model saved to {model_path}")
    
    return model

@st.cache_resource
def load_sbert_model():
    """Load SBERT model"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    
    # Check for available devices in order of preference: CUDA, MPS, CPU
    if torch.cuda.is_available():
        device = "cuda"
        st.info("Using CUDA GPU for SBERT")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        st.info("Using Apple MPS (Metal) for SBERT")
    else:
        device = "cpu"
        st.info("Using CPU for SBERT")
    
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model = model.to(device)
    return model

def get_doc_embedding_w2v(tokens, model):
    """Get document embedding using Word2Vec"""
    if not GENSIM_AVAILABLE or model is None:
        return np.zeros(300, dtype="float32")
    
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(300, dtype="float32")
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

@st.cache_data
def compute_job_embeddings_sbert(job_texts, _model):
    """Compute embeddings for jobs using SBERT"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE or _model is None:
        return None
    
    embeddings = _model.encode(job_texts, batch_size=64, show_progress_bar=False, convert_to_tensor=False)
    return np.array(embeddings)

@st.cache_data
def find_similar_jobs_w2v(query_text, job_texts, job_embeddings, df, valid_indices, top_k=5):
    """Find similar jobs using Word2Vec"""
    query_tokens = simple_tokenize(query_text)
    query_emb = get_doc_embedding_w2v(query_tokens, st.session_state.get('w2v_model'))
    query_emb = query_emb.reshape(1, -1)
    
    similarities = cosine_similarity(query_emb, job_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        # Use the original dataframe index from valid_indices
        original_idx = valid_indices[idx]
        job_row = df.loc[original_idx]
        results.append({
            'job_id': job_row.get('Job Id', 'N/A'),
            'job_title': job_row.get('Job Title', 'N/A'),
            'company': job_row.get('Company', 'N/A'),
            'job_link': job_row.get('Job Link', 'N/A'),
            'company_link': job_row.get('Company Link', 'N/A'),
            'location': job_row.get('location', 'N/A'),
            'country': job_row.get('Country', 'N/A'),
            'salary_range': job_row.get('Salary Range', 'N/A'),
            'experience': job_row.get('Experience', 'N/A'),
            'benefits': job_row.get('Benefits', 'N/A'),
            'skills': job_row.get('skills', 'N/A'),
            'responsibilities': job_row.get('Responsibilities', 'N/A'),
            'job_description': job_row.get('Job Description', job_row.get('Description', 'N/A')),
            'similarity': similarities[idx]
        })
    return results

@st.cache_data
def find_similar_jobs_sbert(query_text, job_texts, job_embeddings, _model, df, valid_indices, top_k=5):
    """Find similar jobs using SBERT"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE or _model is None:
        return []
    
    query_emb = _model.encode([query_text], batch_size=1, show_progress_bar=False, convert_to_tensor=False)[0]
    query_emb = query_emb.reshape(1, -1)
    
    similarities = cosine_similarity(query_emb, job_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        # Use the original dataframe index from valid_indices
        original_idx = valid_indices[idx]
        job_row = df.loc[original_idx]
        results.append({
            'job_id': job_row.get('Job Id', 'N/A'),
            'job_title': job_row.get('Job Title', 'N/A'),
            'company': job_row.get('Company', 'N/A'),
            'job_link': job_row.get('Job Link', 'N/A'),
            'company_link': job_row.get('Company Link', 'N/A'),
            'location': job_row.get('location', 'N/A'),
            'country': job_row.get('Country', 'N/A'),
            'salary_range': job_row.get('Salary Range', 'N/A'),
            'experience': job_row.get('Experience', 'N/A'),
            'benefits': job_row.get('Benefits', 'N/A'),
            'skills': job_row.get('skills', 'N/A'),
            'responsibilities': job_row.get('Responsibilities', 'N/A'),
            'job_description': job_row.get('Job Description', job_row.get('Description', 'N/A')),
            'similarity': similarities[idx]
        })
    return results

# Page configuration
st.set_page_config(
    page_title="NLP Analytics - Job Analysis",
    page_icon="🤖",
    layout="wide"
)

# Load global CSS
try:
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.title("NLP Analytics - Job Description Analysis")

# Initialize session state
if 'jobs_df' not in st.session_state:
    st.session_state.jobs_df = None
if 'resumes_df' not in st.session_state:
    st.session_state.resumes_df = None
if 'ner_results' not in st.session_state:
    st.session_state.ner_results = None
if 'topic_model_results' not in st.session_state:
    st.session_state.topic_model_results = None
if 'embedding_results' not in st.session_state:
    st.session_state.embedding_results = None
if 'w2v_model' not in st.session_state:
    st.session_state.w2v_model = None
if 'sbert_model' not in st.session_state:
    st.session_state.sbert_model = None
if 'job_embeddings_w2v' not in st.session_state:
    st.session_state.job_embeddings_w2v = None
if 'job_embeddings_sbert' not in st.session_state:
    st.session_state.job_embeddings_sbert = None

# Function to load job data
@st.cache_data
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
        else:
            # Fallback to cleaned job data (as used in notebooks)
            data_path = os.path.join(workspace_path, "Data_Cleaning", "cleaned_job_data_dedup.csv")
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                return df
            else:
                # Fallback to other locations
                data_path = os.path.join(workspace_path, "Data", "Jobs_data.csv")
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path)
                    return df
    return None

# Function to load resume data
@st.cache_data
def load_resume_data():
    """Load resume data from workspace"""
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        resume_path = os.path.join(workspace_path, "Data_Cleaning", "cleaned_resume.csv")
        if os.path.exists(resume_path):
            df = pd.read_csv(resume_path)
            return df
    return None

# Tabs for different NLP tasks
tab1, tab2, tab3, tab4 = st.tabs([
    "Named Entity Recognition",
    "Topic Modeling",
    "Word Embeddings",
    "Resume Matching"
])

with tab1:
    st.markdown("### Named Entity Recognition (NER)")
    st.markdown("""
    Extract structured information from job descriptions including:
    - **Skills & Technologies**: Programming languages, frameworks, tools
    - **Qualifications**: Degrees, certifications, experience levels
    - **Companies & Organizations**: Company names and related entities
    - **Locations**: Cities, countries, and work settings
    """)
    
    if st.session_state.jobs_df is None:
        if st.button("Load Job Data", key="ner_load"):
            with st.spinner("Loading job data..."):
                df = load_job_data()
                if df is not None:
                    st.session_state.jobs_df = df
                    st.success(f"✅ Loaded {len(df):,} job postings")
                    st.rerun()
                else:
                    st.error("❌ Could not load data")
    else:
        df = st.session_state.jobs_df
        st.success(f"✅ Working with {len(df):,} job postings")
        
        # NER Model Selection
        col1, col2 = st.columns([2, 1])
        with col1:
            ner_model = st.selectbox(
                "Select NER Model",
                ["spaCy (en_core_web_sm)", "spaCy (en_core_web_lg)", "Custom Trained Model"],
                help="Choose the NER model to extract entities"
            )
        with col2:
            sample_size = st.number_input(
                "Sample Size",
                min_value=10,
                max_value=len(df),
                value=min(100, len(df)),
                help="Number of job descriptions to analyze"
            )
        
        if st.button("Run NER Analysis", type="primary"):
            if not SPACY_AVAILABLE:
                st.error("spaCy is not available. Please install requirements-nlp.txt")
            else:
                with st.spinner("Running Named Entity Recognition..."):
                    # Load models
                    nlp = load_spacy_model()
                    skill_matcher = build_skill_ner(MASTER_SKILL_LIST)
                    
                    if nlp is None or skill_matcher is None:
                        st.error("Could not load spaCy model")
                    else:
                        # Sample some job descriptions
                        # Use appropriate column for job text
                        text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                        if text_column not in df.columns:
                            st.error(f"Could not find job text column. Available columns: {list(df.columns)}")
                            st.stop()
                        
                        sample_texts = df[text_column].dropna().sample(min(sample_size, len(df))).tolist()
                        
                        all_skills = []
                        all_spacy_entities = []
                        
                        for text in sample_texts:
                            skills = extract_skill_entities(text, skill_matcher)
                            all_skills.extend(skills)
                            
                            spacy_ents = extract_spacy_entities(text)
                            all_spacy_entities.extend(spacy_ents)
                        
                        # Count frequencies
                        skill_counts = Counter(all_skills)
                        entity_counts = Counter([label for _, label in all_spacy_entities])
                        
                        # Store results
                        st.session_state.ner_results = {
                            'total_entities': len(all_spacy_entities),
                            'unique_skills': len(skill_counts),
                            'unique_orgs': entity_counts.get('ORG', 0),
                            'skill_counts': skill_counts.most_common(20),
                            'entity_counts': entity_counts.most_common(10)
                        }
                        
                        st.success("✅ NER Analysis completed!")
                        st.rerun()
        
        if st.session_state.ner_results:
            results = st.session_state.ner_results
            st.markdown("#### NER Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Entities", f"{results['total_entities']:,}")
            with col2:
                st.metric("Unique Skills", f"{results['unique_skills']:,}")
            with col3:
                st.metric("Unique Organizations", f"{results['unique_orgs']:,}")
            
            # Show top skills
            if 'skill_counts' in results:
                st.markdown("**Top Skills:**")
                skill_df = pd.DataFrame(results['skill_counts'], columns=['Skill', 'Count'])
                st.dataframe(skill_df, use_container_width=True)
            
            # Show entity distribution
            if 'entity_counts' in results:
                st.markdown("**Entity Types:**")
                ent_df = pd.DataFrame(results['entity_counts'], columns=['Entity Type', 'Count'])
                fig = px.bar(ent_df, x='Entity Type', y='Count', title='Entity Distribution')
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Topic Modeling")
    st.markdown("""
    Discover hidden themes and topics in job descriptions using:
    - **Latent Dirichlet Allocation (LDA)**: Probabilistic topic modeling
    - **Latent Semantic Analysis (LSA)**: SVD-based topic extraction
    """)
    
    if st.session_state.cleaned_jobs_df is None:
        if st.button("Load Job Data", key="topic_load"):
            with st.spinner("Loading job data..."):
                df = load_job_data()
                if df is not None:
                    st.session_state.cleaned_jobs_df = df
                    st.success(f"✅ Loaded {len(df):,} job postings")
                    st.rerun()
                else:
                    st.error("❌ Could not load data")
    else:
        df = st.session_state.cleaned_jobs_df
        st.success(f"✅ Working with {len(df):,} job postings")
        
        # Topic Modeling Settings
        col1, col2, col3 = st.columns(3)
        with col1:
            method = st.selectbox(
                "Topic Modeling Method",
                ["LDA", "LSA"],
                help="Choose the topic modeling algorithm"
            )
        with col2:
            num_topics = st.slider(
                "Number of Topics",
                min_value=3,
                max_value=20,
                value=10,
                help="Number of topics to extract"
            )
        with col3:
            num_words = st.slider(
                "Words per Topic",
                min_value=5,
                max_value=20,
                value=10,
                help="Number of top words to display per topic"
            )
        
        # Model saving option
        save_model = st.checkbox(
            "Save trained model for future use",
            value=True,
            help="Save the trained topic model to disk for faster loading on subsequent runs"
        )
        
        if st.button("Run Topic Modeling", type="primary"):
            if not SKLEARN_AVAILABLE:
                st.error("scikit-learn is not available. Please install requirements.")
            else:
                with st.spinner(f"Running {method} topic modeling..."):
                    # Get job texts
                    text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                    if text_column not in df.columns:
                        st.error(f"Could not find job text column. Available columns: {list(df.columns)}")
                        st.stop()
                    texts = df[text_column].dropna().tolist()
                    
                    # Run topic modeling
                    results = run_topic_modeling(texts, method, num_topics, num_words, save_model)
                    
                    if results:
                        st.session_state.topic_model_results = results
                        st.success(f"✅ {method} topic modeling completed!")
                        st.rerun()
                    else:
                        st.error("Topic modeling failed")
        
        if st.session_state.topic_model_results:
            results = st.session_state.topic_model_results
            st.markdown("#### Topic Modeling Results")
            st.success(f"✅ Extracted {results['n_topics']} topics using {results['method']}")
            
            if 'vocab_size' in results:
                st.info(f"Vocabulary size: {results['vocab_size']:,}")
            if 'explained_variance' in results:
                st.info(f"Explained variance: {results['explained_variance']:.3f}")
            
            # Show topics
            st.markdown("**Discovered Topics:**")
            for i, words in enumerate(results['topic_words'][:min(5, len(results['topic_words']))]):
                st.write(f"**Topic {i+1}:** {', '.join(words)}")
            
            if len(results['topic_words']) > 5:
                with st.expander("Show all topics"):
                    for i, words in enumerate(results['topic_words']):
                        st.write(f"**Topic {i+1}:** {', '.join(words)}")
        
        # Show saved models
        st.markdown("#### Saved Models")
        workspace_path = st.session_state.get('workspace_path')
        if workspace_path:
            models_dir = os.path.join(workspace_path, "models")
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.startswith('topic_model_') and f.endswith('.joblib')]
                if model_files:
                    st.markdown("**Available saved models:**")
                    for model_file in sorted(model_files):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"📁 {model_file}")
                        with col2:
                            if st.button("Load", key=f"load_{model_file}"):
                                model_path = os.path.join(models_dir, model_file)
                                try:
                                    saved_data = joblib.load(model_path)
                                    st.session_state.topic_model_results = saved_data['results']
                                    st.success(f"✅ Loaded {model_file}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to load model: {e}")
                        with col3:
                            if st.button("Delete", key=f"delete_{model_file}"):
                                model_path = os.path.join(models_dir, model_file)
                                try:
                                    os.remove(model_path)
                                    st.success(f"🗑️ Deleted {model_file}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to delete: {e}")
                else:
                    st.info("No saved topic models found.")
            else:
                st.info("Models directory not found.")

with tab3:
    st.markdown("### Word Embeddings")
    st.markdown("""
    Analyze semantic relationships between words and find similar jobs using:
    - **Word2Vec**: Neural word embeddings
    - **Sentence-BERT (SBERT)**: Sentence-level embeddings for job matching
    """)
    
    if st.session_state.cleaned_jobs_df is None:
        if st.button("Load Job Data", key="embedding_load_jobs"):
            with st.spinner("Loading job data..."):
                df = load_job_data()
                if df is not None:
                    st.session_state.cleaned_jobs_df = df
                    st.success(f"✅ Loaded {len(df):,} job postings")
                    st.rerun()
                else:
                    st.error("❌ Could not load data")
    else:
        df = st.session_state.cleaned_jobs_df
        st.success(f"✅ Working with {len(df):,} job postings")
        
        # Load resume data for training
        if st.session_state.resumes_df is None:
            if st.button("Load Resume Data", key="embedding_load_resumes"):
                with st.spinner("Loading resume data..."):
                    resume_df = load_resume_data()
                    if resume_df is not None:
                        st.session_state.resumes_df = resume_df
                        st.success(f"✅ Loaded {len(resume_df):,} resumes")
                        st.rerun()
                    else:
                        st.error("❌ Could not load resume data")
        else:
            resume_df = st.session_state.resumes_df
            st.success(f"✅ Working with {len(resume_df):,} resumes")
        
        # Word Embedding Settings
        embedding_method = st.selectbox(
            "Embedding Method",
            ["Word2Vec", "Sentence-BERT (SBERT)"],
            help="Choose the embedding method"
        )
        
        # Model saving option for Word2Vec
        if embedding_method == "Word2Vec":
            save_w2v_model = st.checkbox(
                "Save trained Word2Vec model for future use",
                value=True,
                help="Save the trained Word2Vec model to disk for faster loading on subsequent runs"
            )
        
        # Load/Train models
        if embedding_method == "Word2Vec":
            if st.session_state.w2v_model is None:
                if st.button("Train Word2Vec Model", key="train_w2v"):
                    if not GENSIM_AVAILABLE:
                        st.error("Gensim not available")
                    elif st.session_state.resumes_df is None:
                        st.error("Please load resume data first")
                    else:
                        with st.spinner("Training Word2Vec model..."):
                            text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                            job_texts = df[text_column].dropna().tolist()
                            resume_texts = resume_df['cleaned_text'].dropna().tolist()
                            model = train_word2vec_model(job_texts, resume_texts, save_model=save_w2v_model)
                            if model:
                                st.session_state.w2v_model = model
                                st.success("✅ Word2Vec model trained!")
                                st.rerun()
            else:
                st.success("✅ Word2Vec model ready")
                
                # Show saved models
                st.markdown("#### Saved Word2Vec Models")
                workspace_path = st.session_state.get('workspace_path')
                if workspace_path:
                    models_dir = os.path.join(workspace_path, "models")
                    if os.path.exists(models_dir):
                        w2v_files = [f for f in os.listdir(models_dir) if f == 'word2vec_model.joblib']
                        if w2v_files:
                            st.markdown("**Available saved Word2Vec model:**")
                            model_file = w2v_files[0]
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                st.write(f"📁 {model_file}")
                            with col2:
                                if st.button("Load", key="load_w2v"):
                                    model_path = os.path.join(models_dir, model_file)
                                    try:
                                        saved_data = joblib.load(model_path)
                                        st.session_state.w2v_model = saved_data['model']
                                        st.success(f"✅ Loaded {model_file}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to load model: {e}")
                            with col3:
                                if st.button("Delete", key="delete_w2v"):
                                    model_path = os.path.join(models_dir, model_file)
                                    try:
                                        os.remove(model_path)
                                        st.success(f"🗑️ Deleted {model_file}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to delete: {e}")
                        else:
                            st.info("No saved Word2Vec model found.")
                    else:
                        st.info("Models directory not found.")
                
                # Compute embeddings
                if st.session_state.job_embeddings_w2v is None:
                    if st.button("Compute Job Embeddings", key="compute_w2v_emb"):
                        with st.spinner("Computing job embeddings..."):
                            text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                            job_texts = df[text_column].dropna().tolist()
                            embeddings = compute_job_embeddings_w2v(job_texts, st.session_state.w2v_model)
                            st.session_state.job_embeddings_w2v = embeddings
                            st.success("✅ Job embeddings computed!")
                            st.rerun()
                else:
                    st.success("✅ Job embeddings ready")
        
        elif embedding_method == "Sentence-BERT (SBERT)":
            if st.session_state.sbert_model is None:
                if st.button("Load SBERT Model", key="load_sbert"):
                    if not SENTENCE_TRANSFORMERS_AVAILABLE:
                        st.error("Sentence Transformers not available")
                    else:
                        with st.spinner("Loading SBERT model..."):
                            model = load_sbert_model()
                            if model:
                                st.session_state.sbert_model = model
                                st.success("✅ SBERT model loaded!")
                                st.rerun()
            else:
                st.success("✅ SBERT model ready")
                
                # Compute embeddings
                if st.session_state.job_embeddings_sbert is None:
                    if st.button("Compute Job Embeddings", key="compute_sbert_emb"):
                        with st.spinner("Computing job embeddings..."):
                            text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                            job_texts = df[text_column].dropna().tolist()
                            embeddings = compute_job_embeddings_sbert(job_texts, st.session_state.sbert_model)
                            if embeddings is not None:
                                st.session_state.job_embeddings_sbert = embeddings
                                st.success("✅ Job embeddings computed!")
                                st.rerun()
                else:
                    st.success("✅ Job embeddings ready")
        
        st.markdown("#### Find Similar Jobs")
        
        query_text = st.text_area(
            "Enter job description or resume text",
            height=100,
            placeholder="Paste a job description or resume text to find similar jobs..."
        )
        
        top_k = st.slider("Number of similar jobs to show", 3, 20, 5)
        
        if st.button("Find Similar Jobs", type="primary"):
            if not query_text:
                st.error("Please enter some text")
            elif embedding_method == "Word2Vec" and st.session_state.job_embeddings_w2v is None:
                st.error("Please compute Word2Vec embeddings first")
            elif embedding_method == "Sentence-BERT (SBERT)" and st.session_state.job_embeddings_sbert is None:
                st.error("Please compute SBERT embeddings first")
            else:
                with st.spinner("Finding similar jobs..."):
                    text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                    # Keep track of valid indices before dropping NaN
                    valid_mask = df[text_column].notna()
                    job_texts = df[text_column][valid_mask].tolist()
                    valid_indices = df[valid_mask].index.tolist()
                    
                    if embedding_method == "Word2Vec":
                        results = find_similar_jobs_w2v(query_text, job_texts, st.session_state.job_embeddings_w2v, df, valid_indices, top_k)
                    else:
                        results = find_similar_jobs_sbert(query_text, job_texts, st.session_state.job_embeddings_sbert, st.session_state.sbert_model, df, valid_indices, top_k)
                    
                    st.session_state.embedding_results = {
                        'method': embedding_method,
                        'query': query_text,
                        'results': results
                    }
                    st.success("✅ Similar jobs found!")
                    st.rerun()
        
        if st.session_state.embedding_results:
            results = st.session_state.embedding_results
            st.markdown("#### Similar Jobs Results")
            st.success(f"✅ Found similar jobs using {results['method']}")
            
            # Show results
            for i, job in enumerate(results['results'], 1):
                with st.expander(f"Match {i}: Similarity {job['similarity']:.3f} - {job['job_title']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Job Title:** {job['job_title']}")
                        st.markdown(f"**Company:** {job['company']}")
                        if job['location'] != 'N/A':
                            st.markdown(f"**Location:** {job['location']}, {job['country']}")
                        elif job['country'] != 'N/A':
                            st.markdown(f"**Country:** {job['country']}")
                        if job['salary_range'] != 'N/A':
                            st.markdown(f"**Salary Range:** {job['salary_range']}")
                        if job['experience'] != 'N/A':
                            st.markdown(f"**Experience:** {job['experience']}")
                    
                    with col2:
                        if job['job_link'] != 'N/A':
                            st.markdown(f"[🔗 Job Link]({job['job_link']})")
                        if job['company_link'] != 'N/A':
                            st.markdown(f"[🏢 Company Link]({job['company_link']})")
                    
                    st.markdown("**Job Description:**")
                    st.write(job['job_description'])
                    
                    if job['skills'] != 'N/A':
                        st.markdown("**Skills:**")
                        st.write(job['skills'])
                    
                    if job['responsibilities'] != 'N/A':
                        st.markdown("**Responsibilities:**")
                        st.write(job['responsibilities'])
                    
                    if job['benefits'] != 'N/A':
                        st.markdown("**Benefits:**")
                        st.write(job['benefits'])

with tab4:
    st.markdown("### Resume Matching")
    st.markdown("""
    Match resumes to job descriptions using OpenAI embeddings and vector similarity:
    - Generate 1536-dimensional embeddings for resumes and jobs
    - Store job embeddings in PostgreSQL with pgvector
    - Find top 10 similar jobs using cosine similarity
    """)
    
    # Database setup check
    try:
        from functions.database import generate_openai_embedding, insert_job_with_embedding, find_similar_jobs, batch_insert_jobs
        DB_AVAILABLE = True
    except ImportError:
        DB_AVAILABLE = False
        st.error("Database functions not available")
    
    if not DB_AVAILABLE:
        st.stop()
    
    # Load and insert jobs into DB
    if st.session_state.jobs_df is None:
        if st.button("Load Job Data", key="resume_load_jobs"):
            with st.spinner("Loading job data..."):
                df = load_job_data()
                if df is not None:
                    st.session_state.jobs_df = df
                    st.success(f"✅ Loaded {len(df):,} job postings")
                    st.rerun()
                else:
                    st.error("❌ Could not load data")
    else:
        df = st.session_state.jobs_df
        st.success(f"✅ Working with {len(df):,} job postings")
        
        # Insert jobs into database
        if 'jobs_inserted' not in st.session_state:
            if st.button("Insert Jobs into Database", key="insert_jobs"):
                with st.spinner("Inserting jobs with embeddings..."):
                    success_count = batch_insert_jobs(df)
                    st.session_state.jobs_inserted = True
                    st.success(f"✅ Inserted {success_count} jobs into database")
                    st.rerun()
        else:
            st.success("✅ Jobs inserted into database")
        
        # Resume upload and matching
        uploaded_resume = st.file_uploader(
            "Upload Resume (TXT or PDF)",
            type=['txt', 'pdf'],
            help="Upload your resume to find matching jobs"
        )
        
        resume_text = st.text_area(
            "Or paste resume text",
            height=200,
            placeholder="Paste your resume text here..."
        )
        
        if uploaded_resume or resume_text:
            if uploaded_resume:
                # Extract text from uploaded file
                try:
                    if uploaded_resume.type == "text/plain":
                        resume_content = uploaded_resume.read().decode("utf-8")
                    else:
                        # For PDF, use pypdf or similar
                        from pypdf import PdfReader
                        pdf_reader = PdfReader(uploaded_resume)
                        resume_content = ""
                        for page in pdf_reader.pages:
                            resume_content += page.extract_text()
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    resume_content = ""
            else:
                resume_content = resume_text
            
            if resume_content:
                st.success(f"✅ Resume loaded ({len(resume_content)} characters)")
                
                if st.button("Find Matching Jobs", type="primary"):
                    with st.spinner("Generating embedding and finding matches..."):
                        # Generate embedding for resume
                        resume_embedding = generate_openai_embedding(resume_content)
                        
                        if resume_embedding:
                            # Find similar jobs
                            similar_jobs = find_similar_jobs(resume_embedding, top_k=10)
                            
                            if similar_jobs:
                                st.markdown("#### Top 10 Matching Jobs")
                                for i, job in enumerate(similar_jobs, 1):
                                    with st.expander(f"Match {i}: Similarity {job['similarity']:.3f} - {job.get('title', 'N/A')} at {job.get('company', 'N/A')}"):
                                        st.write(job['text'][:500] + "..." if len(job['text']) > 500 else job['text'])
                            else:
                                st.error("No similar jobs found")
                        else:
                            st.error("Failed to generate embedding. Check OpenAI API key.")
        
        st.info("💡 Upload a resume or paste text to get started with job matching. Jobs must be inserted into database first.")

# Footer
st.markdown("---")
st.markdown("""
### 📚 Available Resources

This Streamlit interface implements the NLP techniques from the following Jupyter notebooks:
- **NER**: `workspace/NER/NER.ipynb` - Custom skill extraction using spaCy PhraseMatcher
- **Topic Modeling**: 
  - LDA: `workspace/Topic Modeling/TopicModeling_LDA.ipynb` - CountVectorizer + LDA
  - LSA: `workspace/Topic Modeling/TopicModeling_LSA.ipynb` - TF-IDF + TruncatedSVD
- **Word Embeddings**: 
  - Word2Vec: `workspace/Word Embedding/Word Embedding_Word2Vector_UseDedup.ipynb` - Document embeddings for job matching
  - SBERT: `workspace/Word Embedding/Word_Embedding_SBERT_UseDedup.ipynb` - Sentence-BERT for semantic similarity
- **Resume Testing**: `workspace/Resume_testing/` - Resume-job matching examples

**Database Integration:**
- Jobs are stored in PostgreSQL with pgvector for vector similarity search
- Resume-job matching uses OpenAI embeddings (1536 dimensions) and cosine similarity
- Run `create_jobs_table.sql` to set up the database schema

**Model Persistence:**
- Topic models (LDA/LSA) are automatically saved to `workspace/models/` directory
- Word2Vec models can be saved for future use and reloaded to avoid retraining
- Saved models can be reloaded for faster analysis without retraining
- Models are stored in joblib format with metadata

**Note**: The implementations above use the cleaned datasets from `Data_Cleaning/cleaned_job_data_dedup.csv` and `Data_Cleaning/cleaned_resume.csv`.
""")
