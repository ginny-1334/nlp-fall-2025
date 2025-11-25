import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import streamlit as st
from datetime import datetime

# Try to import NLP libraries
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
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

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

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

# NER functions
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

def skill_jaccard_score(resume_skills, job_skills):
    """
    Jaccard similarity between resume skills and job skills.
    = overlap / union
    """
    resume_set = set(resume_skills)
    job_set = set(job_skills)

    # If both are empty, return 0
    union = resume_set | job_set
    if not union:
        return 0.0

    overlap = resume_set & job_set
    score = len(overlap) / len(union)
    return score

# Tokenization functions
@st.cache_data
def simple_tokenize(text):
    """Simple tokenization"""
    if pd.isna(text):
        return []
    return str(text).split()

# SBERT functions
@st.cache_resource
def load_sbert_model():
    """Load SBERT model"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    # Check for available devices in order of preference: CUDA, MPS, CPU
    device = "cpu"
    
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU for SBERT")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Test MPS with a small tensor to ensure it works
        try:
            test_device = torch.device('mps')
            test_tensor = torch.randn(1).to(test_device)
            device = "mps"
            print("Using Apple MPS (Metal) for SBERT")
        except Exception as e:
            print(f"MPS test failed ({e}), falling back to CPU")
            device = "cpu"
    else:
        device = "cpu"
        print("Using CPU for SBERT")

    try:
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
        return model
    except Exception as e:
        print(f"Failed to load model on {device}: {e}")
        # Fallback to CPU
        try:
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
            return model
        except Exception as e2:
            print(f"Failed to load model on CPU: {e2}")
            return None

@st.cache_data
def compute_job_embeddings_sbert(job_texts, _model):
    """Compute embeddings for jobs using SBERT"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE or _model is None:
        return None

    # Ensure model is on the correct device
    device = next(_model.parameters()).device
    print(f"Computing SBERT embeddings on device: {device}")

    # Adjust batch size based on device
    if device.type == 'mps':
        batch_size = 64  # Smaller for MPS memory
    elif device.type == 'cuda':
        batch_size = 128
    else:
        batch_size = 32  # Smaller for CPU memory

    try:
        embeddings = _model.encode(
            job_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=False,
            device=device
        )
        return np.array(embeddings)
    except Exception as e:
        print(f"Embedding computation failed on {device}: {e}")
        # Try on CPU as fallback
        if device.type != 'cpu':
            print("Falling back to CPU...")
            try:
                cpu_model = _model.to('cpu')
                embeddings = cpu_model.encode(
                    job_texts,
                    batch_size=16,  # Very small batch for CPU
                    show_progress_bar=True,
                    convert_to_tensor=False,
                    device='cpu'
                )
                return np.array(embeddings)
            except Exception as e2:
                print(f"CPU fallback also failed: {e2}")
                return None
        return None

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
            'id': job_row.get('id', 'N/A'),
            'title': job_row.get('Job Title', 'N/A'),
            'company': job_row.get('Company', 'N/A'),
            'text': job_row.get('Description', 'N/A'),
            'similarity': similarities[idx]
        })
    return results

# Word2Vec functions
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
        print(f"Loading existing Word2Vec model from {model_path}")
        try:
            model_data = joblib.load(model_path)
            return model_data['model']
        except Exception as e:
            print(f"Error loading model: {e}")

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
        print(f"✅ Word2Vec model saved to {model_path}")

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
            'id': job_row.get('id', 'N/A'),
            'title': job_row.get('title', 'N/A'),
            'company': job_row.get('company', 'N/A'),
            'text': job_row.get('text', 'N/A'),
            'similarity': similarities[idx]
        })
    return results

# Main function to get embeddings for resume matching
def generate_local_embedding(text: str, method: str = "sbert") -> Optional[np.ndarray]:
    """Generate embedding using local models (SBERT or Word2Vec)"""
    if method == "sbert":
        model = load_sbert_model()
        if model is None:
            return None
        device = next(model.parameters()).device
        try:
            return model.encode([text], batch_size=1, show_progress_bar=False, convert_to_tensor=False, device=device)[0]
        except Exception as e:
            print(f"SBERT encoding failed on {device}: {e}")
            # Fallback to CPU
            if device.type != 'cpu':
                try:
                    cpu_model = model.to('cpu')
                    return cpu_model.encode([text], batch_size=1, show_progress_bar=False, convert_to_tensor=False, device='cpu')[0]
                except Exception as e2:
                    print(f"CPU fallback failed: {e2}")
                    return None
            return None
    elif method == "word2vec":
        # This would require a trained Word2Vec model
        # For now, return None as we need to train/load the model first
        return None
    else:
        return None

# Function to find similar jobs using local embeddings
def find_similar_jobs_local(query_embedding: np.ndarray, job_embeddings: np.ndarray,
                           df: pd.DataFrame, top_k: int = 10) -> List[Dict]:
    """Find similar jobs using local embeddings and cosine similarity"""
    if not SKLEARN_AVAILABLE or job_embeddings is None or query_embedding is None:
        return []

    query_emb = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_emb, job_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        job_row = df.iloc[idx]
        results.append({
            'id': job_row.get('id', 'N/A'),
            'title': job_row.get('title', 'N/A'),
            'company': job_row.get('company', 'N/A'),
            'text': job_row.get('text', 'N/A'),
            'similarity': similarities[idx]
        })
    return results

# Functions for loading and using trained models
@st.cache_resource
def load_trained_word2vec_model():
    """Load trained Word2Vec model from disk"""
    if not JOBLIB_AVAILABLE:
        return None

    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        models_dir = os.path.join(workspace_path, "models")
    else:
        models_dir = "models"

    model_path = os.path.join(models_dir, "word2vec_model.joblib")

    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            return model_data['model']
        except Exception as e:
            print(f"Error loading Word2Vec model: {e}")
            return None
    else:
        print(f"Word2Vec model not found at {model_path}")
        return None

@st.cache_resource
def load_trained_topic_model(method='LDA', n_topics=10):
    """Load trained topic model from disk"""
    if not JOBLIB_AVAILABLE:
        return None

    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        models_dir = os.path.join(workspace_path, "models")
    else:
        models_dir = "models"

    model_filename = f"topic_model_{method.lower()}_{n_topics}topics.joblib"
    model_path = os.path.join(models_dir, model_filename)

    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            return {
                'vectorizer': model_data['vectorizer'],
                'model': model_data['model'],
                'results': model_data['results']
            }
        except Exception as e:
            print(f"Error loading topic model: {e}")
            return None
    else:
        print(f"Topic model not found at {model_path}")
        return None

def get_document_topics(text, topic_model_data):
    """Get topic distribution for a document"""
    if topic_model_data is None or not SKLEARN_AVAILABLE:
        return None

    vectorizer = topic_model_data['vectorizer']
    model = topic_model_data['model']

    # Transform the text
    dtm = vectorizer.transform([text])

    # Get topic distribution
    if hasattr(model, 'transform'):  # For LSA/SVD
        topic_dist = model.transform(dtm)[0]
    else:  # For LDA
        topic_dist = model.transform(dtm)[0]

    return topic_dist

def compute_topic_similarity(topic_dist1, topic_dist2):
    """Compute similarity between two topic distributions"""
    if topic_dist1 is None or topic_dist2 is None:
        return 0.0

    if not SKLEARN_AVAILABLE:
        return 0.0

    # Use cosine similarity for topic distributions
    dist1 = np.array(topic_dist1).reshape(1, -1)
    dist2 = np.array(topic_dist2).reshape(1, -1)

    similarity = cosine_similarity(dist1, dist2)[0][0]
    return similarity

def generate_trained_embedding(text: str, method: str = "word2vec") -> Optional[np.ndarray]:
    """Generate embedding using trained models"""
    if method == "word2vec":
        model = load_trained_word2vec_model()
        if model is None:
            return None
        tokens = simple_tokenize(text)
        return get_doc_embedding_w2v(tokens, model)
    else:
        return None

def find_similar_jobs_trained(query_text: str, job_texts: List[str], df: pd.DataFrame,
                             top_k: int = 10, weights: Dict[str, float] = None) -> List[Dict]:
    """
    Find similar jobs using trained models (Word2Vec + Topic Modeling + Skills)

    Args:
        query_text: Resume text to match against
        job_texts: List of job descriptions
        df: DataFrame containing job data
        top_k: Number of top matches to return
        weights: Weights for different similarity components (skills, semantic, topic)

    Returns:
        List of dictionaries with job matches and combined similarity scores
    """
    if weights is None:
        weights = {'skills': 0.45, 'semantic': 0.35, 'topic': 0.20}

    # Load models
    w2v_model = load_trained_word2vec_model()
    topic_model_data = load_trained_topic_model()
    skill_matcher = build_skill_ner(MASTER_SKILL_LIST)

    # Extract skills from query
    query_skills = extract_skill_entities(query_text, skill_matcher) if skill_matcher else []

    # Get query embeddings and topics
    query_w2v_emb = None
    query_topics = None

    if w2v_model:
        query_tokens = simple_tokenize(query_text)
        query_w2v_emb = get_doc_embedding_w2v(query_tokens, w2v_model)

    if topic_model_data:
        query_topics = get_document_topics(query_text, topic_model_data)

    # Process each job
    similarities = []

    for idx, job_text in enumerate(job_texts):
        job_row = df.iloc[idx]

        # Skills similarity
        job_skills = extract_skill_entities(job_text, skill_matcher) if skill_matcher else []
        skills_sim = skill_jaccard_score(query_skills, job_skills)

        # Semantic similarity (Word2Vec)
        semantic_sim = 0.0
        if w2v_model and query_w2v_emb is not None:
            job_tokens = simple_tokenize(job_text)
            job_w2v_emb = get_doc_embedding_w2v(job_tokens, w2v_model)
            if job_w2v_emb is not None:
                semantic_sim = cosine_similarity(
                    query_w2v_emb.reshape(1, -1),
                    job_w2v_emb.reshape(1, -1)
                )[0][0]

        # Topic similarity
        topic_sim = 0.0
        if topic_model_data and query_topics is not None:
            job_topics = get_document_topics(job_text, topic_model_data)
            if job_topics is not None:
                topic_sim = compute_topic_similarity(query_topics, job_topics)

        # Combined similarity score
        combined_sim = (
            weights['skills'] * skills_sim +
            weights['semantic'] * semantic_sim +
            weights['topic'] * topic_sim
        )

        similarities.append({
            'index': idx,
            'skills_sim': skills_sim,
            'semantic_sim': semantic_sim,
            'topic_sim': topic_sim,
            'combined_sim': combined_sim,
            'job_data': {
                'id': job_row.get('id', 'N/A'),
                'title': job_row.get('Job Title', 'N/A'),
                'company': job_row.get('Company', 'N/A'),
                'text': job_row.get('Description', 'N/A')
            }
        })

    # Sort by combined similarity and return top_k
    similarities.sort(key=lambda x: x['combined_sim'], reverse=True)
    top_results = similarities[:top_k]

    # Format results
    results = []
    for sim in top_results:
        result = sim['job_data'].copy()
        result.update({
            'similarity': sim['combined_sim'],
            'skills_similarity': sim['skills_sim'],
            'semantic_similarity': sim['semantic_sim'],
            'topic_similarity': sim['topic_sim']
        })
        results.append(result)

    return results