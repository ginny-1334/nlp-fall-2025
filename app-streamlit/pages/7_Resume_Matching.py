import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import re
import json
import tempfile
import unicodedata
from typing import Optional, List, Dict, Tuple

from components.header import render_header
from utils import initialize_workspace

# Initialize workspace path and imports
initialize_workspace()

# Page configuration
st.set_page_config(
    page_title="Resume Matching - Job Search",
    page_icon="📄",
    layout="wide"
)

# Initialize session state variables used on this page
if "matching_results" not in st.session_state:
    st.session_state.matching_results = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "resume_embedding" not in st.session_state:
    st.session_state.resume_embedding = None

# Load global CSS
try:
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.title("Resume Matching")

# Professional introduction
st.markdown("""
<div style="margin-bottom: 2rem;">
    <p style="color: #6b7280; font-size: 1.1rem; line-height: 1.6;">
        Upload your resume and discover job opportunities that match your skills and experience.
        Our AI analyzes job requirements and resume content to find the best career matches.
    </p>
</div>
""", unsafe_allow_html=True)


# Try to import required libraries
try:
    from functions.nlp_models import (
        load_sbert_model, generate_local_embedding, find_similar_jobs_local,
        compute_job_embeddings_sbert, build_skill_ner, extract_skill_entities, skill_jaccard_score,
        load_spacy_model,
        SENTENCE_TRANSFORMERS_AVAILABLE,
        find_similar_jobs_trained, load_trained_word2vec_model, 
        simple_tokenize, get_doc_embedding_w2v,
        load_trained_topic_model, get_document_topics, compute_topic_similarity
    )
    import torch
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False

# Import skill lists from nlp_config
try:
    from functions.nlp_config import MASTER_SKILL_LIST, EXTRA_SKILLS
except ImportError:
    # Fallback if nlp_config is not available
    MASTER_SKILL_LIST = []
    EXTRA_SKILLS = []

try:
    from functions.database import find_similar_jobs, find_similar_jobs_vector, create_db_engine
    from sqlalchemy import text
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# Try to import spaCy for NER
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Try to import Ollama for LLM evaluation
try:
    import ollama  # type: ignore
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Try to import transformers for token counting
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Note: build_skill_ner, extract_skill_entities, and skill_jaccard_score 
# are imported from functions.nlp_models to ensure consistency across pages

# Simple keyword-based skill extraction (without NER)
def extract_skills_keywords(text: str, skill_list: List[str]) -> List[str]:
    """
    Extract skills from text using simple keyword matching (no NER).
    Checks if skills from the skill list appear in the text (case-insensitive).
    
    Args:
        text: Text to extract skills from
        skill_list: List of skills to search for
    
    Returns:
        List of matching skills (lowercase, deduplicated, sorted)
    """
    if not text or not skill_list:
        return []
    
    text_lower = text.lower()
    skills_found = set()
    
    for skill in skill_list:
        skill_lower = skill.lower().strip()
        if not skill_lower:
            continue
        
        # Check if skill appears in text (word boundary matching for single words,
        # substring matching for multi-word skills)
        if ' ' in skill_lower:
            # Multi-word skill: check if it appears as substring
            if skill_lower in text_lower:
                skills_found.add(skill_lower)
        else:
            # Single-word skill: use word boundary matching
            pattern = r'\b' + re.escape(skill_lower) + r'\b'
            if re.search(pattern, text_lower):
                skills_found.add(skill_lower)
    
    return sorted(list(skills_found))

# Ollama client setup
def _get_ollama_client() -> Optional["ollama.Client"]:  # type: ignore[name-defined]
    """Construct an Ollama client using environment variables"""
    if not OLLAMA_AVAILABLE:
        return None
    
    api_url = (
        os.getenv("OLLAMA_API_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    )
    
    try:
        return ollama.Client(host=api_url)  # type: ignore[attr-defined]
    except Exception:
        return None

# Get model name from environment
def _get_ollama_model() -> str:
    """Get Ollama model name from environment or use default"""
    return (
        os.getenv("OLLAMA_MODEL")
        or os.getenv("OLLAMA_DEFAULT_MODEL")
        or "llama3.2"
    )

# Function to analyze job description with LLM (optionally with resume comparison)
def analyze_job_description_with_llm(job_text: str, job_title: str = "N/A", company: str = "N/A", resume_text: Optional[str] = None) -> any:  # type: ignore
    """Analyze a job description using LLM. If resume_text is provided, also includes comparison and recommendations.
    
    Returns:
        If resume_text is None: returns a string with job analysis
        If resume_text is provided: returns a dict with:
            - job_analysis: string with job analysis
            - match: True/False/None
            - reasoning: string with match reasoning
            - recommendations: list of recommendations (if No match)
            - linkedin_keywords: list of keywords (if No match)
            - error: error message if any
    """
    if not OLLAMA_AVAILABLE:
        if resume_text:
            return {'error': 'Ollama not available', 'job_analysis': None, 'match': None, 'reasoning': None, 'recommendations': None, 'linkedin_keywords': None}
        return "⚠️ Ollama not available. Please install: pip install ollama"
    
    client = _get_ollama_client()
    if client is None:
        if resume_text:
            return {'error': 'Could not connect to Ollama', 'job_analysis': None, 'match': None, 'reasoning': None, 'recommendations': None, 'linkedin_keywords': None}
        return "⚠️ Could not connect to Ollama. Please ensure Ollama is running."
    
    model_name = _get_ollama_model()
    
    # Clean texts
    cleaned_job_text = clean_text(job_text)
    if resume_text:
        cleaned_resume_text = clean_text(resume_text)
    
    # Token budget allocation
    MAX_INPUT_TOKENS = 5000
    
    if resume_text:
        # Combined analysis with resume comparison
        system_prompt = (
            "You are an expert job analyst and career advisor with deep knowledge of job markets, "
            "technical requirements, and hiring practices. You help candidates understand job requirements "
            "and evaluate their fit. Focus on ESSENTIAL qualifications needed to perform core job functions. "
            "Consider transferable skills, related experience, and learning ability. "
            "Be practical and realistic in your assessments."
        )
        
        # Calculate token budget
        system_tokens = count_tokens(system_prompt)
        template_tokens = count_tokens("""JOB POSTING:
Title: {title}
Company: {company}
Description: {job_text}

RESUME:
{resume_text}

ANALYSIS AND EVALUATION:""")
        title_tokens = count_tokens(job_title)
        company_tokens = count_tokens(company)
        
        reserved_tokens = system_tokens + template_tokens + title_tokens + company_tokens + 500  # 500 for prompt structure
        available_tokens = MAX_INPUT_TOKENS - reserved_tokens
        tokens_per_text = max(500, available_tokens // 2)
        
        # Truncate texts based on token count
        job_text_preview = truncate_by_tokens(cleaned_job_text, tokens_per_text)
        resume_text_preview = truncate_by_tokens(cleaned_resume_text, tokens_per_text)
        
        user_prompt = f"""Analyze this job posting AND evaluate if the provided resume matches it. Provide a comprehensive analysis.

JOB POSTING:
Title: {job_title}
Company: {company}
Description: {job_text_preview}

RESUME:
{resume_text_preview}

ANALYSIS REQUIREMENTS:
1. **Job Analysis**:
   - Essential Requirements: Identify 3-5 CORE qualifications that are absolutely necessary (not nice-to-haves)
   - Key Responsibilities: Summarize main day-to-day duties
   - Required Skills & Technologies: List specific technical skills, tools, and technologies mentioned as required
   - Preferred/Nice-to-Have Skills: Distinguish what's optional vs. required
   - Experience Level: Assess expected experience level (entry-level, mid-level, senior, etc.)

2. **Resume-Job Match Evaluation**:
   - Identify 3-5 CORE skills/requirements essential for this role
   - Check if candidate has these core skills OR transferable/equivalent experience
   - Assess if candidate's experience level aligns with role expectations
   - Determine: Can they reasonably perform the core job functions?

OUTPUT FORMAT:
You MUST respond with valid JSON only. Use this exact structure:

{{
  "job_analysis": {{
    "essential_requirements": ["requirement 1", "requirement 2", "requirement 3"],
    "key_responsibilities": "Summary of main day-to-day duties",
    "required_skills": ["skill1", "skill2", "skill3"],
    "preferred_skills": ["optional skill1", "optional skill2"],
    "experience_level": "Entry/Mid/Senior level",
    "key_insights": ["insight 1", "insight 2", "insight 3"]
  }},
  "match_evaluation": {{
    "match": true,
    "reasoning": "One-sentence explanation of the match decision",
    "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
    "linkedin_keywords": ["keyword1", "keyword2", "keyword3"]
  }}
}}

DECISION RULES:
- Set "match": true if: Candidate has essential qualifications OR strong transferable skills that indicate they can learn/adapt
- Set "match": false only if: Missing critical core requirements that would prevent basic job performance
- If "match": false, provide "recommendations" (3-5 specific, actionable items) and "linkedin_keywords" (5-10 relevant keywords)
- If "match": true, "recommendations" and "linkedin_keywords" can be empty arrays

IMPORTANT: Return ONLY valid JSON. Do not include any text before or after the JSON."""
    else:
        # Job analysis only (no resume)
        # Clean and truncate job text if too long
        if len(cleaned_job_text) > 3000:
            truncated = cleaned_job_text[:3000]
            last_period = truncated.rfind('.')
            last_newline = truncated.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > 2500:
                cleaned_job_text = truncated[:cut_point] + "..."
            else:
                cleaned_job_text = truncated + "..."
        
        system_prompt = (
            "You are an expert job analyst and career advisor with deep knowledge of job markets, "
            "technical requirements, and hiring practices. Your analysis helps candidates understand "
            "job requirements clearly and identify what skills and experience are truly essential versus nice-to-have. "
            "Provide structured, actionable insights that are practical and realistic."
        )
        
        user_prompt = f"""Analyze this job posting and provide a comprehensive, structured analysis.

JOB POSTING:
Title: {job_title}
Company: {company}

Description:
{cleaned_job_text}

ANALYSIS REQUIREMENTS:
1. **Essential Requirements**: Identify 3-5 CORE qualifications that are absolutely necessary for this role (not nice-to-haves)
2. **Key Responsibilities**: Summarize the main day-to-day responsibilities and duties
3. **Required Skills & Technologies**: List the specific technical skills, tools, and technologies mentioned as required
4. **Preferred/Nice-to-Have Skills**: Distinguish what's truly optional vs. required
5. **Experience Level**: Assess the expected experience level (entry-level, mid-level, senior, etc.)
6. **Industry/Context**: Note any industry-specific requirements or context

OUTPUT FORMAT:
Provide your analysis in a clear, structured format with the following sections:
- **Essential Requirements**: [List 3-5 core requirements]
- **Key Responsibilities**: [Summarize main duties]
- **Required Skills**: [List required technical skills and technologies]
- **Preferred Skills**: [List optional/nice-to-have skills, if any]
- **Experience Level**: [Entry/Mid/Senior level assessment]
- **Key Insights**: [2-3 actionable insights about what makes a strong candidate for this role]

Be specific and practical. Focus on what candidates need to know to assess their fit and prepare their application."""

    try:
        response = client.chat(  # type: ignore[attr-defined]
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.get('message', {}).get('content', 'No response from LLM')
        
        if resume_text:
            # Parse combined response
            return parse_combined_llm_response(content)
        else:
            # Return simple string for job analysis only
            return content
            
    except Exception as e:
        if resume_text:
            return {'error': f'LLM evaluation error: {str(e)}', 'job_analysis': None, 'match': None, 'reasoning': None, 'recommendations': None, 'linkedin_keywords': None}
        return f"⚠️ Error analyzing job description: {str(e)}"

# Helper function to parse combined LLM response (job analysis + match evaluation)
def parse_combined_llm_response(content: str) -> Dict[str, any]:  # type: ignore
    """Parse LLM response that contains both job analysis and match evaluation.
    First tries to parse as JSON, falls back to text parsing if JSON fails."""
    
    result = {
        'job_analysis': None,
        'match': None,
        'reasoning': None,
        'recommendations': None,
        'linkedin_keywords': None,
        'error': None
    }
    
    # Try to parse as JSON first
    try:
        # Clean content - remove markdown code blocks if present
        cleaned_content = content.strip()
        if cleaned_content.startswith('```'):
            # Remove markdown code blocks
            lines = cleaned_content.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_content = '\n'.join(lines)
        
        # Try to find JSON object in the content
        json_start = cleaned_content.find('{')
        json_end = cleaned_content.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = cleaned_content[json_start:json_end]
            parsed_json = json.loads(json_str)
            
            # Extract job_analysis
            if 'job_analysis' in parsed_json:
                job_analysis = parsed_json['job_analysis']
                # Format job analysis as readable text
                job_analysis_parts = []
                if 'essential_requirements' in job_analysis:
                    reqs = job_analysis['essential_requirements']
                    if isinstance(reqs, list):
                        job_analysis_parts.append("**Essential Requirements:**")
                        for req in reqs:
                            job_analysis_parts.append(f"- {req}")
                    else:
                        job_analysis_parts.append(f"**Essential Requirements:** {reqs}")
                
                if 'key_responsibilities' in job_analysis:
                    job_analysis_parts.append(f"**Key Responsibilities:** {job_analysis['key_responsibilities']}")
                
                if 'required_skills' in job_analysis:
                    skills = job_analysis['required_skills']
                    if isinstance(skills, list):
                        job_analysis_parts.append(f"**Required Skills:** {', '.join(skills)}")
                    else:
                        job_analysis_parts.append(f"**Required Skills:** {skills}")
                
                if 'preferred_skills' in job_analysis and job_analysis['preferred_skills']:
                    pref_skills = job_analysis['preferred_skills']
                    if isinstance(pref_skills, list):
                        job_analysis_parts.append(f"**Preferred Skills:** {', '.join(pref_skills)}")
                    else:
                        job_analysis_parts.append(f"**Preferred Skills:** {pref_skills}")
                
                if 'experience_level' in job_analysis:
                    job_analysis_parts.append(f"**Experience Level:** {job_analysis['experience_level']}")
                
                if 'key_insights' in job_analysis:
                    insights = job_analysis['key_insights']
                    if isinstance(insights, list):
                        job_analysis_parts.append("**Key Insights:**")
                        for insight in insights:
                            job_analysis_parts.append(f"- {insight}")
                    else:
                        job_analysis_parts.append(f"**Key Insights:** {insights}")
                
                result['job_analysis'] = '\n\n'.join(job_analysis_parts)
            
            # Extract match_evaluation
            if 'match_evaluation' in parsed_json:
                match_eval = parsed_json['match_evaluation']
                result['match'] = match_eval.get('match', None)
                result['reasoning'] = match_eval.get('reasoning', None)
                result['recommendations'] = match_eval.get('recommendations', None)
                result['linkedin_keywords'] = match_eval.get('linkedin_keywords', None)
            
            return result
            
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # JSON parsing failed, fall back to text parsing
        pass
    
    # Fallback to text parsing
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    # Find where job analysis ends and match evaluation begins
    job_analysis_lines = []
    match_eval_start = -1
    
    for i, line in enumerate(lines):
        if line.upper().startswith('MATCH EVALUATION:'):
            match_eval_start = i
            break
        job_analysis_lines.append(line)
    
    # Extract job analysis
    if job_analysis_lines:
        # Remove "JOB ANALYSIS:" header if present
        if job_analysis_lines[0].upper().startswith('JOB ANALYSIS:'):
            job_analysis_lines = job_analysis_lines[1:]
        result['job_analysis'] = '\n'.join(job_analysis_lines)
    
    # Parse match evaluation if present
    if match_eval_start >= 0 and match_eval_start + 1 < len(lines):
        match_lines = lines[match_eval_start + 1:]  # Skip "MATCH EVALUATION:" line
        
        if match_lines:
            # Handle case where "Yes" or "No" might be on the same line as "MATCH EVALUATION:"
            first_line = match_lines[0].upper() if match_lines else ""
            
            # Check if first line contains "Yes" or "No"
            if 'YES' in first_line:
                result['match'] = True
                # Extract reasoning from the same line or next line
                if ':' in match_lines[0]:
                    reasoning_part = match_lines[0].split(':', 1)[1].strip()
                    if reasoning_part and not reasoning_part.upper().startswith('YES'):
                        result['reasoning'] = reasoning_part
                elif len(match_lines) > 1:
                    result['reasoning'] = match_lines[1]
            elif 'NO' in first_line:
                result['match'] = False
                # Extract reasoning
                if ':' in match_lines[0]:
                    reasoning_part = match_lines[0].split(':', 1)[1].strip()
                    if reasoning_part and not reasoning_part.upper().startswith('NO'):
                        result['reasoning'] = reasoning_part
                elif len(match_lines) > 1:
                    result['reasoning'] = match_lines[1]
            
            # Parse recommendations and keywords if No
            if result['match'] is False and len(match_lines) > 1:
                recommendations = []
                linkedin_keywords = None
                in_recommendations = False
                in_keywords = False
                
                for i, line in enumerate(match_lines[1:], start=1):
                    line_upper = line.upper()
                    
                    if line_upper.startswith('RECOMMENDATIONS:'):
                        in_recommendations = True
                        in_keywords = False
                        if ':' in line and len(line.split(':', 1)) > 1:
                            rec_part = line.split(':', 1)[1].strip()
                            if rec_part.startswith('-'):
                                recommendations.append(rec_part[1:].strip())
                        continue
                    
                    elif line_upper.startswith('LINKEDIN_KEYWORDS:'):
                        in_recommendations = False
                        in_keywords = True
                        if ':' in line:
                            keywords_str = line.split(':', 1)[1].strip()
                            if keywords_str:
                                linkedin_keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                        if not linkedin_keywords and i + 1 < len(match_lines):
                            next_line = match_lines[i + 1]
                            if not next_line.upper().startswith('RECOMMENDATIONS:'):
                                linkedin_keywords = [k.strip() for k in next_line.split(',') if k.strip()]
                        break
                    
                    elif in_recommendations:
                        if line.startswith('-'):
                            recommendations.append(line[1:].strip())
                        elif line and not line_upper.startswith('LINKEDIN_KEYWORDS:'):
                            if len(recommendations) < 5:
                                recommendations.append(line)
                    
                    elif in_keywords and not linkedin_keywords:
                        if ',' in line or ' ' in line:
                            linkedin_keywords = [k.strip() for k in line.split(',') if k.strip()]
                
                # Clean up
                recommendations = [r for r in recommendations if r and len(r.strip()) > 0]
                result['recommendations'] = recommendations if recommendations else None
                
                if linkedin_keywords:
                    linkedin_keywords = [k for k in linkedin_keywords if k and len(k.strip()) > 0]
                    result['linkedin_keywords'] = linkedin_keywords if linkedin_keywords else None
    
    return result

# Helper functions for text processing
def clean_text(text: str) -> str:
    """
    Cleans text (works for both job descriptions and resumes):
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
    text = re.sub(r"(linkedin|glassdoor|indeed|monster|career|company|github)\S*",
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

# Token counting and truncation function
@st.cache_resource
def _get_tokenizer():
    """Get tokenizer for counting tokens - cached"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        # Use a common tokenizer that works well for most models
        # GPT-2 tokenizer is fast and widely compatible
        return AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        return None

def count_tokens(text: str) -> int:
    """Count tokens in text using tokenizer if available, otherwise use approximation"""
    if not text:
        return 0
    
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception:
            pass
    
    # Fallback: approximate token count (roughly 4 characters per token for English)
    return len(text) // 4

def truncate_by_tokens(text: str, max_tokens: int, suffix: str = "...") -> str:
    """Truncate text to fit within max_tokens, preserving word boundaries when possible"""
    if not text:
        return text
    
    tokenizer = _get_tokenizer()
    
    if tokenizer is not None:
        try:
            # Encode the text
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # If text fits, return as is
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate tokens
            truncated_tokens = tokens[:max_tokens]
            
            # Decode back to text
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            # Add suffix if text was truncated
            if len(tokens) > max_tokens:
                return truncated_text + suffix
            
            return truncated_text
        except Exception:
            pass
    
    # Fallback: character-based truncation (approximate)
    # Roughly 4 characters per token
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    # Try to cut at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:  # If we can find a space reasonably close
        truncated = truncated[:last_space]
    
    return truncated + suffix

# Function to compare resume and job description with LLM
def compare_resume_job_with_llm(resume_text: str, job: Dict) -> Dict[str, any]:  # type: ignore
    """Compare resume with job description using LLM and provide recommendations
    
    Returns a dictionary with:
    - llm_match: True/False/None
    - llm_reasoning: Explanation string
    - llm_recommendations: List of recommendations (if No)
    - linkedin_keywords: List of keywords (if No)
    - llm_error: Error message if any
    """
    if not OLLAMA_AVAILABLE:
        return {
            'llm_match': None,
            'llm_reasoning': None,
            'llm_recommendations': None,
            'linkedin_keywords': None,
            'llm_error': 'Ollama not available'
        }
    
    client = _get_ollama_client()
    if client is None:
        return {
            'llm_match': None,
            'llm_reasoning': None,
            'llm_recommendations': None,
            'linkedin_keywords': None,
            'llm_error': 'Could not connect to Ollama'
        }
    
    model_name = _get_ollama_model()
    
    # Get job description
    job_description = job.get('text', '')
    
    # Clean text for better readability
    cleaned_job_text = clean_text(job_description)
    cleaned_resume_text = clean_text(resume_text)
    
    system_prompt = (
        "You are an expert recruiter evaluating resume-job matches. "
        "Job descriptions often list ideal candidates with all possible skills - this is unrealistic. "
        "Focus on ESSENTIAL qualifications needed to perform core job functions. "
        "Consider transferable skills, related experience, and learning ability. "
        "A 'Yes' means the candidate can reasonably do the job, not that they have every skill listed."
    )
    
    # Token budget allocation (max 5000 tokens total for input)
    MAX_INPUT_TOKENS = 5000
    system_tokens = count_tokens(system_prompt)
    prompt_template = """Evaluate if this resume matches this job posting. Be practical and realistic.

JOB POSTING:
Title: {title}
Company: {company}
Description: {job_text}

RESUME:
{resume_text}

EVALUATION APPROACH:
1. Identify 3-5 CORE skills/requirements essential for this role (ignore nice-to-haves)
2. Check if candidate has these core skills OR transferable/equivalent experience
3. Assess if candidate's experience level aligns with role expectations
4. Consider: Can they reasonably perform the core job functions? If yes → "Yes"

DECISION RULES:
- Answer "Yes" if: Candidate has essential qualifications OR strong transferable skills that indicate they can learn/adapt
- Answer "No" only if: Missing critical core requirements that would prevent basic job performance

OUTPUT FORMAT:
Line 1: "Yes" or "No"
Line 2: One-sentence explanation
If "No", then:
Line 3: "RECOMMENDATIONS:" followed by 3-5 specific, actionable recommendations (one per line, each starting with "-")
Line 4: "LINKEDIN_KEYWORDS:" followed by 5-10 relevant job search keywords (comma-separated)

EXAMPLES:

Yes
The candidate has the essential technical skills and relevant experience level for this role, with transferable capabilities that indicate they can perform effectively.

No
The candidate lacks critical core requirements (e.g., [specific essential skill]) that are fundamental to performing this role.
RECOMMENDATIONS:
- Develop proficiency in [critical missing skill] through projects or training
- Highlight similar technologies or related experience that demonstrates capability
- Obtain certification in [core area] to strengthen qualifications
- Emphasize learning ability and adaptability from past role transitions
LINKEDIN_KEYWORDS:
[relevant keywords, comma-separated]"""
    
    template_tokens = count_tokens(prompt_template.format(title="", company="", job_text="", resume_text=""))
    job_title_tokens = count_tokens(job.get('title', 'N/A'))
    company_tokens = count_tokens(job.get('company', 'N/A'))
    
    # Calculate available tokens for job description and resume
    reserved_tokens = system_tokens + template_tokens + job_title_tokens + company_tokens + 100  # 100 buffer
    available_tokens = MAX_INPUT_TOKENS - reserved_tokens
    
    # Allocate 50/50 between job and resume, but ensure minimum of 500 tokens each
    tokens_per_text = max(500, available_tokens // 2)
    
    # Truncate texts based on token count
    job_text_preview = truncate_by_tokens(cleaned_job_text, tokens_per_text)
    resume_text_preview = truncate_by_tokens(cleaned_resume_text, tokens_per_text)
    
    # Format user prompt using template
    user_prompt = prompt_template.format(
        title=job.get('title', 'N/A'),
        company=job.get('company', 'N/A'),
        job_text=job_text_preview,
        resume_text=resume_text_preview
    ).strip()
    
    try:
        response = client.chat(  # type: ignore[attr-defined]
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        content = response["message"]["content"].strip()
        
        # Parse response
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Check if it's Yes or No
        is_match = None
        reasoning = None
        recommendations = None
        linkedin_keywords = None
        
        if lines:
            first_line = lines[0].upper()
            if first_line.startswith('YES'):
                is_match = True
            elif first_line.startswith('NO'):
                is_match = False
            
            # Get reasoning (second line)
            if len(lines) > 1:
                reasoning = lines[1]
            
            # Parse recommendations and keywords if answer is No
            if is_match is False:
                recommendations = []
                linkedin_keywords = None
                in_recommendations = False
                in_keywords = False
                
                for i, line in enumerate(lines[2:], start=2):
                    line_upper = line.upper()
                    
                    if line_upper.startswith('RECOMMENDATIONS:'):
                        in_recommendations = True
                        in_keywords = False
                        # Check if recommendations are on the same line
                        if ':' in line and len(line.split(':', 1)) > 1:
                            rec_part = line.split(':', 1)[1].strip()
                            if rec_part.startswith('-'):
                                recommendations.append(rec_part[1:].strip())
                        continue
                    
                    elif line_upper.startswith('LINKEDIN_KEYWORDS:'):
                        in_recommendations = False
                        in_keywords = True
                        # Extract keywords from this line
                        if ':' in line:
                            keywords_str = line.split(':', 1)[1].strip()
                            if keywords_str:
                                linkedin_keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                        # Check next line if keywords are on separate line
                        if not linkedin_keywords and i + 1 < len(lines):
                            next_line = lines[i + 1]
                            if not next_line.upper().startswith('RECOMMENDATIONS:'):
                                linkedin_keywords = [k.strip() for k in next_line.split(',') if k.strip()]
                        break
                    
                    elif in_recommendations:
                        # Collect recommendation lines (lines starting with -)
                        if line.startswith('-'):
                            recommendations.append(line[1:].strip())
                        elif line and not line_upper.startswith('LINKEDIN_KEYWORDS:'):
                            # Sometimes recommendations don't have dashes
                            if len(recommendations) < 5:  # Limit to reasonable number
                                recommendations.append(line)
                    
                    elif in_keywords and not linkedin_keywords:
                        # Keywords might be on next line
                        if ',' in line or ' ' in line:
                            linkedin_keywords = [k.strip() for k in line.split(',') if k.strip()]
                
                # Clean up recommendations (remove empty ones)
                recommendations = [r for r in recommendations if r and len(r.strip()) > 0]
                if not recommendations:
                    recommendations = None
                
                # Clean up keywords
                if linkedin_keywords:
                    linkedin_keywords = [k for k in linkedin_keywords if k and len(k.strip()) > 0]
                    if not linkedin_keywords:
                        linkedin_keywords = None
        
        return {
            'llm_match': is_match,
            'llm_reasoning': reasoning,
            'llm_recommendations': recommendations if recommendations else None,
            'linkedin_keywords': linkedin_keywords,
            'llm_error': None
        }
    except Exception as e:
        return {
            'llm_match': None,
            'llm_reasoning': None,
            'llm_recommendations': None,
            'linkedin_keywords': None,
            'llm_error': f'LLM evaluation error: {str(e)}'
        }

# Helper function to calculate matching skills consistently
def calculate_matching_skills(resume_skills: List[str], job_skills: List[str]) -> set:
    """
    Use linkedin-jobs-scraper to fetch a few live jobs based on a list of skills.
    Uses the same scraper pattern and filters as 0_Job_Crawling.py (lines 137–167),
    but with a dynamic query string and a small limit.
    """
    if not skills:
        msg = "No skills provided for LinkedIn scraping."
        return False, None, msg

    # Build a compact query string from the top skill (match linkedin.py behavior)
    query_string = 'Software Engineer'

    # If the user has already set a LinkedIn cookie on another page (e.g., Job Crawling),
    # propagate it to the environment variable expected by linkedin-jobs-scraper.
    li_at_cookie = st.session_state.get("li_at_cookie")
    if li_at_cookie:
        os.environ["LI_AT_COOKIE"] = li_at_cookie

    try:
        from linkedin_jobs_scraper import LinkedinScraper
        from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
        from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
        from linkedin_jobs_scraper.filters import (
            RelevanceFilters,
            TimeFilters,
            TypeFilters,
            ExperienceLevelFilters,
            OnSiteOrRemoteFilters,
            SalaryBaseFilters,
        )
    except ImportError:
        msg = "linkedin-jobs-scraper package is not installed. Please install it to enable live LinkedIn scraping."
        st.error(msg)
        st.session_state.live_linkedin_error = msg
        return False, None, msg

    jobs_data: List[Dict] = []

    # Fired once for each successfully processed job
    def on_data(data: EventData):
        print(f"[DEBUG] on_data called: {data.title} at {data.company}")
        jobs_data.append(
            {
                "Job Title": data.title,
                "Company": data.company,
                "Company Link": data.company_link,
                "Date": data.date,
                "Date Text": data.date_text,
                "Job Link": data.link,
                "Insights": ", ".join(data.insights) if data.insights else "",
                "Description Length": len(data.description) if data.description else 0,
                "Description": (data.description or "").replace("\n", " ").replace("\r", " "),
            }
        )

    # Fired once for each page (25 jobs) – we don’t need it in the UI
    def on_metrics(metrics: EventMetrics):
        print(f"[DEBUG] on_metrics called: {metrics}")

    def on_error(error):
        print(f"[DEBUG] on_error called: {error}")
        msg = f"Error scraping LinkedIn jobs: {error}"
        st.warning(msg)
        st.session_state.live_linkedin_error = str(error)

    def on_end():
        print(f"[DEBUG] on_end called. Total jobs collected: {len(jobs_data)}")

    # Detect if running in Docker (check for CHROME_BINARY env var set by docker-compose)
    # If in Docker, use explicit paths; otherwise use None (auto-detect) like linkedin.py
    chrome_binary = os.getenv("CHROME_BINARY")
    chromedriver_path = os.getenv("CHROMEDRIVER_PATH")
    
    if chrome_binary and chromedriver_path:
        # Running in Docker - use explicit paths
        print(f"[DEBUG] Using Docker Chrome paths: {chrome_binary}, {chromedriver_path}")
        scraper = LinkedinScraper(
            chrome_executable_path=chromedriver_path,
            chrome_binary_location=chrome_binary,
            chrome_options=None,
            headless=True,
            max_workers=1,
            slow_mo=2.0,
            page_load_timeout=40
        )
    else:
        # Running locally - use auto-detection like linkedin.py
        print(f"[DEBUG] Using auto-detected Chrome paths (None)")
        scraper = LinkedinScraper(
            chrome_executable_path=None,
            chrome_binary_location=None,
            chrome_options=None,
            headless=True,
            max_workers=1,
            slow_mo=2.0,
            page_load_timeout=40
        )

    # Add event listeners
    scraper.on(Events.DATA, on_data)
    scraper.on(Events.ERROR, on_error)
    scraper.on(Events.END, on_end)

    # Create query with company filter (aligned with linkedin.py)
    # Use the same company_jobs_url format as linkedin.py which includes currentJobId and origin parameters
    queries = [
        # Warm-up query mirroring linkedin.py
        Query(
            options=QueryOptions(
                limit=1
            )
        ),
        Query(
            query=query_string,
            options=QueryOptions(
                locations=["United States"],
                apply_link=True,  # Try to extract apply link
                skip_promoted_jobs=True,  # Skip promoted jobs
                page_offset=0,  # How many pages to skip
                limit=max_jobs,
                filters=QueryFilters(
                    company_jobs_url="https://www.linkedin.com/jobs/search/?currentJobId=4297199509&f_C=11448%2C1035%2C1418841%2C10073178%2C11206713%2C1148098%2C1386954%2C165397%2C18086638%2C1889423%2C19053704%2C19537%2C2270931%2C2446424%2C263515%2C30203%2C3178875%2C3238203%2C3290211%2C3641570%2C3763403%2C5097047%2C5607466%2C589037%2C692068&geoId=92000000&origin=JOB_SEARCH_PAGE_JOB_FILTER",  # Filter by companies (matching linkedin.py)
                    relevance=RelevanceFilters.RECENT,
                    time=TimeFilters.WEEK,
                    type=[TypeFilters.FULL_TIME, TypeFilters.INTERNSHIP],
                    on_site_or_remote=[OnSiteOrRemoteFilters.REMOTE],
                    experience=[ExperienceLevelFilters.MID_SENIOR],
                    base_salary=SalaryBaseFilters.SALARY_100K
                ),
            ),
        )
    ]

    # Run scraper synchronously (Streamlit will wait here)
    print(f"[DEBUG] Starting scraper with query: '{query_string}'")
    try:
        scraper.run(queries)
    except Exception as e:
        msg = f"Failed to scrape LinkedIn jobs: {e}"
        st.error(msg)
        st.session_state.live_linkedin_error = str(e)
        return False, None, msg

    # Relax filters and retry if no jobs found on the first pass
    if not jobs_data:
        st.info(f"No jobs found with strict filters. Trying relaxed filters for: '{query_string}'")
        # Create a new scraper instance for the fallback attempt (use same Chrome detection logic)
        if chrome_binary and chromedriver_path:
            scraper_fallback = LinkedinScraper(
                chrome_executable_path=chromedriver_path,
                chrome_binary_location=chrome_binary,
                chrome_options=None,
                headless=True,
                max_workers=1,
                slow_mo=2.0,
                page_load_timeout=40
            )
        else:
            scraper_fallback = LinkedinScraper(
                chrome_executable_path=None,
                chrome_binary_location=None,
                chrome_options=None,
                headless=True,
                max_workers=1,
                slow_mo=2.0,
                page_load_timeout=40
            )
        
        # Re-attach event listeners to the new scraper
        scraper_fallback.on(Events.DATA, on_data)
        scraper_fallback.on(Events.ERROR, on_error)
        scraper_fallback.on(Events.END, on_end)
        
        fallback_queries = [
            Query(
                query=query_string,
                options=QueryOptions(
                    locations=["United States"],
                    apply_link=True,
                    skip_promoted_jobs=True,
                    page_offset=0,
                    limit=max_jobs,
                    filters=QueryFilters(
                        relevance=RelevanceFilters.RECENT,
                        time=TimeFilters.WEEK,
                        type=[TypeFilters.FULL_TIME, TypeFilters.INTERNSHIP],
                    ),
                ),
            )
        ]
        try:
            scraper_fallback.run(fallback_queries)
        except Exception as e:
            msg = f"Failed to scrape LinkedIn jobs (fallback): {e}"
            st.error(msg)
            st.session_state.live_linkedin_error = str(e)
            return False, None, msg

    if not jobs_data:
        msg = f"LinkedIn scraping completed, but no jobs were found for query: '{query_string}'. Try adjusting filters or using different keywords."
        return True, None, msg

    df = pd.DataFrame(jobs_data)
    msg = f"LinkedIn scraping completed successfully. Retrieved {len(df)} job(s) for query: '{query_string}'."
    return True, df, msg

# Helper function to calculate matching skills consistently
def calculate_matching_skills(resume_skills: List[str], job_skills: List[str]) -> set:
    """
    Calculate matching skills between resume and job, ensuring consistent normalization.
    Normalizes skills to lowercase and removes duplicates before comparison.
    """
    # Normalize skills: convert to lowercase and remove duplicates
    resume_skills_normalized = {skill.lower().strip() for skill in resume_skills if skill}
    job_skills_normalized = {skill.lower().strip() for skill in job_skills if skill}
    # Return intersection
    return resume_skills_normalized & job_skills_normalized

# Helper function to load the specific LSA 100 topics model
@st.cache_resource
def get_lsa_100_topics_model():
    """Load the saved LSA 100 topics model - cached"""
    try:
        import joblib
        workspace_path = st.session_state.get('workspace_path')
        if workspace_path:
            models_dir = os.path.join(workspace_path, "models")
        else:
            models_dir = "models"
        
        model_filename = "topic_model_lsa_100topics.joblib"
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
                st.warning(f"Error loading LSA 100 topics model: {e}")
                return None
        else:
            st.warning(f"LSA 100 topics model not found at {model_path}")
            return None
    except ImportError:
        st.warning("joblib not available for loading topic model")
        return None
    except Exception:
        return None

# Helper function to compute topic similarity using LDA/LSA (kept for backward compatibility)
@st.cache_resource
def get_topic_model(_method='LDA', _n_topics=10):
    """Load topic model (LDA or LSA) - cached"""
    try:
        return load_trained_topic_model(method=_method, n_topics=_n_topics)
    except Exception:
        return None

def compute_topic_score(resume_text: str, job_text: str, topic_model_data=None) -> float:
    """
    Compute topic similarity score between resume and job using LSA 100 topics model.
    Falls back to 0.0 if topic model is not available.
    
    Args:
        resume_text: Resume text
        job_text: Job description text
        topic_model_data: Pre-loaded topic model data (optional)
    
    Returns:
        Topic similarity score (0.0 to 1.0)
    """
    if topic_model_data is None:
        # Load the specific LSA 100 topics model
        topic_model_data = get_lsa_100_topics_model()
    
    if topic_model_data is None:
        # No topic model available, return 0.0 (will use semantic as fallback)
        return 0.0
    
    try:
        # Get topic distributions for both texts
        resume_topics = get_document_topics(resume_text, topic_model_data)
        job_topics = get_document_topics(job_text, topic_model_data)
        
        if resume_topics is None or job_topics is None:
            return 0.0
        
        # Compute cosine similarity between topic distributions
        topic_score = compute_topic_similarity(resume_topics, job_topics)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, float(topic_score)))
    except Exception as e:
        # If any error occurs, return 0.0
        return 0.0

# Check if services are available
if not LOCAL_MODELS_AVAILABLE:
    st.error("⚠️ Local NLP models not available. Please install required packages: pip install sentence-transformers spacy && python -m spacy download en_core_web_sm")
if not SPACY_AVAILABLE:
    st.warning("⚠️ spaCy not available. Skills-based matching will be disabled. Install: pip install spacy && python -m spacy download en_core_web_sm")


# Check MPS availability
if LOCAL_MODELS_AVAILABLE:
    try:
        # More robust MPS detection
        mps_available = False
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_built():
                try:
                    # Try to create MPS device
                    device = torch.device('mps')
                    # Test with a small tensor
                    test_tensor = torch.randn(1).to(device)
                    mps_available = True
                except:
                    mps_available = False
        

    except Exception as e:
        st.info(f"Debug: Error checking MPS: {e}")


st.markdown("---")

# Function to extract text from PDF
def extract_text_from_pdf(file) -> Optional[str]:
    """Extract text from PDF file"""
    if not PYPDF_AVAILABLE:
        st.error("PyPDF not available. Please install pypdf: pip install pypdf")
        return None

    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Function to extract text from TXT
def extract_text_from_txt(file) -> Optional[str]:
    """Extract text from TXT file"""
    try:
        content = file.read().decode("utf-8")
        return content.strip()
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return None

# Function to process resume and find matches
@st.cache_data
def load_and_prepare_job_data():
    """Load and prepare job data for matching (cached)"""
    try:
        import json
        workspace_path = st.session_state.get('workspace_path', '/workspace')
        json_path = os.path.join(workspace_path, "Data", "combined_data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                job_data = json.load(f)
            jobs_df = pd.DataFrame(job_data)
        else:
            st.error("Job data not found. Please ensure combined_data.json exists in workspace/Data/")
            return None, None
    except Exception as e:
        st.error(f"Error loading job data: {e}")
        return None, None

    # Filter out jobs without text
    valid_jobs = jobs_df[jobs_df['Description'].notna() & (jobs_df['Description'] != '')].copy()
    if len(valid_jobs) == 0:
        st.error("No valid job data found")
        return None, None

    job_texts = valid_jobs['Description'].tolist()
    return job_texts, valid_jobs

def process_resume_and_match(resume_text: str, top_k: int = 10) -> Optional[Dict[str, List[Dict]]]:
    """
    Process resume and find matches using both SBERT and Word2Vec.
    Returns a dictionary with 'sbert' and 'word2vec' keys, each containing top_k matches.
    """
    # Load job data (cached)
    job_texts, valid_jobs = load_and_prepare_job_data()

    if job_texts is None or valid_jobs is None:
        return None

    results = {'sbert': [], 'word2vec': []}
    
    # Extract resume skills once (used for both methods)
    resume_skills = extract_skills_keywords(resume_text, MASTER_SKILL_LIST)
    topic_model_data = get_lsa_100_topics_model()
    
    # Process SBERT matches
    with st.spinner("Finding matches using SBERT vector search..."):
        resume_sbert_emb = generate_local_embedding(resume_text, method="sbert")
        
        if resume_sbert_emb is not None and DATABASE_AVAILABLE:
            try:
                matching_results = find_similar_jobs_vector(
                    resume_sbert_emb.tolist(), 
                    embedding_type='sbert', 
                    top_k=top_k
                )
                
                if matching_results:
                    enhanced_matches = []
                    for job in matching_results:
                        job_text = job.get('text', '')
                        job_skills = extract_skills_keywords(job_text, MASTER_SKILL_LIST)
                        
                        skill_score = skill_jaccard_score(resume_skills, job_skills)
                        semantic_score = job['similarity']
                        
                        topic_score = compute_topic_score(resume_text, job_text, topic_model_data)
                        if topic_score == 0.0:
                            topic_score = semantic_score
                        
                        avg_topic_semantic = (topic_score + semantic_score) / 2
                        final_score = avg_topic_semantic + (1 - avg_topic_semantic) * skill_score
                        
                        enhanced_job = job.copy()
                        enhanced_job.update({
                            'skill_score': skill_score,
                            'semantic_score': semantic_score,
                            'topic_score': topic_score,
                            'final_score': final_score,
                            'resume_skills': resume_skills,
                            'job_skills': job_skills,
                            'method': 'sbert'
                        })
                        enhanced_matches.append(enhanced_job)
                    
                    enhanced_matches.sort(key=lambda x: x['final_score'], reverse=True)
                    results['sbert'] = enhanced_matches[:top_k]
                else:
                    st.warning("No SBERT matches found in database.")
            except Exception as e:
                st.warning(f"SBERT database search failed: {e}")
    
    # Process Word2Vec matches
    with st.spinner("Finding matches using Word2Vec vector search..."):
        w2v_model = load_trained_word2vec_model()
        if w2v_model:
            resume_tokens = simple_tokenize(resume_text)
            resume_w2v_emb = get_doc_embedding_w2v(resume_tokens, w2v_model)
            
            if resume_w2v_emb is None:
                st.warning("Failed to generate Word2Vec embedding for resume. Check if resume text contains valid tokens.")
            elif DATABASE_AVAILABLE:
                try:
                    # Check if database has Word2Vec embeddings
                    engine = create_db_engine()
                    if engine:
                        with engine.connect() as conn:
                            check_query = text("SELECT COUNT(*) FROM jobs WHERE word2vec_embedding IS NOT NULL")
                            result = conn.execute(check_query)
                            count = result.scalar()
                            if count == 0:
                                st.warning(f"No Word2Vec embeddings found in database. Please import Word2Vec embeddings first. Found {count} jobs with Word2Vec embeddings.")
                            else:
                                matching_results = find_similar_jobs_vector(
                                    resume_w2v_emb.tolist(), 
                                    embedding_type='word2vec', 
                                    top_k=top_k
                                )
                                
                                if matching_results:
                                    enhanced_matches = []
                                    for job in matching_results:
                                        job_text = job.get('text', '')
                                        job_skills = extract_skills_keywords(job_text, MASTER_SKILL_LIST)
                                        
                                        skill_score = skill_jaccard_score(resume_skills, job_skills)
                                        semantic_score = job['similarity']  # This is Word2Vec similarity
                                        
                                        topic_score = compute_topic_score(resume_text, job_text, topic_model_data)
                                        if topic_score == 0.0:
                                            topic_score = semantic_score
                                        
                                        avg_topic_semantic = (topic_score + semantic_score) / 2
                                        final_score = avg_topic_semantic + (1 - avg_topic_semantic) * skill_score
                                        
                                        enhanced_job = job.copy()
                                        enhanced_job.update({
                                            'skill_score': skill_score,
                                            'semantic_score': semantic_score,
                                            'topic_score': topic_score,
                                            'final_score': final_score,
                                            'resume_skills': resume_skills,
                                            'job_skills': job_skills,
                                            'method': 'word2vec'
                                        })
                                        enhanced_matches.append(enhanced_job)
                                    
                                    enhanced_matches.sort(key=lambda x: x['final_score'], reverse=True)
                                    results['word2vec'] = enhanced_matches[:top_k]
                                else:
                                    st.warning(f"No Word2Vec matches found in database. Database has {count} jobs with Word2Vec embeddings.")
                except Exception as e:
                    import traceback
                    st.error(f"Word2Vec database search failed: {e}")
                    if st.session_state.get('debug_mode', False):
                        st.code(traceback.format_exc())
            else:
                st.warning("Database not available for Word2Vec search.")
        else:
            st.warning("Word2Vec model not available. Please load the Word2Vec model first.")
    
    return results

# Main content
st.markdown("""
Upload your resume and find the most relevant job opportunities using **two different embedding methods**:

**🔷 SBERT**: Uses transformer-based embeddings for semantic understanding \n
**🔶 Word2Vec**: Uses word-level embeddings trained on job descriptions

Both methods use **combined scoring** that balances:
- **Skills Match (Keyword-based)**: Technical and soft skills alignment using keyword matching
- **Semantic Similarity**: Contextual meaning using embeddings (SBERT or Word2Vec)
- **Topic Relevance**: Thematic alignment using LSA 100 topics model
- **Final Score**: Average(topic, semantic) + (1 - average) × Skill Score

You'll see the **top 10 matches** for each method in separate sections, allowing you to compare results from both approaches.
""")

# File upload section
st.markdown("### Upload Your Resume")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=['pdf', 'txt'],
        help="Upload your resume as PDF or TXT file"
    )

with col2:
    top_k = st.slider(
        "Number of matches to show",
        min_value=5,
        max_value=20,
        value=10,
        help="How many top matching jobs to display"
    )

# Alternative text input
st.markdown("### ✏️ Or Paste Resume Text")
resume_text_input = st.text_area(
    "Paste your resume text here",
    height=200,
    placeholder="Copy and paste your resume content here if you don't have a file...",
    help="Alternatively, paste your resume text directly"
)

# Process resume
if uploaded_file or resume_text_input:
    resume_text = None

    if uploaded_file:
        # Process uploaded file
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif file_type == "text/plain":
            resume_text = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload PDF or TXT files.")

        if resume_text:
            st.success(f"✅ Resume extracted from {uploaded_file.name} ({len(resume_text)} characters)")

    elif resume_text_input:
        resume_text = resume_text_input.strip()
        if resume_text:
            st.success(f"✅ Resume text loaded ({len(resume_text)} characters)")

    # Store resume text
    if resume_text:
        st.session_state.resume_text = resume_text

        # Show resume preview
        with st.expander("📄 Resume Preview"):
            st.text_area(
                "Resume Content",
                value=resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""),
                height=300,
                disabled=True
            )

        # Show resume analysis
        with st.expander("🔍 Resume Analysis"):
            # Extract and display skills using simple keyword matching (no NER)
            resume_skills = extract_skills_keywords(resume_text, MASTER_SKILL_LIST)
            
            st.markdown("**Extracted Skills (Keyword-based):**")
            if resume_skills:
                st.write(f"Found {len(resume_skills)} skills:")
                st.write(", ".join(resume_skills))
            else:
                st.write("No skills extracted.")
                       
            # SBERT
            resume_sbert_emb = generate_local_embedding(resume_text, method="sbert")
            if resume_sbert_emb is not None:
                # Store embedding for later technical details section
                st.session_state.resume_embedding = resume_sbert_emb
                st.write("**SBERT Embedding:**")
                st.write(f"- Dimensions: {len(resume_sbert_emb)}")
                st.write(f"- Range: {min(resume_sbert_emb):.4f} to {max(resume_sbert_emb):.4f}")
                st.write(f"- Mean: {np.mean(resume_sbert_emb):.4f}")
                st.write(f"- Std: {np.std(resume_sbert_emb):.4f}")
            else:
                st.write("SBERT embedding not available.")
            
            # Word2Vec
            w2v_model = load_trained_word2vec_model()
            if w2v_model:
                resume_tokens = simple_tokenize(resume_text)
                resume_w2v_emb = get_doc_embedding_w2v(resume_tokens, w2v_model)
                if resume_w2v_emb is not None:
                    st.write("**Word2Vec Embedding:**")
                    st.write(f"- Dimensions: {len(resume_w2v_emb)}")
                    st.write(f"- Range: {min(resume_w2v_emb):.4f} to {max(resume_w2v_emb):.4f}")
                    st.write(f"- Mean: {np.mean(resume_w2v_emb):.4f}")
                    st.write(f"- Std: {np.std(resume_w2v_emb):.4f}")
                else:
                    st.write("Word2Vec embedding not available.")
            else:
                st.write("Word2Vec model not loaded.")
            
            # Topics
            st.markdown("**Topic Analysis:**")
            topic_model = get_lsa_100_topics_model()
            if topic_model:
                st.write(f"Topic Model: LSA with 100 topics")
            else:
                st.write("LSA 100 topics model not available. Topic score will use semantic similarity as fallback.")

        # Find matches button
        if st.button("🔍 Find Matching Jobs", type="primary", use_container_width=True):
            matching_results = process_resume_and_match(resume_text, top_k)

            if matching_results:
                st.session_state.matching_results = matching_results
                total_matches = len(matching_results.get('sbert', [])) + len(matching_results.get('word2vec', []))
                st.success(f"✅ Found {total_matches} matching jobs!")
                st.rerun()

# Display results
if st.session_state.get("matching_results"):
    st.markdown("---")
    results = st.session_state.matching_results
    
    # Check if results is the new format (dict with 'sbert' and 'word2vec' keys) or old format (list)
    if isinstance(results, dict) and 'sbert' in results and 'word2vec' in results:
        # New format: Display separate sections for SBERT and Word2Vec
        
        # SBERT Section
        st.markdown("### 🔷 Top 10 Matching Jobs - SBERT (Sentence-BERT)")
        sbert_results = results.get('sbert', [])
        
        if sbert_results:
            # Summary metrics for SBERT
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jobs Found", len(sbert_results))
            with col2:
                avg_final = np.mean([job['final_score'] for job in sbert_results])
                st.metric("Avg Final Score", f"{avg_final:.3f}")
            with col3:
                max_final = max(job['final_score'] for job in sbert_results)
                st.metric("Best Match", f"{max_final:.3f}")
            
            # Display SBERT job matches
            for i, job in enumerate(sbert_results, 1):
                final_score_percent = job['final_score'] * 100
                
                if final_score_percent >= 75:
                    color = "🟢"
                elif final_score_percent >= 60:
                    color = "🟡"
                else:
                    color = "🟠"
                
                with st.expander(f"{color} SBERT Match #{i}: {job.get('title', 'N/A')} at {job.get('company', 'N/A')} - {final_score_percent:.1f}% Match"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Job Title:** {job.get('title', 'N/A')}")
                        st.markdown(f"**Company:** {job.get('company', 'N/A')}")
                        st.markdown(f"**Final Score:** {final_score_percent:.1f}%")
                        
                        skill_pct = job['skill_score'] * 100
                        semantic_pct = job['semantic_score'] * 100
                        topic_pct = job['topic_score'] * 100
                        st.markdown(f"**Skill Score:** {skill_pct:.1f}% | **Semantic Score (SBERT):** {semantic_pct:.1f}% | **Topic Score:** {topic_pct:.1f}%")
                    
                    with col2:
                        st.metric("Match Strength", f"{final_score_percent:.1f}%")
                    
                    job_text = job.get('text', '')
                    if len(job_text) > 1:
                        with st.expander("Job Description"):
                            st.write(job_text)
                    else:
                        st.write(job_text)
                    
                    if job.get('resume_skills') and job.get('job_skills'):
                        with st.expander("Skills Analysis"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Resume Skills:**")
                                st.write(", ".join(job['resume_skills'][:10]))
                                if len(job['resume_skills']) > 10:
                                    st.write(f"... and {len(job['resume_skills']) - 10} more")
                            
                            with col2:
                                st.markdown("**Job Skills:**")
                                st.write(", ".join(job['job_skills'][:10]))
                                if len(job['job_skills']) > 10:
                                    st.write(f"... and {len(job['job_skills']) - 10} more")
                            
                            overlap = calculate_matching_skills(
                                job.get('resume_skills', []), 
                                job.get('job_skills', [])
                            )
                            if overlap:
                                st.markdown("**Matching Skills:**")
                                st.write(", ".join(sorted(overlap)))
                    
                    # LLM Analysis Section with Button (includes resume comparison if available)
                    st.markdown("---")
                    st.markdown("#### LLM Analysis & Recommendations")
                    
                    # Check if resume is available
                    resume_text = st.session_state.get('resume_text')
                    
                    # Create a unique key for each job's LLM analysis
                    llm_key = f"llm_analysis_{job.get('id', i)}"
                    
                    # Check if analysis already exists in session state
                    if llm_key not in st.session_state:
                        st.session_state[llm_key] = None
                    
                    # Button text based on whether resume is available
                    button_text = "Get LLM Analysis & Recommendations" if resume_text else "Get LLM Analysis"
                    
                    # Button to trigger LLM analysis
                    if st.button(button_text, key=f"llm_btn_{job.get('id', i)}"):
                        with st.spinner("Analyzing job description with LLM..." + (" and comparing with resume..." if resume_text else "")):
                            job_title = job.get('title', 'N/A')
                            company = job.get('company', 'N/A')
                            llm_result = analyze_job_description_with_llm(job_text, job_title, company, resume_text)
                            st.session_state[llm_key] = llm_result
                            st.rerun()
                    
                    # Display LLM analysis if available
                    if st.session_state[llm_key] is not None:
                        result = st.session_state[llm_key]
                        
                        # Check if result is a dict (with resume) or string (job only)
                        if isinstance(result, dict):
                            # Combined result with resume comparison
                            if result.get('error'):
                                st.error(f"❌ {result['error']}")
                            else:
                                # Display job analysis
                                if result.get('job_analysis'):
                                    st.markdown("**Job Analysis:**")
                                    st.write(result['job_analysis'])
                                
                                # Display match evaluation if available
                                if result.get('match') is not None:
                                    st.markdown("---")
                                    st.markdown("**Resume-Job Match Evaluation:**")
                                    
                                    match = result.get('match')
                                    if match is True:
                                        st.success("✅ **Match: Yes** - This resume appears to be a good fit for this job.")
                                    elif match is False:
                                        st.warning("⚠️ **Match: No** - This resume may need improvements for this job.")
                                    
                                    # Display reasoning
                                    if result.get('reasoning'):
                                        st.markdown("**Reasoning:**")
                                        st.write(result['reasoning'])
                                    
                                    # Display recommendations if No
                                    if match is False and result.get('recommendations'):
                                        st.markdown("**Recommendations:**")
                                        for rec in result['recommendations']:
                                            st.write(f"- {rec}")
                                    
                                    # Display LinkedIn keywords if No
                                    if match is False and result.get('linkedin_keywords'):
                                        st.markdown("**LinkedIn Keywords:**")
                                        keywords_str = ", ".join(result['linkedin_keywords'])
                                        st.write(keywords_str)
                        else:
                            # Simple string result (job analysis only)
                            st.markdown("**Job Analysis:**")
                            st.write(result)
                            if resume_text:
                                st.info("ℹ️ Resume was uploaded but comparison was not included. Please click the button again to get recommendations.")
                    elif not OLLAMA_AVAILABLE:
                        st.info("ℹ️ LLM analysis requires Ollama. Install with: `pip install ollama`")
                    elif resume_text is None:
                        st.info("ℹ️ Upload a resume to get personalized recommendations along with job analysis.")
        else:
            st.info("No SBERT matches found.")
        
        st.markdown("---")
        
        # Word2Vec Section
        st.markdown("### 🔶 Top 10 Matching Jobs - Word2Vec")
        w2v_results = results.get('word2vec', [])
        
        if w2v_results:
            # Summary metrics for Word2Vec
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jobs Found", len(w2v_results))
            with col2:
                avg_final = np.mean([job['final_score'] for job in w2v_results])
                st.metric("Avg Final Score", f"{avg_final:.3f}")
            with col3:
                max_final = max(job['final_score'] for job in w2v_results)
                st.metric("Best Match", f"{max_final:.3f}")
            
            # Display Word2Vec job matches
            for i, job in enumerate(w2v_results, 1):
                final_score_percent = job['final_score'] * 100
                
                if final_score_percent >= 75:
                    color = "🟢"
                elif final_score_percent >= 60:
                    color = "🟡"
                else:
                    color = "🟠"
                
                with st.expander(f"{color} Word2Vec Match #{i}: {job.get('title', 'N/A')} at {job.get('company', 'N/A')} - {final_score_percent:.1f}% Match"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Job Title:** {job.get('title', 'N/A')}")
                        st.markdown(f"**Company:** {job.get('company', 'N/A')}")
                        st.markdown(f"**Final Score:** {final_score_percent:.1f}%")
                        
                        skill_pct = job['skill_score'] * 100
                        semantic_pct = job['semantic_score'] * 100
                        topic_pct = job['topic_score'] * 100
                        st.markdown(f"**Skill Score:** {skill_pct:.1f}% | **Semantic Score (Word2Vec):** {semantic_pct:.1f}% | **Topic Score:** {topic_pct:.1f}%")
                    
                    with col2:
                        st.metric("Match Strength", f"{final_score_percent:.1f}%")
                    
                    st.markdown("**Job Description:**")
                    job_text = job.get('text', '')
                    if len(job_text) > 1000:
                        st.write(job_text[:1000] + "...")
                        with st.expander("Read Full Description"):
                            st.write(job_text)
                    else:
                        st.write(job_text)
                    
                    if job.get('resume_skills') and job.get('job_skills'):
                        with st.expander("Skills Analysis"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Resume Skills:**")
                                st.write(", ".join(job['resume_skills'][:10]))
                                if len(job['resume_skills']) > 10:
                                    st.write(f"... and {len(job['resume_skills']) - 10} more")
                            
                            with col2:
                                st.markdown("**Job Skills:**")
                                st.write(", ".join(job['job_skills'][:10]))
                                if len(job['job_skills']) > 10:
                                    st.write(f"... and {len(job['job_skills']) - 10} more")
                            
                            overlap = calculate_matching_skills(
                                job.get('resume_skills', []), 
                                job.get('job_skills', [])
                            )
                            if overlap:
                                st.markdown("**Matching Skills:**")
                                st.write(", ".join(sorted(overlap)))
                    
                    # LLM Analysis Section with Button (includes resume comparison if available)
                    st.markdown("---")
                    st.markdown("#### LLM Analysis & Recommendations")
                    
                    # Check if resume is available
                    resume_text = st.session_state.get('resume_text')
                    
                    # Create a unique key for each job's LLM analysis (use 'w2v' prefix to avoid conflicts)
                    llm_key = f"llm_analysis_w2v_{job.get('id', i)}"
                    
                    # Check if analysis already exists in session state
                    if llm_key not in st.session_state:
                        st.session_state[llm_key] = None
                    
                    # Button text based on whether resume is available
                    button_text = "Get LLM Analysis & Recommendations" if resume_text else "Get LLM Analysis"
                    
                    # Button to trigger LLM analysis
                    if st.button(button_text, key=f"llm_btn_w2v_{job.get('id', i)}"):
                        with st.spinner("Analyzing job description with LLM..." + (" and comparing with resume..." if resume_text else "")):
                            job_title = job.get('title', 'N/A')
                            company = job.get('company', 'N/A')
                            llm_result = analyze_job_description_with_llm(job_text, job_title, company, resume_text)
                            st.session_state[llm_key] = llm_result
                            st.rerun()
                    
                    # Display LLM analysis if available
                    if st.session_state[llm_key] is not None:
                        result = st.session_state[llm_key]
                        
                        # Check if result is a dict (with resume) or string (job only)
                        if isinstance(result, dict):
                            # Combined result with resume comparison
                            if result.get('error'):
                                st.error(f"❌ {result['error']}")
                            else:
                                # Display job analysis
                                if result.get('job_analysis'):
                                    st.markdown("**Job Analysis:**")
                                    st.write(result['job_analysis'])
                                
                                # Display match evaluation if available
                                if result.get('match') is not None:
                                    st.markdown("---")
                                    st.markdown("**Resume-Job Match Evaluation:**")
                                    
                                    match = result.get('match')
                                    if match is True:
                                        st.success("✅ **Match: Yes** - This resume appears to be a good fit for this job.")
                                    elif match is False:
                                        st.warning("⚠️ **Match: No** - This resume may need improvements for this job.")
                                    
                                    # Display reasoning
                                    if result.get('reasoning'):
                                        st.markdown("**Reasoning:**")
                                        st.write(result['reasoning'])
                                    
                                    # Display recommendations if No
                                    if match is False and result.get('recommendations'):
                                        st.markdown("**Recommendations:**")
                                        for rec in result['recommendations']:
                                            st.write(f"- {rec}")
                                    
                                    # Display LinkedIn keywords if No
                                    if match is False and result.get('linkedin_keywords'):
                                        st.markdown("**LinkedIn Keywords:**")
                                        keywords_str = ", ".join(result['linkedin_keywords'])
                                        st.write(keywords_str)
                        else:
                            # Simple string result (job analysis only)
                            st.markdown("**Job Analysis:**")
                            st.write(result)
                            if resume_text:
                                st.info("ℹ️ Resume was uploaded but comparison was not included. Please click the button again to get recommendations.")
                    elif not OLLAMA_AVAILABLE:
                        st.info("ℹ️ LLM analysis requires Ollama. Install with: `pip install ollama`")
                    elif resume_text is None:
                        st.info("ℹ️ Upload a resume to get personalized recommendations along with job analysis.")
        else:
            st.info("No Word2Vec matches found.")
        
        # Export results (combine both methods)
        st.markdown("---")
        st.markdown("### Export Results")
        
        if st.button("Export to CSV"):
            export_data = []
            for method_name, method_results in [('SBERT', sbert_results), ('Word2Vec', w2v_results)]:
                for i, job in enumerate(method_results, 1):
                    export_data.append({
                        'Method': method_name,
                        'Rank': i,
                        'Job_ID': job['id'],
                        'Title': job.get('title', ''),
                        'Company': job.get('company', ''),
                        'Final_Score': job['final_score'],
                        'Final_Score_Percent': job['final_score'] * 100,
                        'Skill_Score': job['skill_score'],
                        'Semantic_Score': job['semantic_score'],
                        'Topic_Score': job['topic_score'],
                        'Resume_Skills_Count': len(job.get('resume_skills', [])),
                        'Job_Skills_Count': len(job.get('job_skills', [])),
                        'Matching_Skills_Count': len(calculate_matching_skills(
                            job.get('resume_skills', []), 
                            job.get('job_skills', [])
                        ))
                    })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="job_matches_sbert_w2v.csv",
                mime="text/csv",
                key="download_csv"
            )
    
    else:
        # Old format: Display as before (backward compatibility)
        st.markdown("### Top Matching Jobs")
        
        if isinstance(results, list):
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jobs Found", len(results))
            with col2:
                if results:
                    avg_final = np.mean([job['final_score'] for job in results])
                    st.metric("Avg Final Score", f"{avg_final:.3f}")
            with col3:
                if results:
                    max_final = max(job['final_score'] for job in results)
                    st.metric("Best Match", f"{max_final:.3f}")

            # Display job matches
            for i, job in enumerate(results, 1):
                final_score_percent = job['final_score'] * 100

                # Color coding based on final score
                if final_score_percent >= 75:
                    color = "🟢"  # High match
                elif final_score_percent >= 60:
                    color = "🟡"  # Medium match
                else:
                    color = "🟠"  # Lower match

                with st.expander(f"{color} Match #{i}: {job.get('title', 'N/A')} at {job.get('company', 'N/A')} - {final_score_percent:.1f}% Match"):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**Job Title:** {job.get('title', 'N/A')}")
                        st.markdown(f"**Company:** {job.get('company', 'N/A')}")
                        st.markdown(f"**Final Score:** {final_score_percent:.1f}%")
                        
                        # Show component scores
                        skill_pct = job['skill_score'] * 100
                        semantic_pct = job['semantic_score'] * 100
                        topic_pct = job['topic_score'] * 100
                        st.markdown(f"**Skill Score:** {skill_pct:.1f}% | **Semantic Score:** {semantic_pct:.1f}% | **Topic Score:** {topic_pct:.1f}%")

                    with col2:
                        # Final score gauge
                        st.metric("Match Strength", f"{final_score_percent:.1f}%")

                    st.markdown("**Job Description:**")
                    job_text = job.get('text', '')
                    if len(job_text) > 1000:
                        st.write(job_text[:1000] + "...")
                        with st.expander("Read Full Description"):
                            st.write(job_text)
                    else:
                        st.write(job_text)

                    # Show skills comparison
                    if job.get('resume_skills') and job.get('job_skills'):
                        with st.expander("Skills Analysis"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Resume Skills:**")
                                st.write(", ".join(job['resume_skills'][:10]))  # Show first 10
                                if len(job['resume_skills']) > 10:
                                    st.write(f"... and {len(job['resume_skills']) - 10} more")
                            
                            with col2:
                                st.markdown("**Job Skills:**")
                                st.write(", ".join(job['job_skills'][:10]))  # Show first 10
                                if len(job['job_skills']) > 10:
                                    st.write(f"... and {len(job['job_skills']) - 10} more")
                            
                            # Show overlapping skills (using consistent normalization)
                            overlap = calculate_matching_skills(
                                job.get('resume_skills', []), 
                                job.get('job_skills', [])
                            )
                            if overlap:
                                st.markdown("**Matching Skills:**")
                                st.write(", ".join(sorted(overlap)))
            
            # Export results (old format)
            st.markdown("---")
            st.markdown("### Export Results")
            
            if st.button("Export to CSV", key="export_csv_old"):
                export_df = pd.DataFrame([{
                    'Rank': i+1,
                    'Job_ID': job['id'],
                    'Title': job.get('title', ''),
                    'Company': job.get('company', ''),
                    'Final_Score': job['final_score'],
                    'Final_Score_Percent': job['final_score'] * 100,
                    'Skill_Score': job['skill_score'],
                    'Semantic_Score': job['semantic_score'],
                    'Topic_Score': job['topic_score'],
                    'Resume_Skills_Count': len(job.get('resume_skills', [])),
                    'Job_Skills_Count': len(job.get('job_skills', [])),
                    'Matching_Skills_Count': len(calculate_matching_skills(
                        job.get('resume_skills', []), 
                        job.get('job_skills', [])
                    ))
                } for i, job in enumerate(results)])

                csv = export_df.to_csv(index=False)

                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="job_matches_combined.csv",
                    mime="text/csv",
                    key="download_csv_old"
                )

# Footer
st.markdown("---")
st.markdown("""
### ℹ️ How It Works

1. **Upload Resume**: Upload your resume as PDF or TXT, or paste the text directly
2. **AI Processing**: Generate SBERT embeddings and extract skills using spaCy NER
3. **Multi-dimensional Matching**: Compute three similarity scores:
   - **Skill Score (Keyword-based)**: Jaccard similarity between resume and job skills
   - **Semantic Score**: Cosine similarity of SBERT embeddings
   - **Topic Score**: Cosine similarity of LSA 100 topics model distributions (falls back to semantic if model not available)
4. **Combined Scoring**: Final Score = Average(topic, semantic) + (1 - average) × Skill Score
5. **Results**: View ranked job matches with detailed component scores and skills analysis

### 💡 Tips

- **File Formats**: Both PDF and TXT files are supported
- **Text Quality**: Better formatted resumes produce more accurate matches
- **Combined Scoring**: The final score balances skills expertise, semantic relevance, and topical alignment
- **Skills Analysis**: Expand the skills section to see detailed resume-job skill matching
- **Performance**: SBERT embeddings provide high-quality semantic matching without API costs

For technical support or questions, please check the project documentation.
""")

# Add a clear results button
if st.session_state.matching_results:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.matching_results = None
            st.session_state.resume_text = None
            st.session_state.resume_embedding = None
            st.rerun()

