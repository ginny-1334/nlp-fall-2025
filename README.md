# nlp-fall-2025

## LinkedIn Job Analysis Platform

**Project Date:** Fall 2025

This project is an NLP-powered platform for analyzing LinkedIn job postings using advanced natural language processing techniques including Named Entity Recognition (NER), Topic Modeling, and Word Embeddings.

📄 **[Final Presentation PDF](Final_Presentation_NLP.pdf)** - Comprehensive project presentation covering business background, methodology, results, and evaluation.

## 🚀 Features

### 1. Job Data Collection
- **LinkedIn Job Scraper**: Automated scraping of job postings from LinkedIn
- **70+ Job Titles**: Comprehensive coverage across multiple industries
- **Structured Data**: Clean CSV format with job details, descriptions, and metadata

### 2. NLP Analysis Tools
- **Named Entity Recognition (NER)**: Extract skills, technologies, qualifications, and entities
- **Topic Modeling**: Discover themes using LDA and LSA
- **Word Embeddings**: Word2Vec and Sentence-BERT for semantic analysis
- **Resume Matching**: Match resumes to job descriptions

![NLP Analytics](images/1_NLP.png)

### 3. Interactive Dashboard
- **Streamlit Web App**: User-friendly interface for data exploration
- **EDA Visualizations**: Interactive charts and statistics
- **NLP Analytics**: Run NLP models and view results
- **Real-time Analysis**: Process job descriptions on-demand

## 📁 Project Structure

```
nlp-fall-2025/
├── app-streamlit/                    # Streamlit web application
│   ├── Home.py                       # Main dashboard
│   ├── utils.py                      # Utility functions and path helpers
│   ├── pages/                        # Application pages
│   │   ├── 0_Job_Crawling.py         # LinkedIn job scraper interface
│   │   ├── 1_EDA.py                  # Exploratory data analysis
│   │   ├── 2_Data_Cleaning.py        # Data cleaning and preprocessing
│   │   ├── 3_NLP_Analytics.py        # NLP analysis tools (NER, Topic Modeling)
│   │   ├── 4_Import_Embeddings.py    # Import word embeddings
│   │   ├── 5_Synthetic_Resume_Generator.py  # Generate synthetic resumes
│   │   ├── 6_Resume_Evaluation.py    # Evaluate resume quality
│   │   ├── 7_Resume_Matching.py      # Match resumes to jobs
│   │   └── 8_Meet_Our_Team.py        # Team information
│   ├── functions/                    # Helper functions and utilities
│   │   ├── database.py               # Database operations
│   │   ├── nlp_database.py            # NLP data loading utilities
│   │   ├── nlp_models.py             # NLP model implementations
│   │   ├── nlp_components.py         # NLP visualization components
│   │   ├── nlp_config.py             # NLP configuration
│   │   ├── eda_components.py         # EDA visualization components
│   │   ├── visualization.py          # General visualization utilities
│   │   ├── components.py             # UI components
│   │   ├── downloaderCSV.py          # CSV download utilities
│   │   └── menu.py                   # Navigation menu
│   ├── components/                   # Reusable UI components
│   │   ├── header.py                 # Page header component
│   │   ├── footer.py                 # Page footer component
│   │   └── card.py                   # Card component
│   ├── styles/                       # CSS styling
│   │   └── app.css                   # Main stylesheet
│   ├── images/                       # Image assets
│   ├── locales/                      # Internationalization files
│   ├── requirements.txt              # Python dependencies
│   ├── requirements-dev.txt          # Development dependencies
│   ├── Dockerfile                    # Docker configuration
│   └── create_jobs_table.sql         # Database schema
├── workspace/                        # Analysis notebooks and data
│   ├── Data/                         # Job datasets and processing
│   ├── Data_Cleaning/                # Data cleaning notebooks
│   ├── NER/                          # Named Entity Recognition notebooks
│   ├── Topic Modeling/               # LDA and LSA implementations
│   ├── Word Embedding/               # Word2Vec and SBERT notebooks
│   ├── Scrapers/                     # Scraper testing notebooks
│   ├── Resume_testing/               # Resume matching experiments
│   ├── models/                       # Trained models and embeddings
│   │   ├── word2vec_model.joblib
│   │   ├── job_embeddings_sbert_*.npy
│   │   ├── job_embeddings_w2v_*.npy
│   │   ├── topic_model_lda_*.joblib
│   │   ├── topic_model_lsa_*.joblib
│   │   └── ner_results.json
│   └── Proposal/                     # Project proposal documents
├── jupyter/                          # Jupyter notebook environment
│   ├── Dockerfile                    # Jupyter Docker configuration
│   └── requirements.txt              # Jupyter dependencies
├── linkedin-jobs-scraper/            # LinkedIn scraper library
├── scraps/                           # Raw scraped data (CSV files)
├── images/                           # Project images and screenshots
├── linkedin.py                       # LinkedIn scraper script
├── docker-compose.yml                # Docker Compose configuration
├── Final_Presentation_NLP.pdf        # Final project presentation
├── Final_Presentation_NLP.pptx       # Final project presentation (PowerPoint)
└── README.md                         # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Chrome browser (for scraping)
- LinkedIn account

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nlp-fall-2025.git
cd nlp-fall-2025
```

2. **Install dependencies**

For the Streamlit app:
```bash
cd app-streamlit
pip install -r requirements.txt
```

For development dependencies:
```bash
pip install -r requirements-dev.txt
```

For NLP models, you'll also need:
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

3. **Download NLTK data** (if needed)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🚀 Usage

### Running with Docker (Recommended)

The project includes Docker Compose configuration for easy setup:

```bash
# Start all services (PostgreSQL, Jupyter, Streamlit)
docker-compose up -d
```

Services will be available at:
- **Streamlit App**: http://localhost:48501
- **Jupyter Notebook**: http://localhost:48888
- **PostgreSQL**: localhost:45432

### Running the Web Application Locally

```bash
cd app-streamlit
streamlit run Home.py
```

The app will open in your browser at `http://localhost:8501`

**Note**: If running locally, ensure PostgreSQL is set up and configured in your `.env` file.

### Scraping LinkedIn Jobs

**Option 1: Using Streamlit UI (Recommended)**
1. Navigate to the "Job Crawling" page (Page 0) in the Streamlit app
2. Select job titles you want to scrape
3. Configure scraping parameters
4. Start scraping (ensure `LI_AT_COOKIE` is set in your `.env` file)

**Option 2: Using Command Line**

**⚠️ Important: LinkedIn Cookie Required**

You need to obtain your `li_at` cookie from LinkedIn (see "Additional Notes" section below for detailed instructions).

Then run:
```bash
LI_AT_COOKIE="your_cookie_here" python linkedin.py
```

### Running NLP Analysis

#### Using the Streamlit App

The Streamlit app provides interactive interfaces for:
- **Job Crawling** (Page 0): Scrape LinkedIn jobs directly from the UI
- **EDA** (Page 1): Explore and visualize job data
- **Data Cleaning** (Page 2): Clean and preprocess job descriptions
- **NLP Analytics** (Page 3): Run NER, Topic Modeling, and other NLP analyses
- **Import Embeddings** (Page 4): Import and manage word embeddings
- **Synthetic Resume Generator** (Page 5): Generate synthetic resumes for testing
- **Resume Evaluation** (Page 6): Evaluate resume quality
- **Resume Matching** (Page 7): Match resumes to job postings
- **Meet Our Team** (Page 8): Team information

#### Using Jupyter Notebooks

Each NLP technique has dedicated Jupyter notebooks in the `workspace/` directory:

**Named Entity Recognition:**
```bash
jupyter notebook workspace/NER/NER.ipynb
# or workspace/NER/NER_kas_edit.ipynb
```

**Topic Modeling:**
```bash
jupyter notebook workspace/Topic\ Modeling/
```

**Word Embeddings:**
```bash
jupyter notebook workspace/Word\ Embedding/
```

**Data Cleaning:**
```bash
jupyter notebook workspace/Data_Cleaning/
```

## 📊 Features Overview

### LinkedIn Job Scraper

The scraper can be run via:
1. **Command line**: `python linkedin.py` (with `LI_AT_COOKIE` environment variable)
2. **Streamlit UI**: Navigate to "Job Crawling" page (Page 0)

Scrapes job listings with filters for:
- Full-time and internship positions
- Remote work options
- Mid to senior experience level
- $100K+ base salary
- Posted within the last month

Output CSV includes:
- Job Title
- Company
- Company Link
- Date
- Date Text
- Job Link
- Insights
- Description Length
- Description

Scraped data is saved to `scraps/` directory with timestamped filenames: `linkedin_jobs_YYYYMMDD_HHMMSS.csv`

### NLP Analytics

**1. Named Entity Recognition (NER)**
- Extract skills and technologies
- Custom entity types for job-specific information

![Named Entity Recognition](images/2_NER.png)

**2. Topic Modeling**
- LDA (Latent Dirichlet Allocation)
- LSA (Latent Semantic Analysis)
- Discover hidden themes in job descriptions
- Visualize topic distributions

![Topic Modeling](images/3_topic_modeling.png)

**3. Word Embeddings**
- Word2Vec for word-level semantics
- Sentence-BERT for document similarity


![Vector Embeddings](images/4_vector_embedding.png)

**4. Resume Matching**
- Extract skills from resumes (PDF parsing)
- Compute similarity scores using embeddings
- Rank jobs by compatibility
- Provide match explanations

![Resume Matching](images/7_top_matching.png)

![Resume Matching](images/8_top_matching2.png)

**5. Synthetic Resume Generation**
- Generate synthetic resumes for testing
- Customize resume content and skills
- Export in various formats

**6. Resume Evaluation**
- Evaluate resume quality and completeness
- Score resumes against job requirements
- Provide improvement suggestions

![Resume Evaluation](images/5_evaluation.png)

![Resume Evaluation](images/6_evaluation2.png)

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root or `app-streamlit/` directory:

```env
# Database Configuration
POSTGRES_USER=admin
POSTGRES_PASSWORD=PassW0rd
POSTGRES_DB=db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql://admin:PassW0rd@localhost:5432/db

# LinkedIn Scraper
LI_AT_COOKIE=your_linkedin_cookie_here

# Optional: OpenAI API (for advanced features)
OPENAI_API_KEY=your_openai_api_key_here
```

### Job Titles to Scrape

Edit `linkedin.py` or use the Streamlit UI (Page 0) to customize job titles. The default list includes 70+ job titles across various industries.

### Scraping Parameters

In `linkedin.py` or the Streamlit interface, adjust filters:
- Experience level
- Work location (remote/hybrid/onsite)
- Salary range
- Time posted
- Job type (full-time/internship)

## 📈 Data Pipeline

1. **Collection**: LinkedIn scraper → CSV files in `scraps/`
   - Can be run via command line or Streamlit UI (Page 0)
2. **Data Cleaning**: Clean and preprocess data → `workspace/Data/`
   - Use Streamlit Page 2 or Jupyter notebooks in `workspace/Data_Cleaning/`
3. **Storage**: Store cleaned data in PostgreSQL database
   - Database schema defined in `app-streamlit/create_jobs_table.sql`
4. **Analysis**: NLP models process descriptions → Extract insights
   - NER: Extract skills, technologies, qualifications
   - Topic Modeling: Discover themes using LDA/LSA
   - Word Embeddings: Generate semantic representations
5. **Visualization**: Streamlit app displays results
   - Interactive dashboards and visualizations
   - Export capabilities for analysis results

## 🤝 Team

Capstone Fall 2025 Team Members
Boni Vasius Rosen@bvr2105-boni

Minkyung (Ginny) Kim@ginny-1334

Kas Kiatsukasem

Kibaek Kim@hyper07

Suchakrey (Philip) Nitisanon

## 📝 Notes

- The scraper uses random delays (60-240 seconds) between job title searches to avoid rate limiting
- All scraped data is timestamped: `linkedin_jobs_YYYYMMDD_HHMMSS.csv`
- Keep your LinkedIn cookie secure and don't commit it to version control
- Runs in headless mode by default for efficiency

## 🔒 Privacy & Ethics

- Only scrapes publicly available job postings
- Respects LinkedIn's rate limits with delays
- No personal data collection
- For educational/research purposes only

## 📚 Resources

### Project Documentation
- [Final Presentation PDF](Final_Presentation_NLP.pdf) - Complete project presentation with methodology, results, and evaluation

### External Resources
- [spaCy Documentation](https://spacy.io/)
- [Gensim Topic Modeling](https://radimrehurek.com/gensim/)
- [Sentence-BERT](https://www.sbert.net/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LinkedIn Jobs Scraper](https://github.com/spaam/linkedin-jobs-scraper)
- [pgvector (PostgreSQL vector extension)](https://github.com/pgvector/pgvector)


## 📄 License

This project is available for educational and research purposes. If you would like to use this code, data, or any part of this project for commercial purposes or in your own projects, please contact the project maintainers to request permission.

**For use requests, please include:**
- Your name and affiliation
- Intended use case
- Scope of usage (commercial, academic, personal, etc.)

We're happy to discuss usage terms and are generally open to collaboration and sharing, but we'd like to know how the project is being used.

## 📝 Additional Notes

### LinkedIn Cookie Setup

**⚠️ Important: LinkedIn Cookie Required**

You MUST obtain the `li_at` cookie from your LinkedIn account for the scraper to work.

#### How to get your `li_at` cookie:

1. Open Chrome (or your preferred browser) and log into your LinkedIn account
2. Open Developer Tools (F12 or Right-click → Inspect)
3. Go to the **Application** tab (Chrome) or **Storage** tab (Firefox)
4. In the left sidebar, expand **Cookies** and click on `https://www.linkedin.com`
5. Find the cookie named `li_at`
6. Copy the **Value** of the `li_at` cookie (it will be a long string)

**Note:** Keep this cookie value private and secure. Do not share it or commit it to version control. Store it in your `.env` file or pass it as an environment variable.


### Scraper Behavior

- Uses random delays (60-240 seconds) between job title searches to avoid rate limiting
- 2-second delay between individual requests
- Runs in headless mode by default
- All scraped data is timestamped: `linkedin_jobs_YYYYMMDD_HHMMSS.csv`
- Data saved to `scraps/` directory