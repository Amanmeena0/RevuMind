# RevuMind

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

The Multimodal Product Review Intelligence System is an intelligent platform that analyzes product reviews using both textual and visual data. By leveraging advanced machine learning and natural language processing techniques, the system provides deep insights into product perceptions and customer feedback across multiple modalities.

## Project Organization

```text
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
├── models             <- Trained and serialized models, model predictions, or model summaries
├── notebooks          <- Jupyter notebooks for exploration
├── pyproject.toml     <- Project configuration file with package metadata
├── references         <- Data dictionaries, manuals, and all other explanatory materials
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
├── requirements.txt   <- The requirements file for reproducing the analysis environment
├── revumind.db        <- Main SQLite database
└── revumind           <- Source code for use in this project
    ├── __init__.py    <- Makes revumind a Python module
    ├── analytics      <- Builder logic for generating aggregate analytics
    ├── api            <- FastAPI backend endpoints
    ├── core           <- Core database session and configuration logic
    ├── data           <- Data gathering, synthetic data generation, and EDA
    ├── db             <- Database models and initialization scripts
    ├── models         <- Training scripts and inferencing models (ABSA, NLP, topics, etc.)
    ├── models_code    <- Code for specific prediction models like helpfulness and tuning
    ├── nlp            <- Core NLP processing scripts (sentiment, NER, summarizer, etc.)
    ├── pipeline       <- Data ingest, preprocessing, training, and inference pipelines
    ├── utils          <- Utility functions, constants, and helpers
    └── visualization  <- Streamlit dashboards, business insights, and visualizations
```

## Setup & Installation

1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. The project is managed via `pyproject.toml` and requires Python ~3.14.0. You can install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Key Technologies

- **API & Web Framework:** FastAPI, Uvicorn, SQLAlchemy, Streamlit
- **Data Processing:** Pandas, NumPy, PyArrow
- **Machine Learning & Deep Learning:** PyTorch, Torchvision, Scikit-learn
- **NLP & Text Processing:** spaCy, NLTK, BeautifulSoup4
- **Visualization:** Matplotlib, Seaborn, Plotly
