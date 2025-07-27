# Skill Intelligence Platform

## Project Overview

The Skill Intelligence Platform is an advanced system designed to provide personalized skill recommendations, generate dynamic learning roadmaps, offer comprehensive market insights, and facilitate peer/mentor matching. Leveraging a multi-task deep learning model and robust data processing pipelines, the platform aims to empower individuals in their professional development journeys by offering data-driven insights into skill acquisition and career advancement.

## Table of Contents

1.  [Features](#1-features)
2.  [Architecture](#2-architecture)
3.  [Technology Stack](#3-technology-stack)
4.  [Setup and Installation](#4-setup-and-installation)
    * [Prerequisites](#prerequisites)
    * [Local Setup](#local-setup)
    * [Docker Deployment (Conceptual)](#docker-deployment-conceptual)
5.  [API Endpoints](#5-api-endpoints)
6.  [Data Flow and Processing](#6-data-flow-and-processing)
7.  [Machine Learning Model](#7-machine-learning-model)
8.  [Project Structure](#8-project-structure)
9.  [Future Improvements](#9-future-improvements)
10. [Contributing](#10-contributing)
11. [License](#11-license)

---

## 1. Features

The platform offers the following key functionalities:

* **Personalized Skill Recommendations**: Provides tailored skill suggestions based on a user's current skills and career goals, complete with a relevance score and reasoning.
* **Dynamic Learning Roadmaps**: Generates structured learning paths, broken down into phases with estimated durations, milestones, learning tips, and practice projects, all while respecting skill prerequisites and complementary relationships.
* **Comprehensive Skill Analysis**: Offers detailed insights into individual skills, including predicted future relevance, job demand, salary impact, learning time, and certification availability.
* **Market Insights & Competitive Positioning**: Provides data-driven insights into skill market demand, growth trends, salary impact, and competitive advantage, helping users understand their market standing.
* **Peer/Mentor Matching**: Facilitates matching users with suitable peers or mentors based on reciprocal learning/teaching interests, skill categories, and expertise levels.

## 2. Architecture

The system follows a modular, microservices-oriented architecture with a clear data processing and model serving pipeline.


**Key Architectural Components:**

  * **Data Pipeline**: A series of scripts (`data_validator.py`, `feature_engineer.py`, `data_encoder.py`, `relationship_mapper.py`) that process raw data into a model-ready format, including validation, feature creation, and encoding.
  * **Training Pipeline**: Orchestrates the training and evaluation of the deep learning model (`training_pipeline.py`, `neural_architecture.py`, `evaluation_metrics.py`).
  * **Skill Analyzer**: The core intelligence module (`skill_analyzer.py`) that loads the trained model and processed data to provide analytical functionalities like recommendations, roadmaps, and market insights.
  * **FastAPI Application**: The API layer (`main.py`) that exposes the `SkillAnalyzer` functionalities via RESTful endpoints, enabling interaction with external clients.

## 3\. Technology Stack

  * **Programming Language**: Python
  * **Data Manipulation**: Pandas
  * **Numerical Computing**: NumPy
  * **Machine Learning**: TensorFlow 2.x, Keras
  * **Pre-trained Models**: TensorFlow Hub (for BERT embeddings)
  * **Text Preprocessing**: TensorFlow Text (for BERT tokenization)
  * **Machine Learning Utilities**: Scikit-learn (for scaling, encoding, metric calculation, dimensionality reduction)
  * **Graph Processing**: NetworkX (for prerequisite and complementary skill graphs)
  * **API Framework**: FastAPI
  * **ASGI Server**: Uvicorn (Implicit from FastAPI usage)
  * **Fuzzy String Matching**: `fuzzywuzzy` / `thefuzz`
  * **Data Serialization**: JSON, Pickle
  * **Data Storage**: Parquet (for efficient columnar data), CSV (for matrices), GML (for graphs)

## 4\. Setup and Installation

### Prerequisites

  * Python 3.8+
  * Git

### Local Setup

1.  **Clone the Repository**:

    ```bash
    git clone <repository_url>
    cd skill-intelligence-platform
    ```

2.  **Create and Activate Virtual Environment**:
    It's highly recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    Navigate to the project root and install all required packages using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

    (The `requirements.txt` is based on the provided files and should be placed in the root directory. Content of `requirements.txt` is provided in a separate response if needed).

4.  **Data Preparation**:

      * Place your raw skill data file named `all_skills.jsonl` into the `data/raw/` directory.

      * **Run Data Processing Pipeline**: Execute the data processing scripts in sequence. This will validate, engineer features, encode data, and map relationships.

        ```bash
        python src/data_processing/data_validator.py
        python src/data_processing/feature_engineer.py
        python src/data_processing/data_encoder.py
        python src/data_processing/relationship_mapper.py
        ```

        *Verify that `data/processed/encoded_features_for_model.parquet` and various graph files are created.*

5.  **Model Training**:
    Train the deep learning model. This might take some time depending on your dataset size and hardware.

    ```bash
    python src/models/training_pipeline.py
    ```

    *This will save the trained model to `models/skill_intelligence_model.keras` and training history.*

6.  **Model Evaluation (Optional)**:
    To evaluate the trained model's performance on a test set:

    ```bash
    python src/models/evaluation_metrics.py
    ```

7.  **Run the FastAPI Application**:
    Navigate back to the project root directory.

    ```bash
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    ```

      * `--host 0.0.0.0`: Makes the server accessible from external devices on your network.
      * `--port 8000`: Specifies the port to run the API on.
      * `--reload`: Enables auto-reloading of the server on code changes (useful for development).

    The API will be accessible at `http://localhost:8000`.

## 5\. API Endpoints

The FastAPI application (`src/api/main.py`) exposes the following endpoints:

  * **`GET /api/v1/skills/details/{skill_name_query}`**: Get details and predictions for a specific skill.
  * **`POST /api/v1/skill/analyze`**: Analyze a single skill and get its predicted attributes.
  * **`POST /api/v1/recommendations/skills`**: Get personalized skill recommendations with reasoning.
  * **`POST /api/v1/learning-roadmap`**: Generate an optimized learning roadmap with complementary skills integration.
  * **`POST /api/v1/learning-roadmap/preview`**: Preview what skills would be included in a roadmap without full generation.
  * **`POST /api/v1/user/skill-market-insights`**: Provide market insights on skill demand and competitive positioning.
  * **`POST /api/v1/peer-match`**: Match a user with suitable peers/mentors.
  * **`POST /api/v1/user/market-position`**: Calculate user's comprehensive market position score and provide actionable insights.

Detailed request and response schemas for each endpoint are defined by Pydantic models within `src/api/main.py`.

## 6\. Data Flow and Processing

The data flows through a well-defined pipeline:

1.  **Raw Data Ingestion**: `data/raw/all_skills.jsonl` contains the initial skill records.
2.  **Data Validation (`data_validator.py`)**: Checks for missing fields, invalid types, out-of-range numerical values, duplicates, and circular dependencies in prerequisites.
3.  **Feature Engineering (`feature_engineer.py`)**: Derives new, informative features such as `skill_complexity_score`, various learning, market, and risk metrics.
4.  **Data Encoding (`data_encoder.py`)**: Transforms features into numerical formats, including `MinMaxScaler` for numerical features, `LabelEncoder` for categorical, and BERT embeddings for text, along with LDA for array features.
5.  **Relationship Mapping (`relationship_mapper.py`)**: Builds various graph structures (prerequisite, complementary, category hierarchies) and similarity matrices from the processed data.

## 7\. Machine Learning Model

The core of the intelligence lies in a multi-task deep learning model:

  * **Model Name**: `SkillIntelligenceModel`
  * **Architecture**: Custom-built multi-layer perceptron (MLP) with a multi-input, shared-backbone, multi-output structure.
      * **Inputs**: Handles numerical, categorical (via embeddings), and pre-computed text embeddings (from BERT).
      * **Shared Backbone**: Consists of `Dense` layers with `ReLU` activation, `BatchNormalization`, and `Dropout` for shared feature learning.
      * **Output Heads**: Branches into a `Regression Output Head` (linear activation) for continuous predictions and a `Binary Classification Output Head` (sigmoid activation) for binary outcomes.
  * **Training**: Orchestrated by `training_pipeline.py`, employing K-Fold cross-validation for robustness. Uses `Adam` optimizer, `mse` loss for regression, and `binary_crossentropy` for classification.
  * **Key Algorithms/Models Used**:
      * **BERT (Bidirectional Encoder Representations from Transformers)**: For generating dense, contextualized embeddings from text data.
      * **Latent Dirichlet Allocation (LDA)**: For dimensionality reduction and topic modeling on multi-hot encoded array features.
      * **Cosine Similarity**: Used for skill similarity calculations based on embeddings and TF-IDF vectors.
      * **Fuzzy String Matching**: `fuzzywuzzy`/`thefuzz` for approximate string matching.
      * **Graph Algorithms**: Implicitly used via NetworkX for representing and traversing skill dependencies.

## 8\. Project Structure

```
.
├── data/
│   ├── processed/
│   │   ├── encoded_features_for_model.parquet  # Encoded data for model training
│   │   ├── prerequisite_graph.gml            # Prerequisite graph
│   │   ├── complementary_skills_graph.gml    # Complementary skills graph
│   │   ├── skill_similarity_matrix.csv       # Skill similarity matrix
│   │   ├── skill_industry_affinity.csv       # Skill-industry affinity matrix
│   │   ├── category_hierarchy.gml            # Category hierarchy graph
│   │   └── skills_engineered_features.jsonl  # Data with engineered features
│   └── raw/
│       └── all_skills.jsonl                  # Raw input skill data
├── models/
│   ├── skill_intelligence_model.keras        # Saved Keras model
│   ├── training_history.json                 # Training history log
│   └── trained_histories/                    # Cross-validation training histories
├── src/
│   ├── api/
│   │   └── main.py                           # FastAPI application entry point
│   ├── analytics/
│   │   └── skill_analyzer.py                 # Core business logic for skill analysis
│   ├── data_processing/
│   │   ├── data_validator.py                 # Data validation and quality checks
│   │   ├── feature_engineer.py               # Feature creation and enhancement
│   │   ├── data_encoder.py                   # Data encoding and transformation
│   │   └── relationship_mapper.py            # Builds skill relationship graphs/matrices
│   └── models/
│       ├── neural_architecture.py            # Defines the deep learning model architecture
│       ├── training_pipeline.py              # Orchestrates model training with cross-validation
│       ├── evaluation_metrics.py             # Evaluates trained model performance
│       └── skill_intelligence_model.py       # (Potentially an older/simplified training script)
└── requirements.txt                          # Python dependencies
```

## 9\. Future Improvements

  * **Enhanced Data Ingestion**: Implement automated data collection from external sources (e.g., job boards, learning platforms) for continuous data updates.
  * **Advanced Model Architectures**: Explore Graph Neural Networks (GNNs) to directly leverage the rich graph structures (prerequisites, complementarities) within the deep learning model.
  * **Hyperparameter Optimization**: Implement automated hyperparameter tuning using frameworks like KerasTuner or Optuna.
  * **Personalized Learning Progress Tracking**: Develop a persistent storage solution (database) for user profiles and learning activities to enable historical progress analysis.
  * **Explainable AI (XAI)**: Integrate techniques to provide more transparent and understandable reasons behind model recommendations.
  * **Scalability**: For production deployment, consider container orchestration (Kubernetes) and cloud deployment strategies.
  * **Real-time Updates**: Explore streaming data processing for near real-time market trend analysis.

## 10\. Contributing

Contributions are welcome\! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

