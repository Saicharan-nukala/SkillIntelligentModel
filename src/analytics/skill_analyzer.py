# src/analytics/skill_analyzer.py (Updated perform_skill_analysis and _get_difficulty_level)

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text
import logging
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
from thefuzz import fuzz
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SkillAnalyzer:
    def __init__(self):
        self.model = None
        self.df_processed = None
        self.bert_preprocess_model = None
        self.bert_encoder_model = None
        self.unique_categories = []
        self.category_embeddings = None
        self._skill_analysis_cache = {}
        # Add attributes for loaded relationship graphs/matrices
        self.prerequisite_graph = None
        self.complementary_skills_graph = None
        self.skill_similarity_matrix_df = None
        self.skill_industry_affinity_df = None
        self.category_hierarchy_graph = None


        self.MODEL_SAVE_PATH = 'models/skill_intelligence_model.keras'
        self.PROCESSED_DATA_PATH = 'data/processed/encoded_features_for_model.parquet'

        # Paths for relationship mapper outputs
        self.PREREQUISITE_GRAPH_PATH = "data/processed/prerequisite_graph.gml"
        self.COMPLEMENTARY_GRAPH_PATH = "data/processed/complementary_skills_graph.gml"
        self.SKILL_SIMILARITY_MATRIX_PATH = "data/processed/skill_similarity_matrix.csv"
        self.SKILL_INDUSTRY_AFFINITY_PATH = "data/processed/skill_industry_affinity.csv"
        self.CATEGORY_HIERARCHY_PATH = "data/processed/category_hierarchy.gml"


        self.BERT_MODEL_URL = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1"
        self.BERT_PREPROCESS_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

        self.NUMERICAL_FEATURES = [
            'learning_time_days', 'popularity_score', 'job_demand_score',
            'salary_impact_percent', 'future_relevance_score', 'learning_resources_quality',
            'skill_complexity_score', 'market_momentum_score', 'ecosystem_richness',
            'industry_diversity_metric', 'resource_availability_index', 'learning_accessibility_score'
        ]

        self.CATEGORICAL_FEATURES_MAPPING = {
            'category_encoded': 'skill_category_encoded_input',
            'skill_type_encoded': 'skill_type_encoded_input',
            'market_trend_encoded': 'market_trend_encoded_input'
        }

        self.TEXT_FEATURES_MAPPING = {
            'skill_name': 'skill_name_embedding_input',
            'prerequisites': 'prerequisites_embedding_input',
            'complementary_skills': 'complementary_skills_embedding_input',
            'industry_usage_text': 'industry_embedding_input'
        }
        
        self.DIFFICULTY_THRESHOLDS = {
            'basic': 0.33,
            'medium': 0.66,
            'hard': 1.0
        }

        # Mapping for model output indices to target names (must match training_pipeline.py/evaluation_metrics.py)
        self.REGRESSION_OUTPUT_NAMES = [
            'future_relevance_score', 'job_demand_score', 'salary_impact_percent', 'learning_time_days'
        ]
        self.BINARY_CLASSIFICATION_OUTPUT_NAME = 'risk_of_obsolescence_binary'


    def _is_technology_extension_match(self, query: str, skill_name: str) -> bool:
        """Check if this is a case like 'Node' matching with 'Node.js'"""
        common_extensions = ['.js', '.net', '.py', '.java', '.ts', '.go', '.rb', '.php', '.sh','1','2','3','5','4','5','6','7','8','9','10']
        query_lower = query.lower()
        skill_lower = skill_name.lower()
        
        # Case 1: Query is base name, skill has extension
        if any(skill_lower == f"{query_lower}{ext}" for ext in common_extensions):
            return True
        
        # Case 2: Skill is base name, query has extension
        if any(query_lower == f"{skill_lower}{ext}" for ext in common_extensions):
            return True
        
        # Case 3: Both have different extensions but same base
        for ext1 in common_extensions:
            for ext2 in common_extensions:
                if (query_lower.endswith(ext1) and skill_lower.endswith(ext2)):
                    base1 = query_lower[:-len(ext1)]
                    base2 = skill_lower[:-len(ext2)]
                    if base1 == base2:
                        return True
        return False

    def load_resources(self):
        """Loads the ML model, data, BERT models, and graph/matrix resources into memory."""
        
        # Load ML Model
        if self.model is None:
            if not os.path.exists(self.MODEL_SAVE_PATH):
                raise RuntimeError(f"Model not found at {self.MODEL_SAVE_PATH}. Please train the model first.")
            try:
                logging.info(f"Loading trained model from {self.MODEL_SAVE_PATH}...")
                custom_objects = {'KerasLayer': hub.KerasLayer}
                self.model = tf.keras.models.load_model(self.MODEL_SAVE_PATH, custom_objects=custom_objects)
                self.model.summary()
                logging.info("Model loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Error loading model: {e}")

        # Load Processed Data
        if self.df_processed is None:
            if not os.path.exists(self.PROCESSED_DATA_PATH):
                raise RuntimeError(f"Processed data not found at {self.PROCESSED_DATA_PATH}. Please run data processing pipeline first.")
            try:
                logging.info(f"Loading processed data from {self.PROCESSED_DATA_PATH}...")
                self.df_processed = pd.read_parquet(self.PROCESSED_DATA_PATH)
                self.df_processed['skill_name_lower'] = self.df_processed['skill_name'].str.lower()

                # Ensure 'difficulty_level_scaled' is always present and numeric
                if 'difficulty_level' in self.df_processed.columns:
                    if pd.api.types.is_numeric_dtype(self.df_processed['difficulty_level']):
                        min_diff = self.df_processed['difficulty_level'].min()
                        max_diff = self.df_processed['difficulty_level'].max()
                        if max_diff > min_diff:
                            self.df_processed['difficulty_level_scaled'] = (self.df_processed['difficulty_level'] - min_diff) / (max_diff - min_diff)
                        else:
                            self.df_processed['difficulty_level_scaled'] = 0.5 # Default if all are same
                    else: # Handle categorical like 'Beginner', 'Intermediate', 'Advanced'
                        difficulty_map = {'Beginner': 0.1, 'Intermediate': 0.5, 'Advanced': 0.9}
                        self.df_processed['difficulty_level_scaled'] = self.df_processed['difficulty_level'].map(difficulty_map).fillna(0.5) # Default to medium
                elif 'difficulty_level_scaled' not in self.df_processed.columns:
                    logging.warning("Neither 'difficulty_level' nor 'difficulty_level_scaled' found in processed data. Setting 'difficulty_level_scaled' to 0.5 (medium).")
                    self.df_processed['difficulty_level_scaled'] = 0.5 # Default to medium if no info

                # Ensure consistent 'industry_usage_text' for BERT
                if 'industry_usage' in self.df_processed.columns and 'industry_usage_text' not in self.df_processed.columns:
                    # Ensure industry_usage is parsed as list if it's stringified
                    self.df_processed['industry_usage'] = self.df_processed['industry_usage'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                    self.df_processed['industry_usage_text'] = self.df_processed['industry_usage'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
                elif 'industry_usage' not in self.df_processed.columns and 'industry_usage_text' not in self.df_processed.columns:
                    logging.warning("Neither 'industry_usage' nor 'industry_usage_text' found in processed data. Setting 'industry_usage_text' to empty string.")
                    self.df_processed['industry_usage_text'] = ""

                if 'prerequisites' in self.df_processed.columns and self.df_processed['prerequisites'].apply(type).eq(str).any():
                    self.df_processed['prerequisites'] = self.df_processed['prerequisites'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                if 'complementary_skills' in self.df_processed.columns and self.df_processed['complementary_skills'].apply(type).eq(str).any():
                    self.df_processed['complementary_skills'] = self.df_processed['complementary_skills'].apply(lambda x: eval(x) if isinstance(x, str) else x)


                if 'description' not in self.df_processed.columns:
                    self.df_processed['description'] = self.df_processed['skill_name'].apply(lambda x: f"A fundamental skill related to {x}.")

                logging.info(f"Data loaded successfully. Total skills: {len(self.df_processed)}")
            except Exception as e:
                raise RuntimeError(f"Error loading processed data: {e}")

        # Load BERT Models and pre-compute embeddings
        if self.bert_preprocess_model is None or self.bert_encoder_model is None:
            try:
                logging.info(f"Loading BERT preprocessor from: {self.BERT_PREPROCESS_URL}")
                self.bert_preprocess_model = hub.KerasLayer(self.BERT_PREPROCESS_URL)
                logging.info(f"Loading BERT encoder from: {self.BERT_MODEL_URL}")
                self.bert_encoder_model = hub.KerasLayer(self.BERT_MODEL_URL)

                # Generate and store skill name embeddings if not present or invalid
                if 'skill_name_embedding_vector' not in self.df_processed.columns or self.df_processed['skill_name_embedding_vector'].isnull().any() or not all(isinstance(x, np.ndarray) and x.shape == (128,) for x in self.df_processed['skill_name_embedding_vector'].dropna()):
                    logging.info("Generating BERT embeddings for all skill names in the dataset...")
                    all_skill_names = self.df_processed['skill_name'].tolist()
                    processed_texts = self.bert_preprocess_model(tf.constant(all_skill_names))
                    skill_embeddings = self.bert_encoder_model(processed_texts)['pooled_output'].numpy()
                    self.df_processed['skill_name_embedding_vector'] = list(skill_embeddings)
                    logging.info("Skill name embeddings generated and stored.")
                else:
                    logging.info("Skill name embeddings already present and valid in data.")
                
                # Pre-compute category embeddings for faster semantic search
                self.unique_categories = self.df_processed['category'].dropna().unique().tolist()
                if self.unique_categories:
                    logging.info("Generating BERT embeddings for unique categories...")
                    processed_categories = self.bert_preprocess_model(tf.constant(self.unique_categories))
                    self.category_embeddings = self.bert_encoder_model(processed_categories)['pooled_output'].numpy()
                    logging.info(f"Category embeddings generated for {len(self.unique_categories)} categories.")

            except Exception as e:
                logging.error(f"Error loading BERT models: {e}")
                logging.error("Please ensure you have an active internet connection or the models are cached locally.")
                raise RuntimeError(f"Failed to load BERT resources: {e}")

        # Load Relationship Mapper Outputs
        if self.prerequisite_graph is None:
            if not os.path.exists(self.PREREQUISITE_GRAPH_PATH):
                logging.warning(f"Prerequisite graph not found at {self.PREREQUISITE_GRAPH_PATH}. Roadmaps might be limited.")
            else:
                try:
                    self.prerequisite_graph = nx.read_gml(self.PREREQUISITE_GRAPH_PATH)
                    logging.info(f"Prerequisite graph loaded with {self.prerequisite_graph.number_of_nodes()} nodes and {self.prerequisite_graph.number_of_edges()} edges.")
                except Exception as e:
                    logging.error(f"Error loading prerequisite graph: {e}")
                    self.prerequisite_graph = nx.DiGraph() # Initialize empty to avoid errors

        if self.complementary_skills_graph is None:
            if not os.path.exists(self.COMPLEMENTARY_GRAPH_PATH):
                logging.warning(f"Complementary skills graph not found at {self.COMPLEMENTARY_GRAPH_PATH}. Recommendations might be less accurate.")
            else:
                try:
                    self.complementary_skills_graph = nx.read_gml(self.COMPLEMENTARY_GRAPH_PATH)
                    logging.info(f"Complementary skills graph loaded with {self.complementary_skills_graph.number_of_nodes()} nodes and {self.complementary_skills_graph.number_of_edges()} edges.")
                except Exception as e:
                    logging.error(f"Error loading complementary skills graph: {e}")
                    self.complementary_skills_graph = nx.Graph() # Initialize empty

        if self.skill_industry_affinity_df is None:
            if not os.path.exists(self.SKILL_INDUSTRY_AFFINITY_PATH):
                logging.warning(f"Skill-Industry Affinity matrix not found at {self.SKILL_INDUSTRY_AFFINITY_PATH}.")
            else:
                try:
                    self.skill_industry_affinity_df = pd.read_csv(self.SKILL_INDUSTRY_AFFINITY_PATH, index_col=0)
                    logging.info(f"Skill-Industry Affinity matrix loaded. Shape: {self.skill_industry_affinity_df.shape}")
                except Exception as e:
                    logging.error(f"Error loading skill-industry affinity matrix: {e}")
                    self.skill_industry_affinity_df = pd.DataFrame() # Initialize empty

        if self.category_hierarchy_graph is None:
            if not os.path.exists(self.CATEGORY_HIERARCHY_PATH):
                logging.warning(f"Category hierarchy graph not found at {self.CATEGORY_HIERARCHY_PATH}.")
            else:
                try:
                    self.category_hierarchy_graph = nx.read_gml(self.CATEGORY_HIERARCHY_PATH)
                    logging.info(f"Category hierarchy graph loaded with {self.category_hierarchy_graph.number_of_nodes()} nodes and {self.category_hierarchy_graph.number_of_edges()} edges.")
                except Exception as e:
                    logging.error(f"Error loading category hierarchy graph: {e}")
                    self.category_hierarchy_graph = nx.DiGraph() # Initialize empty


    def prepare_model_input_from_series(self, row: pd.Series) -> Dict[str, np.ndarray]:
        """
        Prepares input dictionary for the Keras model from a single DataFrame row (pd.Series).
        Ensures all expected input keys from neural_architecture.py are present.
        """
        model_input = {}

        # Numerical features
        numerical_values = []
        for feature in self.NUMERICAL_FEATURES:
            if feature in row and pd.notna(row[feature]):
                numerical_values.append(row[feature])
            else:
                logging.warning(f"Numerical feature '{feature}' not found or is NaN in row. Using 0.0.")
                numerical_values.append(0.0)
        model_input['numerical_features_input'] = np.array([numerical_values]).astype(np.float32)

        # Categorical features
        for encoded_col, model_input_name in self.CATEGORICAL_FEATURES_MAPPING.items():
            if encoded_col in row and pd.notna(row[encoded_col]):
                model_input[model_input_name] = np.array([int(row[encoded_col])]).reshape(1, 1).astype(np.int32)
            else:
                logging.warning(f"Categorical feature '{encoded_col}' not found or is NaN in row. Using 0 for '{model_input_name}'.")
                model_input[model_input_name] = np.array([0]).reshape(1, 1).astype(np.int32)

        # Text embeddings
        for original_text_feature, model_input_name in self.TEXT_FEATURES_MAPPING.items():
            # For list-like features, ensure they are converted to text string for embedding
            text_for_embedding = ''
            value = row.get(original_text_feature, None)
            if value is not None and not (isinstance(value, (np.ndarray, list)) and pd.isna(value).any()):
                if isinstance(row[original_text_feature], list):
                    text_for_embedding = " ".join(row[original_text_feature])
                else:
                    text_for_embedding = str(row[original_text_feature])
            elif original_text_feature == 'industry_usage_text' and 'industry_usage' in row and isinstance(row['industry_usage'], list):
                text_for_embedding = " ".join(row['industry_usage'])
            
            # Check if pre-computed embedding exists and is valid
            if f"{original_text_feature}_embedding_vector" in row and isinstance(row[f"{original_text_feature}_embedding_vector"], np.ndarray):
                model_input[model_input_name] = np.array([row[f"{original_text_feature}_embedding_vector"]]).astype(np.float32)
            else:
                # Dynamically generate embedding if not pre-computed or invalid
                text_value = str(text_for_embedding if text_for_embedding is not None else '')
                logging.debug(f"Generating embedding for {original_text_feature}: '{text_value}'")
                text_preprocessed = self.bert_preprocess_model(tf.constant([text_value]))
                text_embedding = self.bert_encoder_model(text_preprocessed)['pooled_output']
                model_input[model_input_name] = text_embedding

        return model_input

    def get_skill_suggestions(self, skill_query: str) -> List[Dict[str, Any]]:
        """
        Finds and ranks skills based on fuzzy matching and semantic similarity.
        Prioritizes exact matches, then tech extension matches, then high-score fuzzy, then semantic.
        """
        if self.df_processed is None or self.bert_preprocess_model is None or self.bert_encoder_model is None:
            raise RuntimeError("Resources not loaded for skill suggestions.")

        skill_query_lower = skill_query.lower()

        # 1. Exact Match Check
        exact_match = self.df_processed[self.df_processed['skill_name_lower'] == skill_query_lower]
        if not exact_match.empty:
            return [{
                'skill_name': exact_match.iloc[0]['skill_name'],
                'score': 1.0,
                'match_type': 'exact'
            }]

        # 2. Technology Extension Match Check
        tech_matches = []
        for _, row in self.df_processed.iterrows():
            if self._is_technology_extension_match(skill_query, row['skill_name']):
                tech_matches.append({
                    'skill_name': row['skill_name'],
                    'score': 0.95,
                    'match_type': 'tech_extension'
                })
        
        if tech_matches:
            return sorted(tech_matches, key=lambda x: (-x['score'], x['skill_name']))[:1]

        # 3. Prioritized Fuzzy Matching
        best_fuzzy_match = None
        highest_fuzzy_score = 0

        for _, row in self.df_processed.iterrows():
            skill_name_original = row['skill_name']
            skill_name_lower_in_data = row['skill_name_lower']

            score = fuzz.token_sort_ratio(skill_query_lower, skill_name_lower_in_data)

            if score > highest_fuzzy_score:
                highest_fuzzy_score = score
                best_fuzzy_match = {
                    'skill_name': skill_name_original,
                    'score': score / 100.0,
                    'match_type': 'fuzzy'
                }

        if best_fuzzy_match and best_fuzzy_match['score'] >= 0.90:
            logging.info(f"Prioritizing high-score fuzzy match: {best_fuzzy_match['skill_name']} with score {best_fuzzy_match['score']:.2f}")
            return [best_fuzzy_match]

        suggestions = []

        if best_fuzzy_match and best_fuzzy_match['score'] >= 0.60:
            suggestions.append(best_fuzzy_match)

        # 4. BERT-based Semantic Similarity Search
        try:
            input_skill_processed = self.bert_preprocess_model(tf.constant([skill_query]))
            input_skill_embedding = self.bert_encoder_model(input_skill_processed)['pooled_output'].numpy()

            valid_embeddings_df = self.df_processed[self.df_processed['skill_name_embedding_vector'].apply(lambda x: isinstance(x, np.ndarray) and x.shape == (128,))]

            if not valid_embeddings_df.empty:
                similarities_bert = cosine_similarity(input_skill_embedding, np.stack(valid_embeddings_df['skill_name_embedding_vector'].values))

                bert_similarity_df = pd.DataFrame({
                    'skill_name': valid_embeddings_df['skill_name'],
                    'similarity_score': similarities_bert[0]
                })

                excluded_skill_name = best_fuzzy_match['skill_name'].lower() if best_fuzzy_match else ""

                bert_suggestions = bert_similarity_df[
                    (bert_similarity_df['skill_name'].str.lower() != skill_query_lower) &
                    (bert_similarity_df['skill_name'].str.lower() != excluded_skill_name)
                ].sort_values(by='similarity_score', ascending=False)

                for _, row in bert_suggestions.head(5).iterrows():
                    if row['similarity_score'] > 0.5:
                        suggestions.append({
                            'skill_name': row['skill_name'],
                            'score': row['similarity_score'],
                            'match_type': 'semantic'
                        })
        except Exception as e:
            logging.warning(f"Error during BERT similarity search: {e}")

        # Final deduplication and sorting
        unique_suggestions = []
        seen_skills = set()

        all_suggestions_sorted = sorted(suggestions, key=lambda x: x['score'], reverse=True)

        for sug in all_suggestions_sorted:
            skill_name_lower = sug['skill_name'].lower()
            is_duplicate_of_added = False
            for seen_lower_name in seen_skills:
                if fuzz.token_sort_ratio(skill_name_lower, seen_lower_name) > 90:
                    is_duplicate_of_added = True
                    break
            
            if not is_duplicate_of_added:
                unique_suggestions.append(sug)
                seen_skills.add(skill_name_lower)

        return unique_suggestions

    # --- Updated perform_skill_analysis (clean, patched for BERT and industry_usage_text issues) ---
    def perform_skill_analysis(self, skill_name_query: str) -> Dict[str, Any]:
        if self.df_processed is None or self.model is None or self.bert_preprocess_model is None or self.bert_encoder_model is None:
            raise RuntimeError("SkillAnalyzer resources are not yet loaded.")

        cache_key = skill_name_query.lower()
        if cache_key in self._skill_analysis_cache:
            logger.info(f"Returning cached analysis for skill: {skill_name_query}")
            return self._skill_analysis_cache[cache_key]

        skill_name_lower_query = skill_name_query.lower()
        found_skill_row = pd.DataFrame()

        try:
            mask = self.df_processed['skill_name_lower'] == skill_name_lower_query
            found_skill_row = self.df_processed[mask].copy()
            logger.info(f"Direct search for '{skill_name_query}': {len(found_skill_row)} matches found")
        except Exception as e:
            logger.error(f"Error during direct skill search for '{skill_name_query}': {e}")

        if len(found_skill_row) == 0:
            try:
                suggestions = self.get_skill_suggestions(skill_name_query)
                if suggestions:
                    top_suggestion = suggestions[0]
                    score_good = top_suggestion['score'] >= 0.9
                    match_type_good = top_suggestion['match_type'] in ['exact', 'semantic', 'fuzzy', 'tech_extension']
                    use_suggestion = score_good or (top_suggestion['score'] >= 0.8 and match_type_good)

                    if use_suggestion:
                        suggestion_mask = self.df_processed['skill_name_lower'] == top_suggestion['skill_name'].lower()
                        found_skill_row = self.df_processed[suggestion_mask].copy()

                        if len(found_skill_row) == 0:
                            raise ValueError(f"Suggested skill '{top_suggestion['skill_name']}' not found in dataset")
                    else:
                        suggestion_names = [s['skill_name'] for s in suggestions[:3]]
                        raise ValueError(
                            f"Skill '{skill_name_query}' not found. Best suggestion '{top_suggestion['skill_name']}' "
                            f"has insufficient score ({top_suggestion['score']:.2f}). "
                            f"Available suggestions: {suggestion_names}"
                        )
                else:
                    raise ValueError(f"Skill '{skill_name_query}' not found and no suggestions available")
            except Exception as e:
                raise ValueError(f"Could not find or suggest skill '{skill_name_query}': {str(e)}")

        if len(found_skill_row) > 1:
            logger.warning(f"Multiple skills found for '{skill_name_query}'. Analyzing the first match.")
            found_skill_row = found_skill_row.iloc[[0]]

        try:
            skill_data = found_skill_row.iloc[0].to_dict()
            actual_skill_name = skill_data.get('skill_name', skill_name_query)
            logger.info(f"Processing skill data for: {actual_skill_name}")
        except Exception as e:
            raise ValueError(f"Could not extract data for skill '{skill_name_query}': {str(e)}")

        try:
            # Fix malformed industry_usage_text
            if 'industry_usage_text' in skill_data:
                raw_text = skill_data['industry_usage_text']
                if isinstance(raw_text, str) and raw_text.startswith("["):
                    try:
                        parsed_list = eval(raw_text)
                        if isinstance(parsed_list, list):
                            skill_data['industry_usage_text'] = " ".join(parsed_list)
                    except:
                        skill_data['industry_usage_text'] = raw_text.replace("[", "").replace("]", "").replace("'", "")

            if 'difficulty_level' in skill_data and pd.notna(skill_data['difficulty_level']):
                if isinstance(skill_data['difficulty_level'], (int, float)):
                    skill_data['difficulty_level_scaled'] = float(skill_data['difficulty_level'])
                else:
                    difficulty_map = {'Beginner': 0.1, 'Intermediate': 0.5, 'Advanced': 0.9}
                    skill_data['difficulty_level_scaled'] = difficulty_map.get(str(skill_data['difficulty_level']).lower(), 0.5)
            elif 'difficulty_level_scaled' not in skill_data or pd.isna(skill_data.get('difficulty_level_scaled')):
                skill_data['difficulty_level_scaled'] = 0.5

            # Normalize other key fields
            for key, default, scale in [
                ('learning_time_days', 30, 365),
                ('salary_impact_percent', 15, 100),
            ]:
                if key in skill_data and pd.notna(skill_data[key]):
                    val = skill_data[key]
                    try:
                        val = float(val)
                        skill_data[key] = int(round(val * scale)) if val <= 1.0 else int(round(val))
                    except:
                        skill_data[key] = default
                else:
                    skill_data[key] = default

            if 'description' not in skill_data or pd.isna(skill_data.get('description')):
                skill_data['description'] = f"A skill related to {actual_skill_name}."

        except Exception as e:
            logger.error(f"Error processing skill data attributes: {e}")

        try:
            logger.info(f"Getting model predictions for: {actual_skill_name}")
            model_input_data = self.prepare_model_input_from_series(found_skill_row.iloc[0])
            predictions = self.model.predict(model_input_data, verbose=0)

            skill_data['predicted_future_relevance_score'] = None
            skill_data['predicted_salary_impact_percent'] = None
            skill_data['predicted_job_demand_score'] = None
            skill_data['predicted_learning_time_days'] = None
            skill_data['predicted_certification_available'] = None

            if isinstance(predictions, dict):
                if 'regression_outputs' in predictions:
                    reg_preds = predictions['regression_outputs'][0]
                    skill_data['predicted_future_relevance_score'] = round(float(reg_preds[0]) * 10, 2)
                    skill_data['predicted_job_demand_score'] = round(float(reg_preds[1]) * 10, 2)
                    skill_data['predicted_salary_impact_percent'] = int(round(float(reg_preds[2]) * 100))
                    skill_data['predicted_learning_time_days'] = int(round(float(reg_preds[3]) * 365))
                if 'binary_classification_outputs' in predictions:
                    prob = float(tf.nn.sigmoid(predictions['binary_classification_outputs'][0][0]).numpy())
                    skill_data['predicted_certification_available'] = bool(prob > 0.5)

            elif isinstance(predictions, (list, tuple)) and len(predictions) >= 2:
                reg_preds = predictions[0][0]
                bin_preds = predictions[1][0][0]
                skill_data['predicted_future_relevance_score'] = round(float(reg_preds[0]) * 10, 2)
                skill_data['predicted_job_demand_score'] = round(float(reg_preds[1]) * 10, 2)
                skill_data['predicted_salary_impact_percent'] = int(round(float(reg_preds[2]) * 100))
                skill_data['predicted_learning_time_days'] = int(round(float(reg_preds[3]) * 365))
                prob = float(tf.nn.sigmoid(bin_preds).numpy())
                skill_data['predicted_certification_available'] = bool(prob > 0.5)

        except Exception as e:
            logger.error(f"Error during model prediction for skill '{actual_skill_name}': {e}", exc_info=True)

        # Cleanup
        for col in list(skill_data):
            if '_embedding_vector' in col or '_encoded' in col or col == 'skill_name_lower':
                skill_data.pop(col, None)

        if 'difficulty_level_scaled' in skill_data and 'difficulty_level' in skill_data:
            skill_data.pop('difficulty_level', None)

        if 'industry_usage_text' in skill_data and 'industry_usage' in skill_data:
            skill_data.pop('industry_usage', None)

        skill_data['skill_name'] = actual_skill_name
        self._skill_analysis_cache[cache_key] = skill_data.copy()
        logger.info(f"Successfully analyzed and cached skill: {actual_skill_name}")

        return skill_data
    
    def _get_relevant_categories_from_goals(self, goals: List[str]) -> List[str]:
        """
        Enhanced goal matching: searches goals in category -> industry_usage -> complementary_skills
        Leverages loaded category_hierarchy_graph for broader matches.
        """
        relevant_categories = set()
        if not self.unique_categories:
            logging.warning("Unique categories not loaded. Cannot predict categories.")
            return []

        for goal in goals:
            goal_lower = goal.lower()
            matched = False
            
            # Step 1: Search in categories first (fuzzy matching)
            best_fuzzy_category = None
            highest_fuzzy_score = 0
            for category in self.unique_categories:
                score = fuzz.token_sort_ratio(goal_lower, category.lower())
                if score > highest_fuzzy_score:
                    highest_fuzzy_score = score
                    best_fuzzy_category = category
            
            if best_fuzzy_category and highest_fuzzy_score >= 70:  # Lower threshold for categories
                relevant_categories.add(best_fuzzy_category)
                logging.info(f"Goal '{goal}' matched to category '{best_fuzzy_category}' (score: {highest_fuzzy_score})")
                matched = True
                continue
            
            # Step 2: If not found in categories, search in industry_usage
            if not matched:
                industry_matched_categories = set()
                for _, row in self.df_processed.iterrows():
                    industry_text = str(row.get('industry_usage', '')).lower()
                    if fuzz.partial_ratio(goal_lower, industry_text) >= 75:  # Fuzzy match in industry usage
                        industry_matched_categories.add(row['category'])
                
                if industry_matched_categories:
                    relevant_categories.update(industry_matched_categories)
                    logging.info(f"Goal '{goal}' matched via industry_usage to categories: {industry_matched_categories}")
                    matched = True
                    continue
            
            # Step 3: If still not found, search in complementary_skills
            if not matched:
                comp_matched_categories = set()
                for _, row in self.df_processed.iterrows():
                    comp_skills = row.get('complementary_skills', '')
                    if isinstance(comp_skills, str):
                        comp_skills_text = comp_skills.lower()
                    elif isinstance(comp_skills, list):
                        comp_skills_text = ' '.join([str(c).lower() for c in comp_skills])
                    else:
                        comp_skills_text = ''
                    
                    if comp_skills_text and fuzz.partial_ratio(goal_lower, comp_skills_text) >= 75:
                        comp_matched_categories.add(row['category'])
                
                if comp_matched_categories:
                    relevant_categories.update(comp_matched_categories)
                    logging.info(f"Goal '{goal}' matched via complementary_skills to categories: {comp_matched_categories}")
                    matched = True
                    continue
            
            # Step 4: Fallback to BERT semantic similarity if nothing found
            if not matched and self.category_embeddings is not None:
                try:
                    goal_processed = self.bert_preprocess_model(tf.constant([goal]))
                    goal_embedding = self.bert_encoder_model(goal_processed)['pooled_output'].numpy()
                    
                    similarities = cosine_similarity(goal_embedding, self.category_embeddings)[0]
                    top_category_indices = similarities.argsort()[-2:][::-1]
                    
                    for idx in top_category_indices:
                        if similarities[idx] > 0.6:  # Higher threshold for semantic matching
                            relevant_categories.add(self.unique_categories[idx])
                            logging.info(f"Goal '{goal}' semantically matched to category '{self.unique_categories[idx]}' (score: {similarities[idx]:.2f})")
                            matched = True
                except Exception as e:
                    logging.warning(f"Could not semantically predict category for goal '{goal}': {e}")

        # NEW: Expand matched categories using the category hierarchy graph
        if self.category_hierarchy_graph:
            expanded_categories = set(relevant_categories)
            for cat in relevant_categories:
                # Add child categories (more specific categories under a matched parent)
                if cat in self.category_hierarchy_graph: # Check if node exists
                    for child in nx.descendants(self.category_hierarchy_graph, cat):
                        expanded_categories.add(child)
                # Add parent categories (broader context for a matched specific category)
                for ancestor in nx.ancestors(self.category_hierarchy_graph, cat):
                     expanded_categories.add(ancestor)
            relevant_categories = expanded_categories
            logging.info(f"Categories expanded via hierarchy: {relevant_categories}")

        return list(relevant_categories)


    def _get_difficulty_level(self, score: float) -> str:
        """Categorizes a scaled difficulty score into 'basic', 'medium', or 'hard'."""
        # Ensure score is a scalar float, in case it's a 0-dim numpy array or similar
        if isinstance(score, np.ndarray):
            if score.size == 1:
                score = float(score.item()) # Extract scalar from 0-dim array
            else:
                # This case should ideally not happen if data preparation is correct.
                # If it does, it indicates a deeper issue in how `score` is being passed.
                logging.warning(f"'_get_difficulty_level' received a multi-element array: {score}. Using its mean for categorization.")
                score = float(np.mean(score))


        if score <= self.DIFFICULTY_THRESHOLDS['basic']:
            return 'basic'
        elif score <= self.DIFFICULTY_THRESHOLDS['medium']: # Corrected from DIFFICOLDS_THRESHOLDS
            return 'medium'
        else:
            return 'hard'

    def recommend_skills(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhanced skill recommendations using BERT, job demand, future relevance,
        complementary skills graph, industry affinity, and skill type matching.
        """
        if self.df_processed is None or self.model is None or self.bert_preprocess_model is None or self.bert_encoder_model is None:
            raise RuntimeError("SkillAnalyzer resources are not yet loaded for recommendations.")

        user_skill_embeddings = []
        user_skill_difficulty_scores = []
        user_skill_types = set()  # Track user's skill types

        for skill in user_profile['current_skills']:
            try:
                processed_text = self.bert_preprocess_model(tf.constant([skill]))
                embedding = self.bert_encoder_model(processed_text)['pooled_output'].numpy()
                user_skill_embeddings.append(embedding)

                matched_skill = self.df_processed[self.df_processed['skill_name_lower'] == skill.lower()]
                if not matched_skill.empty:
                    user_skill_difficulty_scores.append(matched_skill.iloc[0]['difficulty_level_scaled'])
                    # Collect user's skill types
                    skill_type = matched_skill.iloc[0].get('skill_type', '')
                    if skill_type:
                        user_skill_types.add(skill_type)
            except Exception as e:
                logging.warning(f"Could not embed or get difficulty for user skill '{skill}': {e}")

        user_goal_embeddings = []
        for goal in user_profile['goals']:
            try:
                processed_text = self.bert_preprocess_model(tf.constant([goal]))
                embedding = self.bert_encoder_model(processed_text)['pooled_output'].numpy()
                user_goal_embeddings.append(embedding)
            except Exception as e:
                logging.warning(f"Could not embed user goal '{goal}': {e}")

        user_combined_embedding = None
        if user_skill_embeddings or user_goal_embeddings:
            all_user_embeddings = user_skill_embeddings + user_goal_embeddings
            user_combined_embedding = np.mean(all_user_embeddings, axis=0)

        if user_combined_embedding is None:
            raise ValueError("Could not process user profile: no valid skills or goals provided.")

        user_level = 0.5
        if user_skill_difficulty_scores:
            user_level = np.mean(user_skill_difficulty_scores)
        logging.info(f"Calculated user_level: {user_level:.2f}")
        logging.info(f"User skill types: {user_skill_types}")

        # Use enhanced goal matching
        relevant_goal_categories = self._get_relevant_categories_from_goals(user_profile['goals'])
        logging.info(f"Relevant categories identified from goals: {relevant_goal_categories}")

        valid_embeddings_df = self.df_processed[self.df_processed['skill_name_embedding_vector'].apply(lambda x: isinstance(x, np.ndarray) and x.shape == (128,))]

        if valid_embeddings_df.empty:
            raise RuntimeError("No valid skill embeddings found in the dataset for recommendations.")

        skill_embeddings_matrix = np.stack(valid_embeddings_df['skill_name_embedding_vector'].values)
        similarities = cosine_similarity(user_combined_embedding.reshape(1, -1), skill_embeddings_matrix)[0]

        recommendations_df = pd.DataFrame({
            'skill_name': valid_embeddings_df['skill_name'],
            'category': valid_embeddings_df['category'],
            'skill_type': valid_embeddings_df.get('skill_type', pd.Series([""] * len(valid_embeddings_df))),
            'job_demand_score': valid_embeddings_df['job_demand_score'],
            'future_relevance_score': valid_embeddings_df['future_relevance_score'],
            'current_skill_match': valid_embeddings_df['skill_name_lower'].isin([s.lower() for s in user_profile['current_skills']]),
            'similarity_to_user_profile': similarities,
            'difficulty_level_scaled': valid_embeddings_df['difficulty_level_scaled'],
            'industry_usage': valid_embeddings_df.get('industry_usage', pd.Series([[]] * len(valid_embeddings_df))), # Ensure this is a list
            'complementary_skills': valid_embeddings_df.get('complementary_skills', pd.Series([[]] * len(valid_embeddings_df))) # Ensure this is a list
        })

        # Exclude skills the user already has
        recommendations_df = recommendations_df[~recommendations_df['current_skill_match']]

        # Get complementary skills for current user skills using the graph
        complementary_skills_for_current = set()
        if self.complementary_skills_graph:
            for skill in user_profile['current_skills']:
                skill_lower = skill.lower()
                # Find the exact skill name in the graph's nodes (case-sensitive as GML saves it)
                graph_skill_name = next((node for node in self.complementary_skills_graph.nodes if node.lower() == skill_lower), None)
                if graph_skill_name:
                    for neighbor in self.complementary_skills_graph.neighbors(graph_skill_name):
                        complementary_skills_for_current.add(neighbor)
        
        recommendations_df['is_complementary'] = recommendations_df['skill_name'].apply(lambda x: x in complementary_skills_for_current)
        recommendations_df['is_goal_category'] = recommendations_df['category'].isin(relevant_goal_categories)
        
        # Add skill type matching boost
        recommendations_df['matches_user_skill_type'] = recommendations_df['skill_type'].apply(
            lambda x: x in user_skill_types if x else False
        )

        # Calculate initial combined score with skill type boost
        recommendations_df['combined_score'] = (
            recommendations_df['similarity_to_user_profile'] * 0.4 +
            recommendations_df['job_demand_score'] * 0.25 +
            recommendations_df['future_relevance_score'] * 0.15 +
            recommendations_df['is_complementary'] * 0.3 +
            recommendations_df['is_goal_category'] * 0.4 +
            recommendations_df['matches_user_skill_type'] * 0.35  # NEW: Skill type boost
        )

        # Enhanced industry usage boost with multiple goal fields
        boost = 0.15
        threshold = 75
        user_goals_lower = [goal.lower() for goal in user_profile['goals']]

        def enhanced_industry_usage_boost(row):
            # Check industry_usage (which should be a list by now)
            industry_list_lower = [str(item).lower() for item in row['industry_usage']]
            for goal in user_goals_lower:
                if any(fuzz.partial_ratio(goal, industry_item) >= threshold for industry_item in industry_list_lower):
                    return True
            
            # Also check complementary_skills (which should be a list by now)
            comp_list_lower = [str(item).lower() for item in row['complementary_skills']]
            for goal in user_goals_lower:
                if any(fuzz.partial_ratio(goal, comp_item) >= threshold for comp_item in comp_list_lower):
                    return True
            
            return False

        recommendations_df['industry_usage_match'] = recommendations_df.apply(enhanced_industry_usage_boost, axis=1)
        recommendations_df.loc[recommendations_df['industry_usage_match'], 'combined_score'] *= (1 + boost)

        # Normalize combined_score to normalized_score in range [0, 10]
        min_score = recommendations_df['combined_score'].min()
        max_score = recommendations_df['combined_score'].max()
        if max_score > min_score:
            recommendations_df['normalized_score'] = (recommendations_df['combined_score'] - min_score) / (max_score - min_score) * 10
        else:
            recommendations_df['normalized_score'] = 0.0

        recommendations_df = recommendations_df.sort_values(by='normalized_score', ascending=False)

        final_recommendations = []
        seen_skill_names_lower = set()

        # Categorize by difficulty
        basic_skills = []
        medium_skills = []
        hard_skills = []

        for _, row in recommendations_df.iterrows():
            skill_difficulty = self._get_difficulty_level(row['difficulty_level_scaled'])
            if skill_difficulty == 'basic':
                basic_skills.append(row)
            elif skill_difficulty == 'medium':
                medium_skills.append(row)
            else:
                hard_skills.append(row)

        num_basic = int(10 * 0.4)  # 4
        num_medium = int(10 * 0.4) # 4
        num_hard = int(10 * 0.1)   # 1
        num_remaining = 10 - (num_basic + num_medium + num_hard)
        num_medium += num_remaining

        def select_skills_by_difficulty(skill_list, target_count, current_recommendations, current_seen_skills, user_level_score):
            selected_count = 0
            temp_selected = []

            sorted_skills = sorted(
                skill_list,
                key=lambda x: (x['normalized_score'], x['is_complementary'], x['matches_user_skill_type'], -abs(x['difficulty_level_scaled'] - user_level_score)),
                reverse=True
            )

            for row in sorted_skills:
                current_skill_name = row['skill_name']
                current_skill_name_lower = current_skill_name.lower()

                if any(fuzz.token_sort_ratio(current_skill_name_lower, seen) > 90 for seen in current_seen_skills):
                    continue

                if selected_count >= target_count:
                    break

                reason_parts = []
                if row['similarity_to_user_profile'] > 0.7:
                    reason_parts.append(f"Highly relevant to your profile (score: {row['similarity_to_user_profile']:.2f}).")
                elif row['similarity_to_user_profile'] > 0.5:
                    reason_parts.append(f"Relevant to your profile (score: {row['similarity_to_user_profile']:.2f}).")

                if row['job_demand_score'] > 0.8:
                    reason_parts.append("High job demand.")
                elif row['job_demand_score'] > 0.6:
                    reason_parts.append("Good job demand.")

                if row['future_relevance_score'] > 0.8:
                    reason_parts.append("Strong future relevance.")
                elif row['future_relevance_score'] > 0.6:
                    reason_parts.append("Good future relevance.")

                if row['is_complementary']:
                    reason_parts.append("Complements your current skills.")

                if row['is_goal_category']:
                    reason_parts.append(f"Highly relevant to your goal category: {row['category']}.")

                if row['matches_user_skill_type']:
                    reason_parts.append(f"Matches your skill type: {row['skill_type']}.")

                if row.get('industry_usage_match', False):
                    reason_parts.append("Matches your goal in industry usage or complementary skills.")

                if not reason_parts:
                    reason_parts.append("Based on general market trends and your profile.")

                reason = " ".join(reason_parts)
                temp_selected.append({
                    'skill_name': current_skill_name,
                    'score': row['normalized_score'],
                    'reason': reason,
                    'category': row['category'],
                    'skill_type': row.get('skill_type', ''),
                    'difficulty_level_scaled': row['difficulty_level_scaled']
                })

                current_seen_skills.add(current_skill_name_lower)
                selected_count += 1

            return temp_selected

        # Rest of your existing selection logic remains the same...
        selected_basic = select_skills_by_difficulty(basic_skills, num_basic, final_recommendations, seen_skill_names_lower, user_level)
        final_recommendations.extend(selected_basic)

        selected_medium = select_skills_by_difficulty(medium_skills, num_medium, final_recommendations, seen_skill_names_lower, user_level)
        final_recommendations.extend(selected_medium)

        selected_hard = select_skills_by_difficulty(hard_skills, num_hard, final_recommendations, seen_skill_names_lower, user_level)
        final_recommendations.extend(selected_hard)

        # Fill remaining recommendations if fewer than 10 selected
        remaining_to_fill = 10 - len(final_recommendations)
        if remaining_to_fill > 0:
            all_remaining = basic_skills + medium_skills + hard_skills
            all_remaining = sorted(all_remaining, key=lambda x: x['normalized_score'], reverse=True)

            for row in all_remaining:
                if len(final_recommendations) >= 10:
                    break
                skill_lower = row['skill_name'].lower()
                if any(fuzz.token_sort_ratio(skill_lower, seen) > 90 for seen in seen_skill_names_lower):
                    continue

                reason_parts = []
                if row['similarity_to_user_profile'] > 0.7:
                    reason_parts.append(f"Highly relevant to your profile (score: {row['similarity_to_user_profile']:.2f}).")
                elif row['similarity_to_user_profile'] > 0.5:
                    reason_parts.append(f"Relevant to your profile (score: {row['similarity_to_user_profile']:.2f}).")

                if row['job_demand_score'] > 0.8:
                    reason_parts.append("High job demand.")
                elif row['job_demand_score'] > 0.6:
                    reason_parts.append("Good job demand.")

                if row['future_relevance_score'] > 0.8:
                    reason_parts.append("Strong future relevance.")
                elif row['future_relevance_score'] > 0.6:
                    reason_parts.append("Good future relevance.")

                if row['is_complementary']:
                    reason_parts.append("Complements your current skills.")

                if row['is_goal_category']:
                    reason_parts.append(f"Highly relevant to your goal category: {row['category']}.")

                if row['matches_user_skill_type']:
                    reason_parts.append(f"Matches your skill type: {row['skill_type']}.")

                if row.get('industry_usage_match', False):
                    reason_parts.append("Matches your goal in industry usage or complementary skills.")

                if not reason_parts:
                    reason_parts.append("Based on general market trends and your profile.")

                reason = " ".join(reason_parts)
                final_recommendations.append({
                    'skill_name': row['skill_name'],
                    'score': row['normalized_score'],
                    'reason': reason,
                    'category': row['category'],
                    'skill_type': row.get('skill_type', ''),
                    'difficulty_level_scaled': row['difficulty_level_scaled']
                })
                seen_skill_names_lower.add(skill_lower)

        # Remove internal fields from final output
        for rec in final_recommendations:
            rec.pop('difficulty_level_scaled', None)

        return final_recommendations[:20]
    
    def generate_learning_roadmap(
        self, 
        user_profile: Dict[str, Any], 
        target_skills: Optional[List[str]] = None,
        roadmap_length_weeks: int = 12,
        skills_per_phase: int = 3
    ) -> Dict[str, Any]:
        """
        Generate an optimized learning roadmap with complementary skills integration.
        Builds upon existing recommendation system with prerequisite chains and learning phases.
        Leverages loaded prerequisite_graph for dependency resolution.
        """
        if self.df_processed is None or self.model is None or self.prerequisite_graph is None:
            raise RuntimeError("SkillAnalyzer resources (model, data, prerequisite_graph) are not yet loaded for roadmap generation.")
        
        logger.info(f"Generating learning roadmap for {roadmap_length_weeks} weeks")
        
        # Step 1: Get base recommendations using existing system
        base_recommendations = self.recommend_skills(user_profile)
        
        # Step 2: Expand with target skills if provided
        if target_skills:
            for target_skill in target_skills:
                # Check if target skill exists in our dataset
                skill_match = self.df_processed[self.df_processed['skill_name_lower'] == target_skill.lower()]
                if not skill_match.empty and target_skill.lower() not in [rec['skill_name'].lower() for rec in base_recommendations]:
                    # Add target skill with high priority score
                    base_recommendations.insert(0, {
                        'skill_name': skill_match.iloc[0]['skill_name'],
                        'score': 10.0,  # High priority for user-specified targets
                        'reason': f"User-specified target skill: {target_skill}",
                        'category': skill_match.iloc[0]['category'],
                        'difficulty_level_scaled': skill_match.iloc[0]['difficulty_level_scaled'], # Need this for _create_learning_phases
                        'learning_time_days': skill_match.iloc[0]['learning_time_days'] # Need this
                    })
        
        # Step 3: Build prerequisite and complementary skill graph (using loaded graphs)
        # We pass self.prerequisite_graph directly and then filter/enrich it.
        skill_graph_for_roadmap = self._build_skill_dependency_graph_for_roadmap(base_recommendations, user_profile)
        
        # Step 4: Create learning phases based on dependencies and difficulty
        learning_phases = self._create_learning_phases(
            skill_graph_for_roadmap, # Use the prepared graph
            user_profile, 
            roadmap_length_weeks, 
            skills_per_phase
        )
        
        # Step 5: Estimate learning times and create timeline
        roadmap_timeline = self._create_roadmap_timeline(learning_phases, roadmap_length_weeks)
        
        # Step 6: Add learning resources and tips
        enriched_roadmap = self._enrich_roadmap_with_resources(roadmap_timeline)
        
        return {
            'roadmap': enriched_roadmap,
            'total_skills': len([skill for phase in learning_phases for skill in phase['skills']]),
            'total_weeks': roadmap_length_weeks,
            'user_level': self._calculate_user_level(user_profile),
            'roadmap_summary': self._generate_roadmap_summary(learning_phases)
        }

    def _build_skill_dependency_graph_for_roadmap(self, recommendations: List[Dict], user_profile: Dict) -> Dict[str, Dict]:
        """
        Build a graph of skill dependencies for roadmap generation, filtering out already known skills
        and using the loaded prerequisite graph.
        """
        roadmap_skill_info = {}
        current_skills_lower = {skill.lower() for skill in user_profile['current_skills']}
        
        # Create a subgraph from the loaded prerequisite graph containing only recommended skills and their direct prereqs
        # This prevents including every skill in the original large graph if not relevant to roadmap.
        relevant_nodes = set()
        for rec in recommendations:
            relevant_nodes.add(rec['skill_name'])
            # Add its direct prerequisites from the global graph
            if self.prerequisite_graph and rec['skill_name'] in self.prerequisite_graph:
                for prereq in self.prerequisite_graph.predecessors(rec['skill_name']):
                    relevant_nodes.add(prereq)
        
        # Filter the global graph or build a new one based on relevant_nodes
        if self.prerequisite_graph:
            # Ensure all nodes in relevant_nodes are actual nodes in the graph
            # Some 'prereq' skills might not be in the original df_processed if they are external or newly added
            # Filter relevant_nodes to only include those present in the actual graph nodes
            filtered_relevant_nodes = [node for node in relevant_nodes if node in self.prerequisite_graph.nodes]
            subgraph = self.prerequisite_graph.subgraph(filtered_relevant_nodes).copy()
        else:
            subgraph = nx.DiGraph()
            subgraph.add_nodes_from([rec['skill_name'] for rec in recommendations])

        # Populate roadmap_skill_info based on recommendations and graph data
        for rec in recommendations:
            skill_name = rec['skill_name']
            skill_data_row = self.df_processed[self.df_processed['skill_name'] == skill_name]
            
            if skill_data_row.empty:
                logging.warning(f"Skill '{skill_name}' from recommendations not found in df_processed for roadmap. Skipping.")
                continue

            skill_data_row = skill_data_row.iloc[0]

            # Determine prerequisites from the loaded graph, filtering out learned skills
            prerequisites_for_roadmap = []
            if skill_name in subgraph:
                for prereq in subgraph.predecessors(skill_name):
                    # Check if the prerequisite is an actual skill in our processed data
                    if prereq.lower() not in current_skills_lower and prereq in self.df_processed['skill_name'].values:
                        prerequisites_for_roadmap.append(prereq)
                    # Handle external prerequisites (not in our dataset, but still a dependency)
                    elif prereq.lower() not in current_skills_lower: # and prereq not in self.df_processed['skill_name'].values:
                        prerequisites_for_roadmap.append(prereq) # Include external prereqs as dependencies


            # Determine complementary skills from the loaded graph
            complementary_skills_for_roadmap = []
            if self.complementary_skills_graph and skill_name in self.complementary_skills_graph:
                for comp_skill in self.complementary_skills_graph.neighbors(skill_name):
                    # Only add complementary skills that are also in our recommendations or present in processed data
                    if comp_skill.lower() not in current_skills_lower and comp_skill.lower() != skill_name.lower() \
                       and comp_skill in self.df_processed['skill_name'].values: # ensure it's a "learnable" skill
                        complementary_skills_for_roadmap.append(comp_skill)
            
            roadmap_skill_info[skill_name] = {
                'prerequisites': prerequisites_for_roadmap,
                'complementary': complementary_skills_for_roadmap,
                'difficulty': skill_data_row.get('difficulty_level_scaled', 0.5),
                'learning_time': skill_data_row.get('learning_time_days', 30), # Assuming this is raw days
                'category': skill_data_row.get('category', ''),
                'score': rec['score'],
                'reason': rec['reason']
            }
        
        return roadmap_skill_info

    def _create_learning_phases(
        self, 
        skill_graph: Dict, 
        user_profile: Dict, 
        total_weeks: int, 
        skills_per_phase: int
    ) -> List[Dict]:
        """
        Create learning phases respecting prerequisites and balancing difficulty.
        """
        phases = []
        learned_skills = set(skill.lower() for skill in user_profile['current_skills'])
        remaining_skills_names = set(skill_graph.keys())
        
        phase_number = 1
        weeks_per_phase = max(2, total_weeks // 6)  # Aim for ~6 phases
        
        while remaining_skills_names and phase_number <= 6:  # Max 6 phases to prevent infinite loops
            phase_skills_list = []
            
            # Find skills whose prerequisites are already learned or are external (not in our graph)
            available_skills_for_current_phase = []
            for skill_name in remaining_skills_names:
                skill_info = skill_graph[skill_name]
                prereqs_met = True
                for prereq in skill_info['prerequisites']:
                    # A prerequisite is met if it's learned, OR if it's external (not a node in our roadmap_skill_info graph)
                    if (prereq in skill_graph and prereq.lower() not in learned_skills) or \
                       (prereq not in self.df_processed['skill_name'].values and prereq.lower() not in learned_skills):
                        prereqs_met = False
                        break
                if prereqs_met:
                    available_skills_for_current_phase.append((skill_name, skill_info))
            
            # If no skills are available based on prerequisites, this indicates a potential issue
            # or all remaining skills have unmet external prerequisites.
            if not available_skills_for_current_phase and remaining_skills_names:
                logging.warning(f"No more skills with met prerequisites in phase {phase_number}. Remaining: {list(remaining_skills_names)}")
                # Break if no progress can be made
                break 

            # Sort by score (higher is better) and then difficulty (lower is better for earlier phases)
            available_skills_for_current_phase.sort(key=lambda x: (-x[1]['score'], x[1]['difficulty']))
            
            # Select skills for this phase
            selected_count = 0
            temp_skills_added_this_phase = set() # To prevent adding duplicates within a phase

            for skill_name, skill_info in available_skills_for_current_phase:
                if selected_count >= skills_per_phase:
                    break
                
                if skill_name.lower() in learned_skills or skill_name.lower() in temp_skills_added_this_phase: # Prevent duplicates
                    continue

                phase_skills_list.append({
                    'skill_name': skill_name,
                    'difficulty': self._get_difficulty_level(skill_info['difficulty']),
                    'estimated_days': int(round(float(skill_info['learning_time']) * (365/100))), # Inverse scale learning time back to days (assuming 0-1 mapped to 0-100)
                    'category': skill_info['category'],
                    'prerequisites': skill_info['prerequisites'],
                    'complementary': skill_info['complementary'],
                    'reason': skill_info['reason']
                })
                
                learned_skills.add(skill_name.lower())
                temp_skills_added_this_phase.add(skill_name.lower())
                remaining_skills_names.remove(skill_name)
                selected_count += 1
                
                # Add highly relevant complementary skills if space allows (from the loaded graph)
                # Only if the complementary skill is also in the overall skill_graph (i.e., it's a known skill in our dataset)
                # and not already learned or added to this phase.
                if self.complementary_skills_graph:
                    for comp_skill_name_candidate in skill_info['complementary']:
                        if (selected_count < skills_per_phase and
                            comp_skill_name_candidate in skill_graph and # Must be a skill in our current roadmap scope
                            comp_skill_name_candidate.lower() not in learned_skills and
                            comp_skill_name_candidate.lower() not in temp_skills_added_this_phase):
                            
                            comp_info = skill_graph[comp_skill_name_candidate]
                            phase_skills_list.append({
                                'skill_name': comp_skill_name_candidate,
                                'difficulty': self._get_difficulty_level(comp_info['difficulty']),
                                'estimated_days': int(round(float(comp_info['learning_time']) * (365/100))),
                                'category': comp_info['category'],
                                'prerequisites': comp_info['prerequisites'],
                                'complementary': comp_info['complementary'],
                                'reason': f"Complementary to {skill_name}: {comp_info['reason']}"
                            })
                            
                            learned_skills.add(comp_skill_name_candidate.lower())
                            temp_skills_added_this_phase.add(comp_skill_name_candidate.lower())
                            if comp_skill_name_candidate in remaining_skills_names:
                                remaining_skills_names.remove(comp_skill_name_candidate)
                            selected_count += 1
            
            if phase_skills_list: # Check if any skills were added in this phase
                phases.append({
                    'phase_number': phase_number,
                    'start_week': (phase_number - 1) * weeks_per_phase + 1,
                    'end_week': phase_number * weeks_per_phase,
                    'skills': phase_skills_list,
                    'estimated_weeks': weeks_per_phase,
                    'focus': self._determine_phase_focus(phase_skills_list)
                })
                phase_number += 1
            else:
                break  # No more skills could be added to any phase
        
        return phases

    def _create_roadmap_timeline(self, phases: List[Dict], total_weeks: int) -> Dict:
        """
        Create a timeline with week-by-week breakdown.
        """
        timeline = []
        current_week = 1
        
        for phase in phases:
            # Ensure cumulative weeks don't exceed total_weeks
            phase_duration = min(phase['estimated_weeks'], total_weeks - current_week + 1)
            
            phase_timeline = {
                'phase_number': phase['phase_number'],
                'start_week': current_week,
                'end_week': current_week + phase_duration - 1,
                'skills': phase['skills'],
                'focus': phase['focus'],
                'milestones': self._generate_phase_milestones(phase['skills'], phase_duration)
            }
            
            timeline.append(phase_timeline)
            current_week += phase_duration
            
            if current_week > total_weeks:
                break
        
        return {
            'phases': timeline,
            'total_duration_weeks': min(current_week - 1, total_weeks)
        }

    def _enrich_roadmap_with_resources(self, timeline: Dict) -> Dict:
        """
        Add learning resources and practical tips to each phase.
        """
        for phase in timeline['phases']:
            phase['learning_tips'] = []
            phase['practice_projects'] = []
            
            # Generate tips based on skills in phase
            categories = set(skill['category'] for skill in phase['skills'])
            difficulties = [skill['difficulty'] for skill in phase['skills']]
            
            # Add difficulty-appropriate tips
            if 'basic' in difficulties:
                phase['learning_tips'].append("Start with fundamentals and practice daily for 30-60 minutes")
            if 'medium' in difficulties:
                phase['learning_tips'].append("Build projects to apply concepts and join relevant communities")
            if 'hard' in difficulties:
                phase['learning_tips'].append("Focus on one complex skill at a time and seek mentorship")
                
            # Add category-specific project ideas
            if any('frontend' in cat.lower() or 'web' in cat.lower() for cat in categories):
                phase['practice_projects'].append("Build a responsive web application")
            if any('backend' in cat.lower() or 'server' in cat.lower() for cat in categories):
                phase['practice_projects'].append("Create a REST API with database integration")
            if any('data' in cat.lower() for cat in categories):
                phase['practice_projects'].append("Analyze a real dataset and create visualizations")
        
        return timeline

    def _determine_phase_focus(self, skills: List[Dict]) -> str:
        """
        Determine the main focus theme for a learning phase.
        """
        categories = [skill['category'] for skill in skills]
        difficulties = [skill['difficulty'] for skill in skills]
        
        # Most common category
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        main_category = max(category_counts, key=category_counts.get)
        difficulty_level = max(set(difficulties), key=difficulties.count)
        
        return f"{difficulty_level.title()} {main_category} Skills"

    def _generate_phase_milestones(self, skills: List[Dict], weeks: int) -> List[str]:
        """
        Generate milestone checkpoints for a phase.
        """
        milestones = []
        skills_per_milestone = max(1, len(skills) // max(1, weeks // 2))
        
        for i in range(0, len(skills), skills_per_milestone):
            batch = skills[i:i + skills_per_milestone]
            milestone = f"Complete {', '.join(skill['skill_name'] for skill in batch)}"
            milestones.append(milestone)
        
        return milestones

    def _calculate_user_level(self, user_profile: Dict) -> str:
        """
        Calculate user's overall skill level.
        """
        if not user_profile.get('current_skills'):
            return 'beginner'
            
        difficulty_scores = []
        for skill in user_profile['current_skills']:
            skill_data = self.df_processed[self.df_processed['skill_name_lower'] == skill.lower()]
            if not skill_data.empty:
                difficulty_scores.append(skill_data.iloc[0].get('difficulty_level_scaled', 0.5))
        
        if not difficulty_scores:
            return 'beginner'
            
        avg_difficulty = np.mean(difficulty_scores)
        
        if avg_difficulty <= 0.33:
            return 'beginner'
        elif avg_difficulty <= 0.66:
            return 'intermediate'
        else:
            return 'advanced'

    def _generate_roadmap_summary(self, phases: List[Dict]) -> Dict:
        """
        Generate a summary of the learning roadmap.
        """
        total_skills = sum(len(phase['skills']) for phase in phases)
        categories = set()
        difficulties = []
        
        for phase in phases:
            for skill in phase['skills']:
                categories.add(skill['category'])
                difficulties.append(skill['difficulty'])
        
        return {
            'total_phases': len(phases),
            'total_skills': total_skills,
            'categories_covered': list(categories),
            'difficulty_distribution': {
                'basic': difficulties.count('basic'),
                'medium': difficulties.count('medium'),
                'hard': difficulties.count('hard')
            },
            'estimated_total_time': f"{sum(phase.get('estimated_weeks', 0) for phase in phases)} weeks"
        }
    def compute_user_matching_score(self, userA: dict, userB: dict) -> dict:
        """
        Compute a compatibility score and explanation between two users.
        Considers:
        - Strict reciprocal teach/learn matching,
        - Category and goal overlap (via fuzzy matching),
        - Category alignment of teach/learn skills,
        - Experience/expertise similarity.
        """
        a_learn = userA.get("want_to_learn", "").strip().lower()
        a_teach = userA.get("can_teach", "").strip().lower()
        b_learn = userB.get("want_to_learn", "").strip().lower()
        b_teach = userB.get("can_teach", "").strip().lower()
        a_goals = [g.lower() for g in userA.get("goals", [])]
        b_goals = [g.lower() for g in userB.get("goals", [])]
        a_known = set(skill.lower() for skill in userA.get("current_skills", []))
        b_known = set(skill.lower() for skill in userB.get("current_skills", []))

        # 1. Strict reciprocal teach/learn
        strict_score = 1.0 if (a_learn == b_teach and b_learn == a_teach) else 0.0

        # 2. Category/goal fuzzy overlap score
        goal_overlap = 0.0
        if a_goals and b_goals:
            scores = [fuzz.partial_ratio(ga, gb) for ga in a_goals for gb in b_goals]
            goal_overlap = max(scores) / 100.0 if scores else 0.0

        # 3. Category alignment helper
        def find_category(skill):
            df = self.df_processed
            row = df[df["skill_name"].str.lower() == skill]
            return row.iloc[0]["category"].lower() if not row.empty else ""
        a_cat_learn = find_category(a_learn)
        a_cat_teach = find_category(a_teach)
        b_cat_learn = find_category(b_learn)
        b_cat_teach = find_category(b_teach)

        cat_match = 0.0
        if a_cat_learn and a_cat_learn == b_cat_teach:
            cat_match += 0.5
        if b_cat_learn and b_cat_learn == a_cat_teach:
            cat_match += 0.5

        # 4. Expertise/level proximity (basic heuristic)
        def get_difficulty(skill):
            df = self.df_processed
            row = df[df['skill_name'].str.lower() == skill]
            if not row.empty:
                return row.iloc[0].get('difficulty_level_scaled', 0.5)
            return 0.5
        a_learn_diff = get_difficulty(a_learn)
        b_teach_diff = get_difficulty(b_teach)
        b_learn_diff = get_difficulty(b_learn)
        a_teach_diff = get_difficulty(a_teach)

        # Check if learner skill difficulty is less than teacher skill difficulty as a proxy for expertise fit
        expertise_score = 1.0 if (a_learn_diff <= b_teach_diff and b_learn_diff <= a_teach_diff) else 0.5

        # Final weighted score (weights can be tuned)
        final_score = round(
            0.4 * strict_score +
            0.2 * cat_match +
            0.2 * goal_overlap +
            0.2 * expertise_score,
            3
        )

        reasons = []
        if strict_score == 1.0:
            reasons.append("Perfect reciprocal teach-learn match.")
        if cat_match > 0.0:
            reasons.append(f"Skill category alignment ({cat_match:.2f}).")
        if goal_overlap > 0:
            reasons.append(f"Shared or related goals ({goal_overlap:.2f} fuzzy overlap).")
        if expertise_score == 1.0:
            reasons.append("Good expertise complementarity.")
        if not reasons:
            reasons.append("Profiles have limited alignment.")

        return {
            "score": final_score,
            "reciprocal_teach_learn": strict_score,
            "category_match": cat_match,
            "goal_overlap_score": goal_overlap,
            "expertise_score": expertise_score,
            "explanation": " | ".join(reasons)
        }

    def match_peer(self, user_profile: dict, peer_profiles: list) -> list:
        """
        Matches one user against multiple peers.
        Returns a list of peers with matching scores and detailed breakdown.
        """
        matches = []
        for peer in peer_profiles:
            score_info = self.compute_user_matching_score(user_profile, peer)
            match = dict(peer)
            match["matching_details"] = score_info
            matches.append(match)
        # Sort by descending score
        matches.sort(key=lambda x: x["matching_details"]["score"], reverse=True)
        return matches


    def calculate_user_market_position(self, user_profile: dict) -> dict:
        """
        Calculate comprehensive market position score for a user based on their skills and goals.
        """
        current_skills = [skill.lower() for skill in user_profile.get('current_skills', [])]
        goals = [goal.lower() for goal in user_profile.get('goals', [])]
        
        if not current_skills:
            return {
                "overall_score": 0.0,
                "position_tier": "Entry Level",
                "detailed_breakdown": {},
                "recommendations": ["Start building foundational skills to establish your market position"],
                "market_percentile": 0,
                "strengths": [],
                "improvement_areas": ["Begin with foundational skills in your area of interest"]
            }
        
        # Calculate component scores
        portfolio_score = self._calculate_portfolio_score(current_skills)
        goal_alignment_score = self._calculate_goal_alignment_score(goals, current_skills)
        diversity_score = self._calculate_skill_diversity_score(current_skills)
        market_demand_score = self._calculate_market_demand_score(current_skills)
        future_readiness_score = self._calculate_future_readiness_score(current_skills, goals)
        
        # Calculate weighted overall score
        overall_score = (
            portfolio_score * 0.40 +
            goal_alignment_score * 0.25 +
            diversity_score * 0.20 +
            market_demand_score * 0.10 +
            future_readiness_score * 0.05
        )
        
        # Determine tier and percentile
        if overall_score >= 0.9:
            position_tier, market_percentile = "Market Leader", 95
        elif overall_score >= 0.8:
            position_tier, market_percentile = "Senior Expert", 85
        elif overall_score >= 0.7:
            position_tier, market_percentile = "Experienced Professional", 75
        elif overall_score >= 0.6:
            position_tier, market_percentile = "Mid-Level Professional", 60
        elif overall_score >= 0.4:
            position_tier, market_percentile = "Junior Professional", 40
        elif overall_score >= 0.2:
            position_tier, market_percentile = "Entry Level", 20
        else:
            position_tier, market_percentile = "Beginner", 5
        
        # Generate recommendations
        recommendations = []
        if portfolio_score < 0.6:
            recommendations.append("Focus on developing high-demand, high-value skills in your domain")
        if goal_alignment_score < 0.5:
            recommendations.append("Align your skill development more closely with your stated career goals")
        if diversity_score < 0.5:
            recommendations.append("Diversify your skill portfolio across complementary areas")
        if market_demand_score < 0.6:
            recommendations.append("Consider learning skills with higher market demand")
        if future_readiness_score < 0.6:
            recommendations.append("Invest in future-relevant technologies like AI, cloud computing, or automation")
        if not recommendations:
            recommendations.append("Maintain your strong market position by staying updated with industry trends")
        
        # Identify strengths
        strengths = self._identify_strengths(current_skills)
        improvement_areas = self._identify_improvement_areas(current_skills, goals)
        
        return {
            "overall_score": round(overall_score, 2),
            "position_tier": position_tier,
            "market_percentile": market_percentile,
            "detailed_breakdown": {
                "portfolio_score": round(portfolio_score, 2),
                "goal_alignment_score": round(goal_alignment_score, 2),
                "diversity_score": round(diversity_score, 2),
                "market_demand_score": round(market_demand_score, 2),
                "future_readiness_score": round(future_readiness_score, 2)
            },
            "recommendations": recommendations,
            "strengths": strengths,
            "improvement_areas": improvement_areas
        }


    def _calculate_portfolio_score(self, current_skills: list) -> float:
        """Calculate score based on quality and relevance of current skills using skill analysis"""
        if not current_skills:
            return 0.0
        
        skill_scores = []
        for skill in current_skills:
            try:
                # Use your existing skill analysis method
                skill_data = self.perform_skill_analysis(skill)
                
                # Extract scores from the analyzed skill data
                # ENSURE a numeric default (e.g., 5.0 or 0.5) if predicted_ score is None
                # or if original score is None.
                job_demand = skill_data.get('predicted_job_demand_score') # Get predicted, which might be None
                if job_demand is None:
                    job_demand = skill_data.get('job_demand_score', 5.0) # Fallback to original, then default
                job_demand = float(job_demand) / 10.0 # Scale to 0-1 from 0-10

                salary_impact = skill_data.get('predicted_salary_impact_percent')
                if salary_impact is None:
                    salary_impact = skill_data.get('salary_impact_percent', 50)
                salary_impact = float(salary_impact) / 100.0  # Convert to 0-1 scale

                future_relevance = skill_data.get('predicted_future_relevance_score')
                if future_relevance is None:
                    future_relevance = skill_data.get('future_relevance_score', 5.0)
                future_relevance = float(future_relevance) / 10.0 # Scale to 0-1 from 0-10
                
                # Combine the scores with weights
                skill_score = (
                    job_demand * 0.4 +
                    salary_impact * 0.3 +
                    future_relevance * 0.3
                )
                skill_scores.append(skill_score)
                
            except (ValueError, RuntimeError) as e:
                # If skill analysis fails or individual score extraction fails, use default score
                logger.warning(f"Could not analyze skill '{skill}' for portfolio score: {e}")
                skill_scores.append(0.3)  # Default for unknown/unanalyzable skills
            except TypeError as e: # Catch if float() conversion on None fails more directly
                logger.error(f"TypeError during score conversion for skill '{skill}': {e}. Value was likely None unexpectedly.")
                skill_scores.append(0.3)
        
        return np.mean(skill_scores) if skill_scores else 0.0
    def _calculate_goal_alignment_score(self, goals: list, current_skills: list) -> float:
        """Calculate how well current skills align with stated goals using skill analysis"""
        if not goals:
            return 0.5  # Neutral score if no goals specified
        
        alignment_scores = []
        relevant_goal_categories = self._get_relevant_categories_from_goals(goals)
        
        for skill in current_skills:
            try:
                skill_data = self.perform_skill_analysis(skill)
                skill_category = skill_data.get('category', '')
                
                if skill_category in relevant_goal_categories:
                    alignment_scores.append(1.0)
                else:
                    # Check industry usage for partial alignment
                    industry_text = skill_data.get('industry_usage_text', '').lower()
                    partial_match = any(
                        fuzz.partial_ratio(goal, industry_text) >= 70 
                        for goal in goals
                    )
                    # Check for partial_match (boolean Series)
                    if isinstance(partial_match, pd.Series): # Add this check for safety
                        partial_match = partial_match.any() # Take .any() if it's a Series

                    alignment_scores.append(0.7 if partial_match else 0.3)
                    
            except (ValueError, RuntimeError):
                alignment_scores.append(0.3)
        
        return np.mean(alignment_scores) if alignment_scores else 0.0

    def _calculate_skill_diversity_score(self, current_skills: list) -> float:
        """Calculate diversity and complementarity using skill analysis and complementary graph"""
        if len(current_skills) < 2:
            return 0.3  # Low diversity for single skills
        
        categories = set()
        skill_types = set()
        complementary_bonus = 0
        
        for skill in current_skills:
            try:
                skill_data = self.perform_skill_analysis(skill)
                
                categories.add(skill_data.get('category', ''))
                skill_types.add(skill_data.get('skill_type', ''))
                
                # Check for complementary skills using the loaded graph
                if self.complementary_skills_graph and skill in self.complementary_skills_graph: # Check if skill is a node in the graph
                    for comp_skill in self.complementary_skills_graph.neighbors(skill):
                        if comp_skill.lower() in [s.lower() for s in current_skills]:
                            complementary_bonus += 0.1 # Small bonus for having complementary skills together
                
            except (ValueError, RuntimeError):
                continue  # Skip skills that can't be analyzed
        
        # Diversity score based on category and type spread
        category_diversity = min(len(categories) / 3, 1.0)  # Max score at 3+ categories
        type_diversity = min(len(skill_types) / 2, 1.0)    # Max score at 2+ types
        
        diversity_score = (category_diversity * 0.6 + type_diversity * 0.4) + min(complementary_bonus, 0.3)
        return min(diversity_score, 1.0)

    def _calculate_market_demand_score(self, current_skills: list) -> float:
        """Calculate overall market demand using skill analysis"""
        demand_scores = []
        
        for skill in current_skills:
            try:
                skill_data = self.perform_skill_analysis(skill)
                demand_scores.append(skill_data.get('predicted_job_demand_score', skill_data.get('job_demand_score', 5.0)) / 10.0) # Use predicted if available, scale to 0-1
            except (ValueError, RuntimeError):
                demand_scores.append(0.3)
        
        return np.mean(demand_scores) if demand_scores else 0.0

    def _calculate_future_readiness_score(self, current_skills: list, goals: list) -> float:
        """Calculate readiness for future market trends using skill analysis"""
        future_scores = []
        
        # Check future relevance of current skills
        for skill in current_skills:
            try:
                skill_data = self.perform_skill_analysis(skill)
                future_scores.append(skill_data.get('predicted_future_relevance_score', skill_data.get('future_relevance_score', 5.0)) / 10.0) # Use predicted if available, scale to 0-1
            except (ValueError, RuntimeError):
                future_scores.append(0.3)
        
        # Bonus for forward-thinking goals
        future_keywords = ['ai', 'machine learning', 'cloud', 'blockchain', 'automation', 'data science']
        goal_future_bonus = sum(
            0.1 for goal in goals 
            if any(keyword in goal for keyword in future_keywords)
        )
        
        base_score = np.mean(future_scores) if future_scores else 0.0
        return min(base_score + goal_future_bonus, 1.0)

    def _identify_strengths(self, current_skills: list) -> list:
        """Identify user's market strengths using skill analysis"""
        strengths = []
        high_value_skills = []
        
        for skill in current_skills:
            try:
                skill_data = self.perform_skill_analysis(skill)
                # Use predicted if available, compare to 7 (out of 10)
                if skill_data.get('predicted_job_demand_score', skill_data.get('job_demand_score', 0)) > 7:
                    high_value_skills.append(skill.title())
            except (ValueError, RuntimeError):
                continue
        
        if high_value_skills:
            strengths.append(f"Strong expertise in high-demand skills: {', '.join(high_value_skills)}")
        
        categories = set()
        for skill in current_skills:
            try:
                skill_data = self.perform_skill_analysis(skill)
                categories.add(skill_data.get('category', ''))
            except (ValueError, RuntimeError):
                continue
        
        if len(categories) >= 3:
            strengths.append("Diverse skill portfolio across multiple domains")
        
        return strengths if strengths else ["Building foundational skills"]


    def _identify_improvement_areas(self, current_skills: list, goals: list) -> list:
        """Identify areas for improvement"""
        improvements = []
        
        # Get recommended skills based on goals
        if goals:
            user_profile = {'current_skills': current_skills, 'goals': goals}
            recommendations = self.recommend_skills(user_profile)
            
            if recommendations:
                top_gap_skills = [rec['skill_name'] for rec in recommendations[:3]]
                improvements.append(f"Consider learning: {', '.join(top_gap_skills)}")
        
        # Check for skill depth vs breadth
        if len(current_skills) > 5:
            improvements.append("Focus on deepening expertise in your strongest skills")
        elif len(current_skills) < 3:
            improvements.append("Expand your skill portfolio for better market positioning")
        
        return improvements if improvements else ["Continue building on your current trajectory"]