# src/models/temp.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text
import logging
from typing import Dict, Any
from fuzzywuzzy import fuzz # For fuzzy string matching (typo correction)
from sklearn.metrics.pairwise import cosine_similarity # For semantic similarity using BERT embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration (must match your training/model build script) ---
MODEL_SAVE_PATH = 'models/skill_intelligence_model.keras'
PROCESSED_DATA_PATH = 'data/processed/encoded_features_for_model.parquet' # Path to your processed data
BERT_MODEL_URL = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1"
BERT_PREPROCESS_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

# Define the features that your model expects, based on neural_architecture.py
NUMERICAL_FEATURES = [
    'learning_time_days', 'popularity_score', 'job_demand_score',
    'salary_impact_percent', 'future_relevance_score', 'learning_resources_quality',
    'skill_complexity_score', 'market_momentum_score', 'ecosystem_richness',
    'industry_diversity_metric', 'resource_availability_index', 'learning_accessibility_score'
]

# Corrected CATEGORICAL_FEATURES_MAPPING based on neural_architecture.py inputs
CATEGORICAL_FEATURES_MAPPING = {
    'skill_category_encoded': 'skill_category_encoded_input',
    'skill_type_encoded': 'skill_type_encoded_input',
    'market_trend_encoded': 'market_trend_encoded_input'
}

# Corrected TEXT_FEATURES_MAPPING based on neural_architecture.py inputs
TEXT_FEATURES_MAPPING = {
    'skill_name': 'skill_name_embedding_input',
    'prerequisites': 'prerequisites_embedding_input',
    'complementary_skills': 'complementary_skills_embedding_input',
    'industry_usage_text': 'industry_embedding_input'
}

REGRESSION_TARGETS = ['difficulty_level_scaled']
BINARY_CLASSIFICATION_TARGETS = ['certification_available']


def load_bert_models():
    """Loads BERT preprocessor and encoder from TensorFlow Hub."""
    try:
        logging.info(f"Loading BERT preprocessor from: {BERT_PREPROCESS_URL}")
        preprocess_model = hub.KerasLayer(BERT_PREPROCESS_URL)
        logging.info(f"Loading BERT encoder from: {BERT_MODEL_URL}")
        encoder_model = hub.KerasLayer(BERT_MODEL_URL)
        return preprocess_model, encoder_model
    except Exception as e:
        logging.error(f"Error loading BERT models: {e}")
        logging.error("Please ensure you have an active internet connection or the models are cached locally.")
        raise

def prepare_model_input_from_dataframe_row(row: pd.Series, bert_preprocess_model, bert_encoder_model) -> Dict[str, np.ndarray]:
    """
    Perpares input dictionary for the Keras model from a single DataFrame row.
    This mimics the input structure expected by build_skill_intelligence_model in neural_architecture.py.
    """
    model_input = {}

    # Numerical features
    numerical_values = []
    for feature in NUMERICAL_FEATURES:
        if feature in row:
            numerical_values.append(row[feature])
        else:
            logging.warning(f"Numerical feature '{feature}' not found in row. Using 0.0.")
            numerical_values.append(0.0)
    model_input['numerical_features_input'] = np.array([numerical_values]).astype(np.float32)

    # Categorical features
    for encoded_col, model_input_name in CATEGORICAL_FEATURES_MAPPING.items():
        if encoded_col in row:
            model_input[model_input_name] = np.array([int(row[encoded_col])]).reshape(1, 1).astype(np.int32)
        else:
            logging.warning(f"Categorical feature '{encoded_col}' not found in row. Using 0 for '{model_input_name}'.")
            model_input[model_input_name] = np.array([0]).reshape(1, 1).astype(np.int32)

    # Text embeddings
    for original_text_feature, model_input_name in TEXT_FEATURES_MAPPING.items():
        text_value = str(row.get(original_text_feature, ''))
        logging.debug(f"Processing text input for {original_text_feature}: '{text_value}'")
        text_preprocessed = bert_preprocess_model([text_value])
        text_embedding = bert_encoder_model(text_preprocessed)['pooled_output']
        model_input[model_input_name] = text_embedding

    return model_input

def process_and_predict_skill(selected_skill_row, bert_preprocess_model, bert_encoder_model, model):
    """Helper function to encapsulate the prediction logic for a given skill row."""
    logging.info(f"\n--- Data for '{selected_skill_row['skill_name']}' ---")
    display_features = ['skill_name', 'category', 'skill_type', 'market_trend',
                        'learning_time_days', 'job_demand_score',
                        'salary_impact_percent', 'future_relevance_score',
                        'certification_available', 'prerequisites', 'complementary_skills',
                        'industry_usage_text',
                        'difficulty_level_scaled']

    for feature in display_features:
        if feature in selected_skill_row:
            logging.info(f"  {feature.replace('_', ' ').title()}: {selected_skill_row[feature]}")

    model_input_data = prepare_model_input_from_dataframe_row(
        selected_skill_row, bert_preprocess_model, bert_encoder_model
    )

    logging.info("\nMaking predictions for this skill...")
    predictions = model.predict(model_input_data)

    if isinstance(predictions, dict):
        logging.info("Model returned predictions as a dictionary.")
        regression_output_key = 'regression_outputs'
        binary_output_key = 'binary_classification_outputs'

        if regression_output_key in predictions:
            reg_preds = predictions[regression_output_key]
            logging.info(f"Regression Output (Shape: {reg_preds.shape}):")
            regression_targets_names = [
                'Future Relevance Score', 'Salary Impact Percent',
                'Job Demand Score', 'Learning Time Days'
            ]

            for i, target_name in enumerate(regression_targets_names):
                if i < reg_preds.shape[1]:
                    logging.info(f"  Predicted {target_name}: {reg_preds[0][i]:.4f}")

        if binary_output_key in predictions:
            bin_preds = predictions[binary_output_key]
            logging.info(f"Binary Classification Output (Shape: {bin_preds.shape}):")
            binary_probs = tf.nn.sigmoid(bin_preds).numpy()[0][0]
            logging.info(f"  Predicted Probability (Certification Available): {binary_probs:.4f}")
            logging.info(f"  Predicted Class (Certification Available): {'Yes (Certified)' if binary_probs > 0.5 else 'No (Not Certified)'}")
    elif isinstance(predictions, list):
        logging.info("Model returned predictions as a list.")
        if len(predictions) >= 1:
            logging.info(f"First Output (Shape: {predictions[0].shape}):")
            regression_targets_names = [
                'Future Relevance Score', 'Salary Impact Percent',
                'Job Demand Score', 'Learning Time Days'
            ]
            for i, target_name in enumerate(regression_targets_names):
                if i < predictions[0].shape[1]:
                    logging.info(f"  Predicted {target_name}: {predictions[0][0][i]:.4f}")
            if 'difficulty_level_scaled' in REGRESSION_TARGETS and predictions[0].shape[1] > 0:
                logging.info(f"  Predicted Difficulty Level Scaled: {predictions[0][0][0]:.4f}")

        if len(predictions) >= 2:
            logging.info(f"Second Output (Shape: {predictions[1].shape}):")
            binary_probs = tf.nn.sigmoid(predictions[1]).numpy()[0][0]
            logging.info(f"  Predicted Probability (Certification Available): {binary_probs:.4f}")
            logging.info(f"  Predicted Class (Certification Available): {'Yes (Certified)' if binary_probs > 0.5 else 'No (Not Certified)'}")
    else:
        logging.info(f"Single Output (Shape: {predictions.shape}):")
        logging.info(predictions[0])


def run_interactive_prediction():
    """
    Loads the trained model and data, then allows interactive skill lookup and prediction.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        logging.error(f"Model not found at {MODEL_SAVE_PATH}. Please train the model first by running training_pipeline.py.")
        return

    if not os.path.exists(PROCESSED_DATA_PATH):
        logging.error(f"Processed data not found at {PROCESSED_DATA_PATH}. Please run your data processing pipeline first (e.g., data_encoder.py).")
        return

    try:
        logging.info(f"Loading trained model from {MODEL_SAVE_PATH}...")
        custom_objects = {'KerasLayer': hub.KerasLayer}
        model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects=custom_objects)
        logging.info("Model loaded successfully.")

        logging.info(f"Loading processed data from {PROCESSED_DATA_PATH}...")
        df_processed = pd.read_parquet(PROCESSED_DATA_PATH)
        df_processed['skill_name_lower'] = df_processed['skill_name'].str.lower()
        logging.info(f"Data loaded successfully. Total skills: {len(df_processed)}")

        # Load BERT models for preprocessing input text
        bert_preprocess_model, bert_encoder_model = load_bert_models()

        logging.info("Generating BERT embeddings for all skill names in the dataset (this might take a moment)...")
        # Pre-calculate embeddings for all skill names to speed up similarity search
        all_skill_names = df_processed['skill_name'].tolist()
        processed_texts = bert_preprocess_model(tf.constant(all_skill_names))
        skill_embeddings = bert_encoder_model(processed_texts)['pooled_output'].numpy()
        df_processed['skill_name_embedding_vector'] = list(skill_embeddings)
        logging.info("Skill name embeddings generated.")

        logging.info("\n--- Interactive Skill Prediction ---")
        logging.info("Enter a skill name to get its data and prediction. Type 'exit' to quit.")

        while True:
            skill_input = input("\nEnter Skill Name: ").strip().lower()
            if skill_input == 'exit':
                logging.info("Exiting interactive prediction.")
                break

            found_skills = df_processed[df_processed['skill_name_lower'] == skill_input]

            if not found_skills.empty:
                process_and_predict_skill(found_skills.iloc[0], bert_preprocess_model, bert_encoder_model, model)
            else:
                logging.info(f"Skill '{skill_input}' not found in the dataset. Searching for similar/related skills...")
                
                # --- BERT-based Semantic Similarity Search ---
                input_skill_processed = bert_preprocess_model([skill_input])
                input_skill_embedding = bert_encoder_model(input_skill_processed)['pooled_output'].numpy()

                # Calculate cosine similarity between input embedding and all pre-calculated skill embeddings
                similarities_bert = cosine_similarity(input_skill_embedding, np.stack(df_processed['skill_name_embedding_vector'].values))
                
                bert_similarity_df = pd.DataFrame({
                    'skill_name': df_processed['skill_name'],
                    'similarity_score': similarities_bert[0]
                })
                
                # Filter out exact matches and sort for BERT suggestions
                bert_suggestions = bert_similarity_df[
                    (bert_similarity_df['skill_name'].str.lower() != skill_input) # Exclude the query itself
                ].sort_values(by='similarity_score', ascending=False)
                
                top_bert_suggestion = None
                # Consider a BERT suggestion strong if similarity is >= 0.8 (can be tuned)
                if not bert_suggestions.empty and bert_suggestions.iloc[0]['similarity_score'] >= 0.8:
                    top_bert_suggestion = bert_suggestions.iloc[0]

                # --- Fuzzy Matching (for potential typos) ---
                fuzzy_matches = []
                for _, row in df_processed.iterrows():
                    skill_name_original = row['skill_name']
                    skill_name_lower_in_data = row['skill_name_lower']
                    score = fuzz.ratio(skill_input, skill_name_lower_in_data)
                    # Collect fuzzy matches if score is reasonably good and not an exact match
                    if score >= 70 and skill_name_lower_in_data != skill_input: 
                        fuzzy_matches.append({'skill_name': skill_name_original, 'fuzzy_score': score})
                
                fuzzy_matches_df = pd.DataFrame(fuzzy_matches).sort_values(by='fuzzy_score', ascending=False)

                top_fuzzy_suggestion = None
                # Consider a fuzzy suggestion strong if score is very high (e.g., >= 90-95 for likely typos)
                if not fuzzy_matches_df.empty and fuzzy_matches_df.iloc[0]['fuzzy_score'] >= 90:
                    top_fuzzy_suggestion = fuzzy_matches_df.iloc[0]

                # --- Combined Suggestion Logic ---
                proceed_with_suggestion = False

                if top_bert_suggestion is not None:
                    logging.info(f"Did you mean '{top_bert_suggestion['skill_name']}' (Semantic Similarity: {top_bert_suggestion['similarity_score']:.4f})?")
                    confirm = input("Enter 'yes' to proceed with this skill, or 'no' to see other suggestions: ").strip().lower()
                    if confirm == 'yes':
                        suggested_skill_row = df_processed[df_processed['skill_name_lower'] == top_bert_suggestion['skill_name'].lower()].iloc[0]
                        process_and_predict_skill(suggested_skill_row, bert_preprocess_model, bert_encoder_model, model)
                        proceed_with_suggestion = True
                
                if not proceed_with_suggestion and top_fuzzy_suggestion is not None:
                    # Only suggest fuzzy if it's a very high match AND it's not already covered by a strong BERT suggestion
                    # or if there was no strong BERT suggestion.
                    if (top_bert_suggestion is None) or \
                       (top_bert_suggestion is not None and top_fuzzy_suggestion['skill_name'].lower() != top_bert_suggestion['skill_name'].lower()):
                        
                        logging.info(f"Perhaps you meant '{top_fuzzy_suggestion['skill_name']}' (Fuzzy Match Score: {top_fuzzy_suggestion['fuzzy_score']}%)?")
                        confirm = input("Enter 'yes' to proceed with this skill, or 'no' to try another search: ").strip().lower()
                        if confirm == 'yes':
                            suggested_skill_row = df_processed[df_processed['skill_name_lower'] == top_fuzzy_suggestion['skill_name'].lower()].iloc[0]
                            process_and_predict_skill(suggested_skill_row, bert_preprocess_model, bert_encoder_model, model)
                            proceed_with_suggestion = True
                
                if not proceed_with_suggestion: # If no strong suggestion was taken, list all relevant ones
                    displayed_suggestions = set()
                    all_suggestions_list = []

                    # Add top BERT suggestions (if score is reasonable)
                    for _, row in bert_suggestions.head(5).iterrows():
                        if row['similarity_score'] >= 0.5 and row['skill_name'].lower() not in displayed_suggestions:
                            all_suggestions_list.append(f"'{row['skill_name']}' (Semantic Similarity: {row['similarity_score']:.4f})")
                            displayed_suggestions.add(row['skill_name'].lower())
                    
                    # Add top Fuzzy suggestions (if score is reasonable and not a duplicate)
                    for _, row in fuzzy_matches_df.head(5).iterrows():
                        if row['fuzzy_score'] >= 70 and row['skill_name'].lower() not in displayed_suggestions:
                            all_suggestions_list.append(f"'{row['skill_name']}' (Fuzzy Match: {row['fuzzy_score']}%)")
                            displayed_suggestions.add(row['skill_name'].lower())

                    if all_suggestions_list:
                        logging.info(f"No exact match found. Consider these similar or related skills:")
                        for i, suggestion_text in enumerate(all_suggestions_list):
                            logging.info(f"  {i+1}. {suggestion_text}")
                        logging.info("\nPlease try entering one of the suggested skill names, or a different query.")
                    else:
                        logging.warning(f"Skill '{skill_input}' not found in the dataset, and no closely related skills were found by either method. Please try another skill or 'exit'.")

    except Exception as e:
        logging.error(f"An error occurred during interactive prediction: {e}")
        logging.info("Please ensure that:")
        logging.info("1. The model file 'skill_intelligence_model.keras' exists in the 'models/' directory.")
        logging.info("2. The processed data file 'encoded_features_for_model.parquet' exists in 'data/processed/'.")
        logging.info("3. The BERT_MODEL_URL and BERT_PREPROCESS_URL match what was used during training.")
        logging.info("4. The feature lists (NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURES_MAPPING) match your dataset and model architecture.")
        logging.info("5. The model's output names in neural_architecture.py match what's expected here (e.g., 'regression_outputs', 'binary_classification_outputs').")


if __name__ == "__main__":
    run_interactive_prediction()