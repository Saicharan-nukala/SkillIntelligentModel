# src/models/skill_intelligence_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.models.neural_architecture import build_skill_intelligence_model
import os
import json
import logging
import tensorflow_text as tf_text # For BERT preprocessing (tokenization etc.)
import tensorflow_hub as hub # For loading pre-trained BERT model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration for data paths and model parameters ---
PROCESSED_DATA_PATH = 'data/processed/encoded_features_for_model.parquet'
MODEL_SAVE_PATH = 'models/skill_intelligence_model.keras' # Keras native format
HISTORY_SAVE_PATH = 'models/training_history.json'

# --- Feature Columns Definition ---
NUMERICAL_FEATURES = [
    'learning_time_days',
    'popularity_score',
    'job_demand_score',
    'salary_impact_percent',
    'future_relevance_score',
    'learning_resources_quality',
    'skill_complexity_score',
    'market_momentum_score',
    'ecosystem_richness',
    'industry_diversity_metric',
    'resource_availability_index',
    'learning_accessibility_score'
]

# BERT model URL - choose a suitable small BERT model for efficiency
BERT_MODEL_URL = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1"
BERT_PREPROCESS_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"


PREREQ_FEATURES_PREFIX = 'prerequisites_topic_' # Corrected to match training_pipeline and evaluation_metrics
COMPLEMENTARY_FEATURES_PREFIX = 'complementary_skills_topic_' # Corrected to match training_pipeline and evaluation_metrics
INDUSTRY_FEATURES_PREFIX = 'industry_usage_topic_' # Corrected to match training_pipeline and evaluation_metrics

CATEGORICAL_FEATURES = ['category_encoded', 'skill_type_encoded', 'market_trend_encoded']
BINARY_FEATURES = ['certification_available'] # This list is not directly used for model inputs but for conceptual grouping

# Target variables - Must match training_pipeline.py and evaluation_metrics.py
REGRESSION_TARGETS = [
    'popularity_score', 'job_demand_score', 'salary_impact_percent',
    'future_relevance_score', 'learning_resources_quality', 'skill_complexity_score',
    'market_momentum_score', 'ecosystem_richness',
    'industry_diversity_metric', 'resource_availability_index',
    'learning_time_days', 'learning_accessibility_score'
]
BINARY_CLASSIFICATION_TARGETS = ['certification_available', 'risk_of_obsolescence_binary']


def prepare_data(df: pd.DataFrame):
    """
    Prepares data for the Keras model, splitting into inputs and outputs.
    Now includes BERT preprocessing for skill_name.
    """
    logging.info("Preparing data for model training with BERT preprocessing.")

    # Identify all dynamic feature columns
    prereq_features = [col for col in df.columns if col.startswith(PREREQ_FEATURES_PREFIX)]
    complementary_features = [col for col in df.columns if col.startswith(COMPLEMENTARY_FEATURES_PREFIX)]
    industry_features = [col for col in df.columns if col.startswith(INDUSTRY_FEATURES_PREFIX)]

    # Load BERT preprocessor
    preprocess_model = hub.KerasLayer(BERT_PREPROCESS_URL, name='skill_name_preprocessor')

    # Apply BERT preprocessing to skill_name
    if 'skill_name' not in df.columns:
        raise ValueError("DataFrame must contain a 'skill_name' column for BERT processing.")

    skill_name_text = tf.constant(df['skill_name'].values.tolist())
    processed_skill_name_inputs = preprocess_model(skill_name_text)

    # Input dictionary for the model
    model_inputs = {
        'numerical_input': df[NUMERICAL_FEATURES].values.astype(np.float32),
        'prereq_input': df[prereq_features].values.astype(np.float32) if prereq_features else np.zeros((len(df), 1), dtype=np.float32),
        'complementary_input': df[complementary_features].values.astype(np.float32) if complementary_features else np.zeros((len(df), 1), dtype=np.float32),
        'industry_input': df[industry_features].values.astype(np.float32) if industry_features else np.zeros((len(df), 1), dtype=np.float32),
        'category_input': df['category_encoded'].values.astype(np.int32),
        'skill_type_input': df['skill_type_encoded'].values.astype(np.int32),
        'market_trend_input': df['market_trend_encoded'].values.astype(np.int32),
        'skill_name_input_word_ids': processed_skill_name_inputs['input_word_ids'],
        'skill_name_input_mask': processed_skill_name_inputs['input_mask'],
        'skill_name_input_type_ids': processed_skill_name_inputs['input_type_ids']
    }

    # Output dictionary for the model
    model_outputs = {
        'regression_outputs': df[REGRESSION_TARGETS].values.astype(np.float32),
        'binary_classification_outputs': df[BINARY_CLASSIFICATION_TARGETS].values.astype(np.float32)
    }

    logging.info("Data preparation complete.")
    return model_inputs, model_outputs


def train_skill_intelligence_model(df: pd.DataFrame):
    logging.info("Starting skill intelligence model training.")

    # Prepare data
    model_inputs, model_outputs = prepare_data(df)

    num_numerical_features = len(NUMERICAL_FEATURES)

    # Dynamically determine vocab sizes and LDA feature counts
    category_vocab_size = int(df['category_encoded'].max()) + 1 if 'category_encoded' in df.columns else 1
    skill_type_vocab_size = int(df['skill_type_encoded'].max()) + 1 if 'skill_type_encoded' in df.columns else 1
    market_trend_vocab_size = int(df['market_trend_encoded'].max()) + 1 if 'market_trend_encoded' in df.columns else 1


    # Get actual shapes from the prepared inputs, handling empty cases
    num_prereq_features = model_inputs['prereq_input'].shape[1] if model_inputs['prereq_input'].size > 0 else 0
    num_complementary_features = model_inputs['complementary_input'].shape[1] if model_inputs['complementary_input'].size > 0 else 0
    num_industry_features = model_inputs['industry_input'].shape[1] if model_inputs['industry_input'].size > 0 else 0

    skill_name_embedding_dim = 128 # The dimension of the BERT output (e.g., for 'small_bert/bert_en_uncased_L-2_H-128_A-2')

    num_regression_outputs = len(REGRESSION_TARGETS)
    num_binary_classification_outputs = len(BINARY_CLASSIFICATION_TARGETS)

    # Build the model
    model = build_skill_intelligence_model(
        num_numerical_features=num_numerical_features,
        skill_name_embedding_dim=skill_name_embedding_dim, # This will be the BERT output dimension
        num_prereq_features=num_prereq_features,
        num_complementary_features=num_complementary_features,
        num_industry_features=num_industry_features,
        category_vocab_size=category_vocab_size,
        skill_type_vocab_size=skill_type_vocab_size,
        market_trend_vocab_size=market_trend_vocab_size,
        num_regression_outputs=num_regression_outputs,
        num_binary_classification_outputs=num_binary_classification_outputs
    )

    # Split data into training and validation sets
    # Get a list of tuples (input_key, input_array) for splitting
    input_items = list(model_inputs.items())
    output_items = list(model_outputs.items())

    # Create a list of all input arrays and all output arrays for train_test_split
    all_input_arrays = [item[1] for item in input_items]
    all_output_arrays = [item[1] for item in output_items]

    # Perform the split
    split_results = train_test_split(*all_input_arrays, *all_output_arrays, test_size=0.2, random_state=42)

    # Reconstruct train_inputs, val_inputs, train_outputs, val_outputs dictionaries
    # The split_results will be: [train_input1, val_input1, train_input2, val_input2, ..., train_output1, val_output1, ...]
    num_inputs = len(all_input_arrays)
    num_outputs = len(all_output_arrays)

    train_inputs = {input_items[i][0]: split_results[i * 2] for i in range(num_inputs)}
    val_inputs = {input_items[i][0]: split_results[i * 2 + 1] for i in range(num_inputs)}
    train_outputs = {output_items[i][0]: split_results[num_inputs * 2 + i * 2] for i in range(num_outputs)}
    val_outputs = {output_items[i][0]: split_results[num_inputs * 2 + i * 2 + 1] for i in range(num_outputs)}


    # Basic check for empty inputs/outputs before training
    for input_name, data_array in train_inputs.items():
         if data_array.size == 0 and 'input_word_ids' not in input_name and 'input_mask' not in input_name and 'input_type_ids' not in input_name:
             logging.warning(f"Training input '{input_name}' is empty. This might cause issues if expected.")

    for output_name, data_array in train_outputs.items():
        if data_array.size == 0:
            logging.warning(f"Training output '{output_name}' is empty. This might cause issues.")


    # Train the model
    logging.info("Starting model training...")
    history = model.fit(
        train_inputs,
        train_outputs,
        epochs=50, # You can adjust the number of epochs
        batch_size=32, # You can adjust the batch size
        validation_data=(val_inputs, val_outputs),
        verbose=1 # Set to 0 for silent training
    )
    logging.info("Model training complete.")

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    logging.info(f"Trained model saved to {MODEL_SAVE_PATH}")

    # Save training history
    os.makedirs(os.path.dirname(HISTORY_SAVE_PATH), exist_ok=True)
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history.history, f)
    logging.info(f"Training history saved to {HISTORY_SAVE_PATH}")

    return model, history

if __name__ == '__main__':
    logging.info("Running example model training from skill_intelligence_model.py")
    try:
        # For a full run, you'd load your data here, e.g.,
        if os.path.exists(PROCESSED_DATA_PATH):
            df_encoded = pd.read_parquet(PROCESSED_DATA_PATH)
            model, history = train_skill_intelligence_model(df_encoded)
        else:
            logging.error(f"Processed data not found at {PROCESSED_DATA_PATH}. Please run data_encoder.py first.")
            # Optionally, create dummy data for testing if no real data is available
            # For a quick test, you might mock a DataFrame:
            # from data_generator import generate_dummy_data # assuming you have such a function
            # df_dummy = generate_dummy_data(num_samples=100)
            # model, history = train_skill_intelligence_model(df_dummy)
    except Exception as e:
        logging.error(f"Failed to run skill_intelligence_model.py example: {e}", exc_info=True)