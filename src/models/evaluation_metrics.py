# src/models/evaluation_metrics.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Forces TensorFlow to use CPU only

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, \
                            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import logging
import tensorflow_hub as hub
import tensorflow_text as tf_text
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration for data paths and model parameters (must match training script) ---
PROCESSED_DATA_PATH = 'data/processed/encoded_features_for_model.parquet'
MODEL_SAVE_PATH = 'models/skill_intelligence_model.keras' # Updated path

# --- Feature and Target Columns Definition (must match training script) ---
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

CATEGORICAL_FEATURES = [
    'skill_category_encoded',
    'skill_type_encoded',
    'market_trend_encoded'
]

TEXT_EMBEDDING_FEATURES = [
    'skill_name_embedding',
    'prerequisites_embedding',
    'complementary_skills_embedding',
    'industry_embedding'
]

REGRESSION_TARGETS = [
    'future_relevance_score',
    'salary_impact_percent',
    'job_demand_score',
    'learning_time_days'
]

BINARY_CLASSIFICATION_TARGETS = [
    'risk_of_obsolescence_binary'
]

def load_data(path: str) -> pd.DataFrame:
    """Loads processed data from a parquet file."""
    logging.info(f"Loading data from {path}...")
    try:
        df = pd.read_parquet(path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Processed data file not found at {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def prepare_data_for_evaluation(df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Prepares features (X) and targets (y) from the dataframe for model evaluation.
    This function should align with the input structure expected by the trained model.
    """
    logging.info("Preparing data for evaluation...")

    X = {}
    numerical_data = []
    
    for feature in NUMERICAL_FEATURES:
        if feature in df.columns:
            numerical_data.append(df[feature].values.astype(np.float32))
    if numerical_data:
        X['numerical_features_input'] = np.stack(numerical_data, axis=1)

    for cat_feature in CATEGORICAL_FEATURES:
        if cat_feature in df.columns:
            X[f'{cat_feature}_input'] = df[cat_feature].values.astype(np.int32)
    
    for text_feature in TEXT_EMBEDDING_FEATURES:
        if text_feature in df.columns:
            if isinstance(df[text_feature].iloc[0], list): # Check if embeddings are stored as lists
                X[f'{text_feature}_input'] = np.stack(df[text_feature].values).astype(np.float32)
            else: # Assume they are already numpy arrays or similar
                X[f'{text_feature}_input'] = df[text_feature].values.astype(np.float32)

    y = {}
    regression_data = []
    for target in REGRESSION_TARGETS:
        if target in df.columns:
            regression_data.append(df[target].values.astype(np.float32))
    if regression_data:
        y['regression_outputs'] = np.stack(regression_data, axis=1)
    
    for target in BINARY_CLASSIFICATION_TARGETS:
        if target in df.columns:
            y['binary_classification_outputs'] = df[target].values.astype(np.float32).reshape(-1, 1) # Ensure 2D for binary output

    logging.info("Data preparation complete.")
    return X, y

def evaluate_model(model: tf.keras.Model, X: Dict[str, np.ndarray], y: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Evaluates the model on the provided data.
    """
    logging.info("Starting model evaluation...")
    
    raw_predictions = model.predict(X)

    predictions_dict = {}
    
    # --- IMPORTANT FIX HERE ---
    # Check if raw_predictions is a dictionary (expected for multi-output models defined with dict outputs)
    if isinstance(raw_predictions, dict):
        logging.info("model.predict returned a dictionary of outputs, as expected for named model outputs.")
        predictions_dict = raw_predictions # Directly assign if it's already a dictionary with correct keys
    elif isinstance(raw_predictions, list):
        # Fallback for models that return a list of outputs (e.g., if outputs were defined as [tensor1, tensor2])
        logging.warning("model.predict returned a list of outputs. Attempting to map to named outputs by order.")
        output_names = [output_tensor.name.split('/')[0] for output_tensor in model.outputs]
        if len(output_names) == len(raw_predictions):
            for i, name in enumerate(output_names):
                predictions_dict[name] = raw_predictions[i]
        else:
            logging.error(f"Mismatch between number of model outputs ({len(output_names)}) and number of prediction arrays ({len(raw_predictions)}) in list format.")
            logging.error(f"Model outputs: {[o.name for o in model.outputs]}")
            return {
                'regression_metrics': {},
                'binary_classification_metrics': {}
            }
    else:
        # This case would handle if a multi-output model somehow produces a single concatenated NumPy array
        # This is less likely given the model's explicit dictionary output definition but included for robustness.
        logging.warning("model.predict returned a single numpy array. Assuming concatenated outputs and attempting split.")
        
        output_names_from_model = [output_tensor.name.split('/')[0] for output_tensor in model.outputs]
        binary_dim = len(BINARY_CLASSIFICATION_TARGETS) # Should be 1
        regression_dim = len(REGRESSION_TARGETS) # Should be 4
        expected_total_dim = binary_dim + regression_dim

        if raw_predictions.shape[1] < expected_total_dim:
            logging.error(f"Single prediction array has {raw_predictions.shape[1]} columns, but expected at least {expected_total_dim} ({binary_dim} binary + {regression_dim} regression). Cannot split.")
            return {
                'regression_metrics': {},
                'binary_classification_metrics': {}
            }
        
        current_col = 0
        for name in output_names_from_model:
            if name == 'binary_classification_outputs':
                predictions_dict[name] = raw_predictions[:, current_col : current_col + binary_dim]
                current_col += binary_dim
            elif name == 'regression_outputs':
                predictions_dict[name] = raw_predictions[:, current_col : current_col + regression_dim]
                current_col += regression_dim
            else:
                logging.warning(f"Unexpected output name '{name}' found in model.outputs. Skipping split for this output.")

        if not all(name.split('/')[0] in predictions_dict for name in model.outputs):
             logging.error("Failed to correctly split the single prediction array into all expected outputs from a single array.")
             return {
                'regression_metrics': {},
                'binary_classification_metrics': {}
            }


    # Ensure the required keys are present before proceeding with metric calculations
    if 'regression_outputs' not in predictions_dict or 'binary_classification_outputs' not in predictions_dict:
        logging.error("Required 'regression_outputs' or 'binary_classification_outputs' are missing from the model's predictions.")
        return {
            'regression_metrics': {},
            'binary_classification_metrics': {}
        }

    regression_metrics_results = {}
    binary_metrics = {}

    # Evaluate Regression Outputs per target
    if 'regression_outputs' in y and 'regression_outputs' in predictions_dict:
        true_regression_all = y['regression_outputs']
        pred_regression_all = predictions_dict['regression_outputs']
        
        for i, target_name in enumerate(REGRESSION_TARGETS):
            if i < true_regression_all.shape[1] and i < pred_regression_all.shape[1]:
                true_target = true_regression_all[:, i]
                pred_target = pred_regression_all[:, i]

                mse = mean_squared_error(true_target, pred_target)
                mae = mean_absolute_error(true_target, pred_target)
                r2 = r2_score(true_target, pred_target)
                
                regression_metrics_results[target_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2
                }
                logging.info(f"Regression Metrics for '{target_name.replace('_', ' ').title()}': MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            else:
                logging.warning(f"Skipping metrics for '{target_name}' due to dimension mismatch in regression outputs.")


    # Evaluate Binary Classification Outputs
    if 'binary_classification_outputs' in y and 'binary_classification_outputs' in predictions_dict:
        true_binary = y['binary_classification_outputs']
        pred_binary_proba = predictions_dict['binary_classification_outputs'] # Probabilities or logits
        
        # Ensure predictions are 1D for metrics functions if they come as (N,1)
        if pred_binary_proba.ndim > 1 and pred_binary_proba.shape[1] == 1:
            pred_binary_proba = pred_binary_proba.flatten()
        
        # Convert probabilities to binary predictions (0 or 1)
        pred_binary = (pred_binary_proba > 0.5).astype(int) # Assuming sigmoid output

        # Ensure true_binary is 1D if it's (N,1)
        if true_binary.ndim > 1 and true_binary.shape[1] == 1:
            true_binary = true_binary.flatten()
        
        target_name = BINARY_CLASSIFICATION_TARGETS[0] if BINARY_CLASSIFICATION_TARGETS else 'binary_output'

        # Calculate metrics, handle cases for single-class predictions in test set
        current_binary_metrics = {}
        
        if len(np.unique(true_binary)) < 2:
            logging.warning(f"Skipping AUC for '{target_name}' because the test set contains only one class.")
            auc_score_val = np.nan # Not applicable
        else:
            try:
                auc_score_val = roc_auc_score(true_binary, pred_binary_proba)
            except ValueError as e:
                logging.warning(f"Could not calculate AUC for '{target_name}': {e}. This often happens if only one class is present in y_true.")
                auc_score_val = np.nan


        # Handle cases where all predictions are the same, which can break precision/recall/f1
        if len(np.unique(pred_binary)) == 1 and len(np.unique(true_binary)) == 1 and np.unique(pred_binary)[0] != np.unique(true_binary)[0]:
             # All predictions are one class, all true labels are another. Precision/Recall will be zero/undefined.
            logging.warning(f"For '{target_name}': All predictions are '{np.unique(pred_binary)[0]}' while all true labels are '{np.unique(true_binary)[0]}'. Accuracy will be 0.")
            current_binary_metrics['accuracy'] = 0.0
            current_binary_metrics['precision'] = 0.0
            current_binary_metrics['recall'] = 0.0
            current_binary_metrics['f1_score'] = 0.0
        elif len(np.unique(pred_binary)) == 1 and len(np.unique(true_binary)) == 1 and np.unique(pred_binary)[0] == np.unique(true_binary)[0]:
             # All predictions are one class, all true labels are the same class. Perfect accuracy, but precision/recall need careful handling.
            logging.warning(f"For '{target_name}': Test set contains only one class ({np.unique(true_binary)[0]}) and model predicts it perfectly.")
            current_binary_metrics['accuracy'] = accuracy_score(true_binary, pred_binary)
            current_binary_metrics['precision'] = 1.0 # If all correctly predicted, precision is 1.0
            current_binary_metrics['recall'] = 1.0    # If all correctly predicted, recall is 1.0
            current_binary_metrics['f1_score'] = 1.0
        else:
            current_binary_metrics['accuracy'] = accuracy_score(true_binary, pred_binary)
            current_binary_metrics['precision'] = precision_score(true_binary, pred_binary, zero_division=0)
            current_binary_metrics['recall'] = recall_score(true_binary, pred_binary, zero_division=0)
            current_binary_metrics['f1_score'] = f1_score(true_binary, pred_binary, zero_division=0)

        current_binary_metrics['auc'] = auc_score_val
        binary_metrics[target_name] = current_binary_metrics
        logging.info(f"Binary Classification Metrics for '{target_name}': Accuracy={current_binary_metrics['accuracy']:.4f}, Precision={current_binary_metrics['precision']:.4f}, Recall={current_binary_metrics['recall']:.4f}, F1-Score={current_binary_metrics['f1_score']:.4f}, AUC={current_binary_metrics['auc']:.4f}")

    logging.info("Model evaluation complete.")

    return {
        'regression_metrics': regression_metrics_results,
        'binary_classification_metrics': binary_metrics
    }


if __name__ == '__main__':
    logging.info("Running evaluation_metrics.py")
    try:
        # Load the saved model
        model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'KerasLayer': hub.KerasLayer})
        logging.info(f"Model loaded successfully from {MODEL_SAVE_PATH}")

        df_encoded = load_data(PROCESSED_DATA_PATH)
        
        X_all, y_all = prepare_data_for_evaluation(df_encoded)

        stratify_col_name = BINARY_CLASSIFICATION_TARGETS[0] if BINARY_CLASSIFICATION_TARGETS else None
        stratify_y_all = df_encoded[stratify_col_name].values.flatten() if stratify_col_name and stratify_col_name in df_encoded.columns else None

        indices = np.arange(len(df_encoded))
        
        _, test_indices, _, _ = train_test_split(
            indices, 
            stratify_y_all, 
            test_size=0.15, 
            random_state=42, 
            shuffle=True, 
            stratify=stratify_y_all
        )

        X_test_fixed = {k: v[test_indices] for k, v in X_all.items()}
        y_test_fixed = {k: v[test_indices] for k, v in y_all.items()}

        logging.info(f"Evaluating on a fixed test set of {len(test_indices)} samples.")
        
        evaluation_results = evaluate_model(model, X_test_fixed, y_test_fixed)

        # Logging the results neatly
        logging.info("\n--- Overall Evaluation Results ---")
        if evaluation_results['regression_metrics']:
            logging.info("\nRegression Metrics:")
            logging.info("Metric\tMSE\tMAE\tR2 Score") # Table header
            for target_label, metrics in evaluation_results['regression_metrics'].items():
                # Format target label for display
                display_label = target_label.replace('_', ' ').title()
                mse_val = metrics.get('mse', np.nan)
                mae_val = metrics.get('mae', np.nan)
                r2_val = metrics.get('r2_score', np.nan)
                logging.info(f"{display_label}\t{mse_val:.4f}\t{mae_val:.4f}\t{r2_val:.4f}")

        if evaluation_results['binary_classification_metrics']:
            logging.info("\nBinary Classification Metrics:")
            for target_label, metrics in evaluation_results['binary_classification_metrics'].items():
                logging.info(f"  {target_label.replace('_', ' ').title()}:\r") # Added \r to ensure newline
                for metric_name, value in metrics.items():
                    if np.isnan(value):
                        logging.info(f"    {metric_name.replace('_', ' ').title()}: N/A (single class in test set)")
                    else:
                        logging.info(f"    {metric_name.replace('_', ' ').title()}: {value:.4f}")
        logging.info("Model evaluation complete.")

    except Exception as e:
        logging.error(f"Evaluation script failed: {str(e)}", exc_info=True)