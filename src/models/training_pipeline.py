# src/models/training_pipeline.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from src.models.neural_architecture import build_skill_intelligence_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import json
import logging
from typing import Tuple, Dict, Any
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Helper function to convert numpy types to native Python types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() # Convert numpy arrays to lists
    elif isinstance(obj, np.floating):
        return float(obj)   # Convert numpy floats to Python floats
    elif isinstance(obj, np.integer):
        return int(obj)     # Convert numpy integers to Python integers
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    else:
        return obj

class TrainingPipeline:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {
            'processed_data_path': 'data/processed/encoded_features_for_model.parquet',
            'model_save_dir': 'models',
            'model_final_save_path': 'models/skill_intelligence_model.keras', # New config for final model save
            'log_dir': 'logs',
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 10,
            'embedding_dim': 128,
            'dropout_rate': 0.3,
            'l2_reg': 0.001,
            'num_folds': 5,
            'numerical_features': [
                'learning_time_days', 'popularity_score', 'job_demand_score',
                'salary_impact_percent', 'future_relevance_score',
                'learning_resources_quality', 'skill_complexity_score',
                'market_momentum_score', 'ecosystem_richness',
                'industry_diversity_metric', 'resource_availability_index',
                'learning_accessibility_score'
            ],
            'categorical_features': [
                'skill_category_encoded', 'skill_type_encoded', 'market_trend_encoded'
            ],
            'text_embedding_features': [
                'skill_name_embedding', 'prerequisites_embedding',
                'complementary_skills_embedding', 'industry_embedding'
            ],
            'regression_targets': ['future_relevance_score', 'salary_impact_percent', 'job_demand_score', 'learning_time_days'],
            'binary_classification_targets': ['risk_of_obsolescence_binary']
        }
        if config:
            self.config.update(config)

        os.makedirs(self.config['model_save_dir'], exist_ok=True)
        os.makedirs(os.path.join(self.config['model_save_dir'], 'trained_histories'), exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)


        self.model = None
        self.history = None
        self.df_encoded = None
        self.num_numerical_features = None
        self.categorical_vocab_sizes = {}
        self.input_shapes = {}


    def load_data(self) -> pd.DataFrame:
        logging.info(f"Loading processed data from {self.config['processed_data_path']}...")
        try:
            df = pd.read_parquet(self.config['processed_data_path'])
            logging.info(f"Data loaded successfully. Shape: {df.shape}")
            self.df_encoded = df
            return df
        except FileNotFoundError:
            logging.error(f"Processed data file not found at {self.config['processed_data_path']}")
            raise
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise


    def prepare_data(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        logging.info("Preparing data for training...")

        X = {}
        numerical_data = []
        numerical_feature_names_found = []
        for feature in self.config['numerical_features']:
            if feature in df.columns:
                numerical_data.append(df[feature].values.astype(np.float32))
                numerical_feature_names_found.append(feature)

        if numerical_data:
            X['numerical_features_input'] = np.stack(numerical_data, axis=1)
            self.num_numerical_features = len(numerical_feature_names_found)
        else:
            self.num_numerical_features = 0

        for cat_feature in self.config['categorical_features']:
            if cat_feature in df.columns:
                X[f'{cat_feature}_input'] = df[cat_feature].values.astype(np.int32)
                self.categorical_vocab_sizes[cat_feature] = int(df[cat_feature].max() + 1)

        for text_feature in self.config['text_embedding_features']:
            if text_feature in df.columns:
                if isinstance(df[text_feature].iloc[0], list):
                    X[f'{text_feature}_input'] = np.stack(df[text_feature].values).astype(np.float32)
                else:
                    X[f'{text_feature}_input'] = df[text_feature].values.astype(np.float32)
                self.input_shapes[text_feature] = X[f'{text_feature}_input'].shape[1:]

        y = {}
        regression_data = []
        for target in self.config['regression_targets']:
            if target in df.columns:
                regression_data.append(df[target].values.astype(np.float32))
        
        if regression_data:
            y['regression_outputs'] = np.stack(regression_data, axis=1)
        
        for target in self.config['binary_classification_targets']:
            if target in df.columns:
                y['binary_classification_outputs'] = df[target].values.astype(np.float32).reshape(-1, 1)

        stratify_y = None
        if self.config['binary_classification_targets'] and self.config['binary_classification_targets'][0] in df.columns:
            stratify_y = df[self.config['binary_classification_targets'][0]].values.flatten()


        indices = np.arange(len(df))

        train_indices_initial, val_test_indices, _, val_test_stratify_y = train_test_split(
            indices, stratify_y, test_size=0.30, random_state=42, shuffle=True, stratify=stratify_y
        )

        val_indices, test_indices, _, _ = train_test_split(
            val_test_indices, val_test_stratify_y, test_size=0.50, random_state=42, shuffle=True, stratify=val_test_stratify_y
        )

        X_train = {k: v[train_indices_initial] for k, v in X.items()}
        y_train = {k: v[train_indices_initial] for k, v in y.items()}

        X_val = {k: v[val_indices] for k, v in X.items()}
        y_val = {k: v[val_indices] for k, v in y.items()}

        X_test = {k: v[test_indices] for k, v in X.items()}
        y_test = {k: v[test_indices] for k, v in y.items()}

        logging.info(f"Data split: Training {len(train_indices_initial)} samples, Validation {len(val_indices)} samples, Test {len(test_indices)} samples")
        return X_train, y_train, X_val, y_val, X_test, y_test


    def train(self) -> Tuple[tf.keras.Model, Dict[str, Any]]:
        logging.info("Starting model training pipeline with cross-validation...")
        if self.df_encoded is None:
            self.load_data()

        X_train_initial, y_train_initial, X_val_fixed, y_val_fixed, X_test_fixed, y_test_fixed = self.prepare_data(self.df_encoded)

        X_for_cv = {}
        for feature_name in X_train_initial.keys():
            X_for_cv[feature_name] = np.concatenate((X_train_initial[feature_name], X_val_fixed[feature_name]), axis=0)

        y_for_cv = {}
        if 'regression_outputs' in y_train_initial and 'regression_outputs' in y_val_fixed:
            y_for_cv['regression_outputs'] = np.concatenate((y_train_initial['regression_outputs'], y_val_fixed['regression_outputs']), axis=0)
        if 'binary_classification_outputs' in y_train_initial and 'binary_classification_outputs' in y_val_fixed:
            y_for_cv['binary_classification_outputs'] = np.concatenate((y_train_initial['binary_classification_outputs'], y_val_fixed['binary_classification_outputs']), axis=0)


        model_save_dir_with_timestamp = os.path.join(self.config['model_save_dir'], datetime.now().strftime("%Y%m%d_%H%M%S_cross_validation"))
        os.makedirs(model_save_dir_with_timestamp, exist_ok=True)

        all_fold_histories = []
        all_fold_eval_results = []

        splitter = None
        if self.config['binary_classification_targets'] and 'binary_classification_outputs' in y_for_cv:
            y_stratify_cv = y_for_cv['binary_classification_outputs'].flatten()
            splitter = StratifiedKFold(n_splits=self.config['num_folds'], shuffle=True, random_state=42)
            indices_cv = np.arange(len(list(X_for_cv.values())[0]))
            split_iterator = splitter.split(indices_cv, y_stratify_cv)
        else:
            splitter = KFold(n_splits=self.config['num_folds'], shuffle=True, random_state=42)
            indices_cv = np.arange(len(list(X_for_cv.values())[0]))
            split_iterator = splitter.split(indices_cv)

        fold_num = 1
        for train_idx, val_idx in split_iterator:
            logging.info(f"--- Starting Fold {fold_num}/{self.config['num_folds']} ---")

            X_train_fold = {k: v[train_idx] for k, v in X_for_cv.items()}
            y_train_fold = {k: v[train_idx] for k, v in y_for_cv.items()}
            X_val_fold = {k: v[val_idx] for k, v in X_for_cv.items()}
            y_val_fold = {k: v[val_idx] for k, v in y_for_cv.items()}

            fold_sample_weight = None
            if self.config['binary_classification_targets'] and 'binary_classification_outputs' in y_train_fold:
                binary_target_name = 'binary_classification_outputs'
                classes_fold = np.unique(y_train_fold[binary_target_name])
                if len(classes_fold) > 1:
                    class_weights_fold = compute_class_weight(
                        class_weight='balanced',
                        classes=classes_fold,
                        y=y_train_fold[binary_target_name].flatten()
                    )
                    class_weight_dict_fold = {cls: weight for cls, weight in zip(classes_fold, class_weights_fold)}
                    sample_weight_binary_fold = np.array([class_weight_dict_fold[label] for label in y_train_fold[binary_target_name].flatten()])
                    fold_sample_weight = {'binary_classification_outputs': sample_weight_binary_fold}
                else:
                    logging.warning(f"Only one class found for binary target '{self.config['binary_classification_targets'][0]}' in fold {fold_num}, skipping class weight calculation.")

            self.model = build_skill_intelligence_model(
                num_numerical_features=self.num_numerical_features,
                categorical_vocab_sizes=self.categorical_vocab_sizes,
                input_shapes=self.input_shapes,
                embedding_dim=self.config['embedding_dim'],
                dropout_rate=self.config['dropout_rate'],
                l2_reg=self.config['l2_reg'],
                learning_rate=self.config['learning_rate']
            )
            logging.info(f"Model rebuilt for Fold {fold_num}.")

            model_save_filepath_fold = os.path.join(model_save_dir_with_timestamp, f"skill_intelligence_model_fold_{fold_num}.keras")
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=self.config['patience'], restore_best_weights=True),
                ModelCheckpoint(
                    filepath=model_save_filepath_fold,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]

            logging.info(f"Training model for Fold {fold_num}...")
            history_fold = self.model.fit(
                X_train_fold,
                y_train_fold,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=(X_val_fold, y_val_fold),
                sample_weight=fold_sample_weight,
                callbacks=callbacks,
                verbose=2
            )
            logging.info(f"Training complete for Fold {fold_num}.")
            all_fold_histories.append(history_fold.history)

            logging.info(f"Evaluating Fold {fold_num} on the fixed test set...")
            test_results_fold = self.model.evaluate(X_test_fixed, y_test_fixed, verbose=0)
            test_metrics = {name: value for name, value in zip(self.model.metrics_names, test_results_fold)}
            logging.info(f"Fold {fold_num} Test Metrics: {test_metrics}")
            all_fold_eval_results.append(test_metrics)

            fold_num += 1

        logging.info("Cross-validation complete.")

        avg_validation_loss = np.mean([h['val_loss'][-1] for h in all_fold_histories])
        logging.info(f"Average Final Validation Loss across folds: {avg_validation_loss:.4f}")

        # Save the final model after cross-validation
        final_model_path = self.config['model_final_save_path']
        self.model.save(final_model_path)
        logging.info(f"Final trained model saved to {final_model_path}")

        aggregate_history_path = os.path.join(self.config['model_save_dir'], 'trained_histories', "aggregated_history_cv.json")
        with open(aggregate_history_path, 'w') as f:
            json.dump(
                {
                    'fold_histories': convert_numpy_types(all_fold_histories),
                    'fold_test_results': convert_numpy_types(all_fold_eval_results)
                },
                f,
                indent=4
            )
        logging.info(f"Aggregated training history and test results saved to {aggregate_history_path}")

        return self.model, {'fold_histories': all_fold_histories, 'fold_test_results': all_fold_eval_results}


    def _prepare_full_data_for_cv(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Helper to prepare full data for CV splitting within the train method,
        ensuring consistent input dictionary keys with the model.
        """
        X = {}
        numerical_data = []
        for feature in self.config['numerical_features']:
            if feature in df.columns:
                numerical_data.append(df[feature].values.astype(np.float32))
        if numerical_data:
            X['numerical_features_input'] = np.stack(numerical_data, axis=1)
        
        for cat_feature in self.config['categorical_features']:
            if cat_feature in df.columns:
                X[f'{cat_feature}_input'] = df[cat_feature].values.astype(np.int32)
                if cat_feature not in self.categorical_vocab_sizes:
                    self.categorical_vocab_sizes[cat_feature] = int(df[cat_feature].max() + 1)
        
        for text_feature in self.config['text_embedding_features']:
            if text_feature in df.columns:
                if isinstance(df[text_feature].iloc[0], list):
                    X[f'{text_feature}_input'] = np.stack(df[text_feature].values).astype(np.float32)
                else:
                    X[f'{text_feature}_input'] = df[text_feature].values.astype(np.float32)
                if text_feature not in self.input_shapes:
                    self.input_shapes[text_feature] = X[f'{text_feature}_input'].shape[1:]

        y = {}
        regression_data = []
        for target in self.config['regression_targets']:
            if target in df.columns:
                regression_data.append(df[target].values.astype(np.float32))
        
        if regression_data:
            y['regression_outputs'] = np.stack(regression_data, axis=1)
        
        for target in self.config['binary_classification_targets']:
            if target in df.columns:
                y['binary_classification_outputs'] = df[target].values.astype(np.float32).reshape(-1, 1)
        return X, y


if __name__ == "__main__":
    try:
        logging.info("Running TrainingPipeline example with cross-validation.")
        pipeline = TrainingPipeline()
        model, results = pipeline.train()
        logging.info("Training pipeline finished successfully.")
    except Exception as e:
        logging.error(f"Training pipeline execution failed: {str(e)}", exc_info=True)