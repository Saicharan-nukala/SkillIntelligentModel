# src/data_processing/data_encoder.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import json
import os
import logging
from typing import List
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as hub

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BERT_PREPROCESS_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
BERT_MODEL_URL = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1"

class DataEncoder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.possible_numerical_cols = [
            'difficulty_level', 'learning_time_days', 'popularity_score',
            'job_demand_score', 'salary_impact_percent', 'future_relevance_score',
            'learning_resources_quality', 'skill_complexity_score',
            'learning_roi', 'market_momentum_score', 'ecosystem_richness',
            'industry_diversity_metric', 'resource_availability_index',
            'skill_adoption_rate', 'learning_accessibility_score'
        ]

        self.possible_categorical_cols = ['category', 'skill_type', 'market_trend']
        self.possible_array_cols = ['industry_usage', 'prerequisites', 'complementary_skills']
        self.possible_binary_cols = ['certification_available', 'risk_of_obsolescence']
        self.possible_text_cols = ['skill_name']

        self.numerical_cols = [col for col in self.possible_numerical_cols if col in self.df.columns]
        self.categorical_cols = [col for col in self.possible_categorical_cols if col in self.df.columns]
        self.array_cols = [col for col in self.possible_array_cols if col in self.df.columns]
        self.binary_cols = [col for col in self.possible_binary_cols if col in self.df.columns]
        self.text_cols = [col for col in self.possible_text_cols if col in self.df.columns]

        self.scaler = MinMaxScaler()
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.label_encoders = {}
        self.lda_model = None

        self.bert_preprocessor = hub.KerasLayer(BERT_PREPROCESS_URL, name='bert_preprocessor_data_encoder')
        self.bert_encoder = hub.KerasLayer(BERT_MODEL_URL, trainable=False, name='bert_encoder_data_encoder')


    def _scale_numerical_features(self):
        logging.info("Scaling numerical features...")

        if not self.numerical_cols:
            logging.warning("No numerical columns found to scale")
            return

        try:
            self.df[self.numerical_cols] = self.df[self.numerical_cols].fillna(0)
            self.df[self.numerical_cols] = self.scaler.fit_transform(self.df[self.numerical_cols])
            logging.info(f"Successfully scaled numerical columns: {self.numerical_cols}")
        except Exception as e:
            logging.error(f"Error scaling numerical features: {str(e)}")
            raise

    def _encode_categorical_features(self):
        logging.info("Encoding categorical features...")

        for col in self.categorical_cols:
            try:
                if col not in self.df.columns:
                    logging.warning(f"Skipping categorical column '{col}' - not found in DataFrame")
                    continue

                self.df[col] = self.df[col].fillna('__MISSING__')
                le = LabelEncoder()
                encoded_col = f"{col}_encoded"
                self.df[encoded_col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                logging.info(f"Encoded categorical column '{col}' with {len(le.classes_)} unique values")
            except Exception as e:
                logging.error(f"Error encoding column '{col}': {str(e)}")
                continue

    def _encode_binary_features(self):
        logging.info("Encoding binary features...")

        for col in self.binary_cols:
            try:
                if col not in self.df.columns:
                    logging.warning(f"Skipping binary column '{col}' - not found in DataFrame")
                    continue

                self.df[col] = self.df[col].astype(int)
                unique_values = set(self.df[col].unique())
                if not unique_values.issubset({0, 1}):
                    logging.warning(f"Binary column '{col}' contains unexpected values: {unique_values}")
                logging.info(f"Encoded binary column '{col}'")
            except Exception as e:
                logging.error(f"Error encoding binary column '{col}': {str(e)}")
                continue

    def _encode_text_features(self):
        logging.info("Encoding text features using BERT...")

        if 'skill_name' not in self.df.columns:
            logging.warning("No 'skill_name' column found to encode with BERT.")
            return

        try:
            skill_name_text = tf.constant(self.df['skill_name'].values.tolist())
            processed_skill_name_inputs = self.bert_preprocessor(skill_name_text)
            skill_name_embeddings = self.bert_encoder(processed_skill_name_inputs)['pooled_output'].numpy()

            embedding_df = pd.DataFrame(
                skill_name_embeddings,
                columns=[f'skill_name_bert_embedding_{i}' for i in range(skill_name_embeddings.shape[1])],
                index=self.df.index
            )
            self.df = pd.concat([self.df, embedding_df], axis=1)
            logging.info(f"Created {skill_name_embeddings.shape[1]} BERT embedding features for skill names.")
        except Exception as e:
            logging.error(f"Error encoding skill_name with BERT: {str(e)}")
            self.df[[f'skill_name_bert_embedding_{i}' for i in range(128)]] = 0.0
            logging.warning("Created dummy BERT embeddings for skill_name due to encoding failure.")


    def _encode_array_features(self, n_components=10):
        logging.info(f"Encoding array features with LDA (n_components={n_components}) and optionally BERT...")

        for col in self.array_cols:
            try:
                if col not in self.df.columns:
                    logging.warning(f"Skipping array column '{col}' - not found in DataFrame")
                    continue

                if self.df[col].apply(type).eq(str).any():
                    self.df[col] = self.df[col].apply(json.loads)

                all_items = [item for sublist in self.df[col].dropna() for item in sublist]
                unique_items = list(set(all_items))

                if unique_items:
                    encoded_df = pd.DataFrame(0, index=self.df.index, columns=unique_items)
                    for idx, items in self.df[col].items():
                        if isinstance(items, list):
                            for item in items:
                                if item in encoded_df.columns:
                                    encoded_df.loc[idx, item] = 1

                    if len(unique_items) > 1 and len(self.df) > n_components:
                        lda = LatentDirichletAllocation(
                            n_components=n_components,
                            random_state=42,
                            learning_method='online'
                        )
                        lda_features = lda.fit_transform(encoded_df)
                        lda_columns = [f'{col.replace("_", "")}_topic_{i}' for i in range(n_components)]
                        lda_df = pd.DataFrame(lda_features, columns=lda_columns, index=self.df.index)
                        self.df = pd.concat([self.df, lda_df], axis=1)
                        logging.info(f"Created {n_components} LDA topics for {col}.")
                    else:
                        logging.warning(f"Insufficient data for LDA on {col}, using multi-hot encoding.")
                        encoded_df = encoded_df.add_prefix(f'{col}_')
                        self.df = pd.concat([self.df, encoded_df], axis=1)
                else:
                    logging.warning(f"Skipping LDA for '{col}' due to no unique items found.")

                text_for_bert = self.df[col].apply(lambda x: " ".join(x) if isinstance(x, list) else "").tolist()
                if any(text_for_bert):
                    logging.info(f"Encoding array column '{col}' with BERT.")
                    bert_text_tensor = tf.constant(text_for_bert)
                    processed_bert_inputs = self.bert_preprocessor(bert_text_tensor)
                    bert_embeddings = self.bert_encoder(processed_bert_inputs)['pooled_output'].numpy()

                    bert_embedding_df = pd.DataFrame(
                        bert_embeddings,
                        columns=[f'{col}_bert_embedding_{i}' for i in range(bert_embeddings.shape[1])],
                        index=self.df.index
                    )
                    self.df = pd.concat([self.df, bert_embedding_df], axis=1)
                    logging.info(f"Created {bert_embeddings.shape[1]} BERT embedding features for {col}.")
                else:
                    logging.warning(f"No valid text found in array column '{col}' for BERT encoding.")

            except Exception as e:
                logging.error(f"Error encoding array column '{col}': {str(e)}", exc_info=True)
                continue

    def encode_and_transform_all_data(self):
        logging.info("Starting full data encoding pipeline")

        try:
            for col in self.array_cols:
                if col in self.df.columns and self.df[col].apply(type).eq(str).any():
                    self.df[col] = self.df[col].apply(json.loads)

            self._scale_numerical_features()
            self._encode_categorical_features()
            self._encode_binary_features()
            self._encode_text_features()
            self._encode_array_features(n_components=10)

            logging.info("Data encoding completed successfully")
            return self.df

        except Exception as e:
            logging.error(f"Data encoding failed: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        input_path = 'data/processed/skills_engineered_features.jsonl'
        output_path = 'data/processed/encoded_features_for_model.parquet'

        logging.info(f"Loading data from {input_path}")
        df = pd.read_json(input_path, lines=True)

        for col in ['prerequisites', 'complementary_skills', 'industry_usage']:
            if col in df.columns and df[col].apply(type).eq(str).any():
                df[col] = df[col].apply(json.loads)

        encoder = DataEncoder(df)
        encoded_df = encoder.encode_and_transform_all_data()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        encoded_df.to_parquet(output_path)
        logging.info(f"Encoded data saved to {output_path}")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")