# src/models/neural_architecture.py

from tensorflow.keras.layers import (
    Input, Dense, Embedding, Flatten, Concatenate,
    Dropout, BatchNormalization, LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
import logging
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GradientMonitor(Callback):
    """
    A Keras Callback to monitor gradients during training.
    This is an example and might not be used directly in the training pipeline by default.
    """
    def on_train_begin(self, logs=None):
        self.gradient_histories = {
            'mean': [],
            'max': [],
            'min': []
        }

    def on_batch_end(self, batch, logs=None):
        if not self.model._is_compiled:
            return

        with tf.GradientTape() as tape:
            # Ensure inputs are correctly passed to the model's call method
            inputs_for_tape = [inp for inp in self.model.inputs if inp is not None]
            if hasattr(self.model.call, '__wrapped__') and 'training' in self.model.call.__wrapped__.__code__.co_varnames:
                _ = self.model(inputs_for_tape, training=True)
            else:
                _ = self.model(inputs_for_tape)

            grads = tape.gradient(self.model.output, self.model.trainable_weights)
            # Filter out None gradients (e.g., for non-trainable weights or disconnected layers)
            grads = [g for g in grads if g is not None]

            if grads:
                # Calculate mean, max, min of absolute gradients
                abs_grads = [tf.abs(g) for g in grads]
                mean_grad = tf.reduce_mean([tf.reduce_mean(g) for g in abs_grads])
                max_grad = tf.reduce_max([tf.reduce_max(g) for g in abs_grads])
                min_grad = tf.reduce_min([tf.reduce_min(g) for g in abs_grads])

                self.gradient_histories['mean'].append(mean_grad.numpy())
                self.gradient_histories['max'].append(max_grad.numpy())
                self.gradient_histories['min'].append(min_grad.numpy())
            else:
                self.gradient_histories['mean'].append(0.0)
                self.gradient_histories['max'].append(0.0)
                self.gradient_histories['min'].append(0.0)


def build_skill_intelligence_model(
    num_numerical_features: int,
    categorical_vocab_sizes: Dict[str, int], # e.g., {'skill_category_encoded': 40}
    input_shapes: Dict[str, Tuple[int, ...]], # e.g., {'skill_name_embedding': (128,)}
    embedding_dim: int = 128,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001,
    learning_rate: float = 0.001
) -> Model:
    logging.info("Building skill intelligence multi-task model...")

    inputs = {} # This dictionary will hold the Input layers with correct keys
    embedding_layers = []
    text_embedding_inputs = []

    # Numerical Inputs
    if num_numerical_features > 0:
        numerical_input = Input(shape=(num_numerical_features,), name='numerical_features_input')
        inputs['numerical_features_input'] = numerical_input # CORRECTED KEY HERE
        numerical_processed = BatchNormalization(name='numerical_bn')(numerical_input)
    else:
        numerical_processed = None

    # Categorical Inputs and Embeddings
    for cat_feature, vocab_size in categorical_vocab_sizes.items():
        cat_input = Input(shape=(1,), name=f'{cat_feature}_input')
        inputs[f'{cat_feature}_input'] = cat_input # CORRECTED KEY HERE
        embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                              name=f'{cat_feature}_embedding')(cat_input)
        flattened_embedding = Flatten(name=f'{cat_feature}_flatten')(embedding)
        embedding_layers.append(flattened_embedding)

    # Text Embedding Inputs (assuming pre-computed embeddings are directly fed)
    for text_feature, shape in input_shapes.items():
        text_input = Input(shape=shape, name=f'{text_feature}_input')
        inputs[f'{text_feature}_input'] = text_input # CORRECTED KEY HERE
        normalized_text = LayerNormalization(name=f'{text_feature}_ln')(text_input)
        text_embedding_inputs.append(normalized_text)


    # Concatenate all processed inputs
    all_features = []
    if numerical_processed is not None:
        all_features.append(numerical_processed)
    all_features.extend(embedding_layers)
    all_features.extend(text_embedding_inputs)

    if not all_features:
        raise ValueError("No input features defined. Model cannot be built.")

    if len(all_features) > 1:
        concatenated_features = Concatenate(name='concatenated_features')(all_features)
    else:
        concatenated_features = all_features[0]

    # Shared dense layers (common backbone)
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg), name='shared_dense_1')(concatenated_features)
    x = BatchNormalization(name='shared_bn_1')(x)
    x = Dropout(dropout_rate, name='shared_dropout_1')(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg), name='shared_dense_2')(x)
    x = BatchNormalization(name='shared_bn_2')(x)
    x = Dropout(dropout_rate, name='shared_dropout_2')(x)

    # --- Multi-task output heads ---

    # Regression Output Head
    regression_output = Dense(len(['future_relevance_score', 'salary_impact_percent', 'job_demand_score', 'learning_time_days']),
                              activation='linear', name='regression_outputs')(x)

    # Binary Classification Output Head (e.g., risk_of_obsolescence_binary)
    binary_classification_output = Dense(1, activation='sigmoid', name='binary_classification_outputs')(x)

    # Create the model
    model = Model(
        inputs=inputs, # This 'inputs' dictionary now has keys matching the Input layer names
        outputs={
            'regression_outputs': regression_output,
            'binary_classification_outputs': binary_classification_output
        },
        name='SkillIntelligenceModel'
    )

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss={
            'regression_outputs': 'mse',
            'binary_classification_outputs': 'binary_crossentropy'
        },
        metrics={
            'regression_outputs': ['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
            'binary_classification_outputs': [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        }
    )

    logging.info("Model compilation complete.")
    return model

if __name__ == '__main__':
    logging.info("Running example model build with BERT integration.")
    # Dummy data for testing the model build function
    dummy_num_features = 12
    dummy_cat_vocab_sizes = {
        'skill_category_encoded': 40,
        'skill_type_encoded': 120,
        'market_trend_encoded': 10
    }
    dummy_input_shapes = {
        'skill_name_embedding': (128,),
        'prerequisites_embedding': (128,),
        'complementary_skills_embedding': (128,),
        'industry_embedding': (128,)
    }

    example_model = build_skill_intelligence_model(
        num_numerical_features=dummy_num_features,
        categorical_vocab_sizes=dummy_cat_vocab_sizes,
        input_shapes=dummy_input_shapes,
        embedding_dim=128,
        dropout_rate=0.3,
        l2_reg=0.001,
        learning_rate=0.001
    )
    example_model.summary()
    logging.info("Example model built successfully.")