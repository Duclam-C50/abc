"""
model.py
========
Keras LSTM Autoencoder for sequence reconstruction-based anomaly detection.

Architecture
------------
- Encoder : stacked LSTM layers  ->  single context vector
- RepeatVector : broadcast context across the sequence length
- Decoder : stacked LSTM layers  ->  reconstructed sequence
- Dense output layer : project back to input feature dimension
"""

import keras
from keras import layers


@keras.saving.register_keras_serializable()
class LSTMAutoencoder(keras.Model):
    """
    LSTM Autoencoder that learns to reconstruct normal sequences.

    Anomalies are identified by an elevated mean-squared reconstruction error.

    Args:
        n_features  : Number of input features per time-step.
        hidden_size : Number of LSTM units in each layer.
        num_layers  : Number of stacked LSTM layers in encoder and decoder.
    """

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder - last LSTM does NOT return sequences (produces context vector)
        self.encoder = keras.Sequential(
            [
                layers.LSTM(hidden_size, return_sequences=(i < num_layers - 1), dropout=0.2)
                for i in range(num_layers)
            ],
            name="encoder",
        )

        # Decoder - all LSTMs return sequences (to rebuild the full sequence)
        self.decoder = keras.Sequential(
            [layers.LSTM(hidden_size, return_sequences=True, dropout=0.2) for _ in range(num_layers)],
            name="decoder",
        )

        self.output_projection = layers.Dense(n_features, name="output_projection")

    def call(self, x, training=False):
        """
        Forward pass.

        Args:
            x        : Input tensor of shape (batch, seq_len, n_features).
            training : Whether the model is in training mode.

        Returns:
            Reconstructed tensor of shape (batch, seq_len, n_features).
        """
        seq_len = keras.ops.shape(x)[1]

        # Encode to context vector: (batch, hidden_size)
        context = self.encoder(x, training=training)

        # Repeat context across time axis: (batch, seq_len, hidden_size)
        repeated = keras.ops.tile(keras.ops.expand_dims(context, axis=1), (1, seq_len, 1))

        # Decode: (batch, seq_len, hidden_size)
        decoded = self.decoder(repeated, training=training)

        # Project to feature space: (batch, seq_len, n_features)
        return self.output_projection(decoded)

    # ------------------------------------------------------------------
    # Serialisation (required for keras.models.save / load_model)
    # ------------------------------------------------------------------
    def get_config(self):
        config = super().get_config()
        config.update(
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


import numpy as np

@keras.saving.register_keras_serializable()
class EventSpecificMaskedMSE(keras.losses.Loss):
    """
    Custom Keras loss function that computes Mean Squared Error (MSE)
    only on features that are relevant to each specific Event ID,
    while masking out zero-padded blank steps.
    """

    def __init__(self, mask_matrix: list, event_col_indices: list, **kwargs):
        super().__init__(**kwargs)
        self.mask_matrix = np.array(mask_matrix, dtype=np.float32)
        self.event_col_indices = list(event_col_indices)

    def call(self, y_true, y_pred):
        # ------------------------------------------------------------------
        # 1. Tạo mặt nạ thời gian Step Mask (Khử bước đệm zero-padding)
        # ------------------------------------------------------------------
        step_mask = keras.ops.any(y_true != 0, axis=2)  # (batch_size, seq_len)
        step_mask_float = keras.ops.cast(step_mask, "float32")
        # ------------------------------------------------------------------
        # 2. Tạo mặt nạ đặc trưng Feature Mask theo Event ID
        # ------------------------------------------------------------------
        one_hot_events = keras.ops.take(y_true, self.event_col_indices, axis=2)
        mask_tensor = keras.ops.convert_to_tensor(self.mask_matrix, dtype="float32")

        # Compute feature mask: (batch_size, seq_len, n_features)
        feature_mask = keras.ops.matmul(one_hot_events, mask_tensor)
        # ------------------------------------------------------------------
        # 3. [CÔNG THỨC 17] Tính MSEt có lọc mặt nạ đặc trưng tại từng bước t
        # ------------------------------------------------------------------
        sq_err = keras.ops.square(y_true - y_pred)
        masked_sq_err = sq_err * feature_mask
        active_feature_counts = keras.ops.sum(feature_mask, axis=2, keepdims=True)
        active_feature_counts = keras.ops.maximum(active_feature_counts, 1.0)
        mse_per_step = keras.ops.sum(masked_sq_err, axis=2) / keras.ops.squeeze(active_feature_counts, axis=2)
        # ------------------------------------------------------------------
        # 4. [CÔNG THỨC 18] Tính Loss_chuỗi bằng cách khử đệm và chia T_thực
        # ------------------------------------------------------------------
        total_active_steps = keras.ops.sum(step_mask_float, axis=1)
        total_active_steps = keras.ops.maximum(total_active_steps, 1.0)
        
        # Loss của từng chuỗi i trong lô
        loss = keras.ops.sum(mse_per_step * step_mask_float, axis=1) / total_active_steps
        # ------------------------------------------------------------------
        # 5. [CÔNG THỨC 19] Trung bình cộng sai số trên toàn bộ Batch (N chuỗi)
        # Keras mặc định tính mean(loss_per_sequence) = (1/N) * sum(Loss_chuỗi)
        # ------------------------------------------------------------------
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "mask_matrix": self.mask_matrix.tolist(),
            "event_col_indices": self.event_col_indices
        })
        return config

