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
