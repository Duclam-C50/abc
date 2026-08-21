"""
anomaly_detection package.

Modules:
    config      -- Centralised configuration (HyperParams + Paths dataclasses).
    model       -- Keras LSTM Autoencoder architecture.
    dataset     -- Sequence construction utilities.
    preprocess  -- Feature encoding and normalisation.
    visualization -- Loss / error plotting helpers.
    trainer     -- Model training logic.
    evaluator   -- Model evaluation and metrics.
    detector    -- Anomaly detection pipeline.
    explainer   -- Integrated Gradients explanation + dashboard generation.
"""
