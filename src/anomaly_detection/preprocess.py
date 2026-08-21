"""
preprocess.py
=============
Feature encoding and normalisation utilities following a scikit-learn-style
``fit`` / ``transform`` pattern.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import List, Optional


class Preprocessor:
    """
    Encodes categorical columns and normalises all features.

    Typical usage
    -------------
    Training phase::

        pp = Preprocessor(feature_columns)
        X_train = pp.fit_transform(df_train)
        pp.save(encoders_path, scaler_path)

    Inference phase::

        pp = Preprocessor.load(feature_columns, encoders_path, scaler_path)
        X_test = pp.transform(df_test)
    """

    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self._encoders: dict[str, LabelEncoder] = {}
        self._scaler: Optional[MinMaxScaler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit encoders and scaler on *df*, then return the scaled array."""
        df = self._select_columns(df)
        df = self._encode(df, fit=True)
        return self._scale(df, fit=True)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply pre-fitted encoders and scaler to *df*."""
        if self._scaler is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit_transform first or load from disk.")
        df = self._select_columns(df)
        df = self._encode(df, fit=False)
        return self._scale(df, fit=False)

    def save(self, encoders_path: str, scaler_path: str) -> None:
        """Persist encoders and scaler to disk."""
        import base64
        import json
        os.makedirs(os.path.dirname(encoders_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        serialized_encoders = {}
        for col, le in self._encoders.items():
            encoded_classes = []
            for cls in le.classes_:
                cls_str = str(cls)
                encoded_cls = base64.b64encode(cls_str.encode('utf-8')).decode('utf-8')
                encoded_classes.append(encoded_cls)
            serialized_encoders[col] = encoded_classes
            
        with open(encoders_path, 'w', encoding='utf-8') as f:
            json.dump(serialized_encoders, f, indent=2)
            
        joblib.dump(self._scaler, scaler_path)

    @classmethod
    def load(cls, feature_columns: List[str], encoders_path: str, scaler_path: str) -> "Preprocessor":
        """Restore a fitted Preprocessor from saved artefact files."""
        import base64
        import json
        pp = cls(feature_columns)
        
        with open(encoders_path, 'r', encoding='utf-8') as f:
            serialized_encoders = json.load(f)
            
        pp._encoders = {}
        for col, encoded_classes in serialized_encoders.items():
            le = LabelEncoder()
            decoded_classes = []
            for enc_cls in encoded_classes:
                dec_cls = base64.b64decode(enc_cls.encode('utf-8')).decode('utf-8')
                decoded_classes.append(dec_cls)
            le.classes_ = np.array(decoded_classes, dtype=object)
            pp._encoders[col] = le
            
        pp._scaler = joblib.load(scaler_path)
        return pp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.feature_columns if c in df.columns]
        return df[cols].copy()

    def _encode(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._encoders[col] = le
            else:
                le = self._encoders.get(col)
                if le is None:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                else:
                    s = df[col].astype(str)
                    unseen = ~s.isin(le.classes_)
                    s[unseen] = le.classes_[0]
                    df[col] = le.transform(s)
        return df

    def _scale(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        if fit:
            self._scaler = MinMaxScaler()
            return self._scaler.fit_transform(df)

        # Align columns to what the scaler was fitted on
        if hasattr(self._scaler, "feature_names_in_"):
            for col in self._scaler.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0
            df = df[self._scaler.feature_names_in_]
        return self._scaler.transform(df)
