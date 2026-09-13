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
        self.event_ids: List[int] = []
        self._encoders: dict[str, LabelEncoder] = {}
        self._scaler: Optional[MinMaxScaler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit encoders and scaler on *df*, then return the scaled array."""
        df = self._select_columns(df)
        df = self._one_hot_event_id(df, fit=True)
        df = self._encode(df, fit=True)
        return self._scale(df, fit=True)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply pre-fitted encoders and scaler to *df*."""
        if self._scaler is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit_transform first or load from disk.")
        df = self._select_columns(df)
        df = self._one_hot_event_id(df, fit=False)
        df = self._encode(df, fit=False)
        return self._scale(df, fit=False)

    def _one_hot_event_id(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if "EventID" not in df.columns:
            return df
        
        eids_series = pd.to_numeric(df["EventID"], errors='coerce').fillna(0).astype(int)
        if fit:
            self.event_ids = [int(x) for x in sorted(list(eids_series.unique()))]
            
        for eid in self.event_ids:
            df[f"EventID_{eid}"] = (eids_series == eid).astype(float)
            
        df = df.drop(columns=["EventID"])
        return df

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
            
        # Serialize event_ids list
        serialized_encoders["__event_ids__"] = self.event_ids
            
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
            
        pp.event_ids = serialized_encoders.get("__event_ids__", [])
        
        pp._encoders = {}
        for col, encoded_classes in serialized_encoders.items():
            if col == "__event_ids__":
                continue
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
        numeric_cols = {"destinationport", "querystatus"}
        for col in df.columns:
            if col.lower() in numeric_cols or pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].fillna("Missing").astype(str))
                self._encoders[col] = le
            else:
                le = self._encoders.get(col)
                if le is None:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].fillna("Missing").astype(str))
                else:
                    s = df[col].fillna("Missing").astype(str)
                    unseen = ~s.isin(le.classes_)
                    s[unseen] = le.classes_[0]
                    df[col] = le.transform(s)
        # Ensure all columns have no remaining NaNs
        df = df.fillna(0.0)
        return df

    def _scale(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        df = df.fillna(0.0)
        if fit:
            self._scaler = MinMaxScaler()
            scaled = self._scaler.fit_transform(df)
            return np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)

        # Align columns to what the scaler was fitted on
        if hasattr(self._scaler, "feature_names_in_"):
            for col in self._scaler.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self._scaler.feature_names_in_]
        scaled = self._scaler.transform(df)
        return np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)

    def get_event_feature_mask(self) -> np.ndarray:
        """
        Build and return a binary mask matrix of shape (len(self.event_ids), n_features)
        where each row corresponds to an Event ID, and each column is 1.0 if the feature
        is relevant to that Event ID, and 0.0 otherwise.
        """
        if self._scaler is None:
            raise RuntimeError("Preprocessor is not fitted. Fit it first.")
            
        feature_names = list(self._scaler.feature_names_in_)
        n_features = len(feature_names)
        n_events = len(self.event_ids)
        
        mask = np.zeros((n_events, n_features), dtype=np.float32)
        
        event_keywords = {
            1: ["image", "commandline", "directory", "user", "integrity", "parent"],
            3: ["image", "user", "protocol", "ip", "host", "port"],
            5: ["image", "user", "integrity"],
            7: ["image", "imageloaded", "signed", "signature"],
            8: ["image", "startaddress", "startmodule", "startfunction"],
            11: ["image", "targetfilename", "isexecutable"],
            12: ["image", "targetobject", "eventtype"],
            13: ["image", "targetobject", "details", "eventtype"],
            15: ["image", "targetfilename", "hashes"],
            16: ["configuration"],
            17: ["image", "pipename"],
            18: ["image", "pipename"],
            22: ["image", "query"],
            26: ["image", "name"]
        }
        
        for i, eid in enumerate(self.event_ids):
            keywords = event_keywords.get(eid, ["image"])
            for j, col in enumerate(feature_names):
                col_lower = col.lower()
                
                if col == f"EventID_{eid}":
                    mask[i, j] = 1.0
                    continue
                elif col.startswith("EventID_"):
                    mask[i, j] = 0.0
                    continue
                
                is_active = any(kw in col_lower for kw in keywords)
                if is_active:
                    mask[i, j] = 1.0
                    
        return mask
