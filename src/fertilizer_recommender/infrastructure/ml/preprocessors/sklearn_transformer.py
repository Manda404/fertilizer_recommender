"""
sklearn_transformer.py

Pourquoi ce fichier existe ?
- Centraliser le preprocessing ML (num + cat).
- Éviter de reconstruire des encoders partout.

À quoi ça sert réellement ?
- Transformer un DataFrame brut en matrice ML.
- Garantir que train et test utilisent le même preprocessing.

Est-ce critique ?
OUI. Sans pipeline propre = data leakage assuré.
"""

from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class SklearnFeatureTransformer:
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
            ]
        )

    def fit(self, df: pd.DataFrame):
        self.transformer.fit(df)
        return self

    def transform(self, df: pd.DataFrame):
        return self.transformer.transform(df)

    def fit_transform(self, df: pd.DataFrame):
        return self.transformer.fit_transform(df)