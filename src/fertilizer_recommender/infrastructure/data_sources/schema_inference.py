"""
schema_inference.py

Pourquoi ce fichier existe ?
- Inférer automatiquement le schéma d’un dataset tabulaire.
- Réduire le hard-coding des features.

À quoi ça sert réellement ?
- Identifier num / cat proprement.
- Sécuriser l’entraînement sur données inconnues.

Très utile ?
OUI. Indispensable pour robustesse et généralisation.
"""

from __future__ import annotations
import pandas as pd
from typing import List, Tuple


def infer_feature_types(
    df: pd.DataFrame,
    target_col: str | None = None,
    id_col: str | None = None,
) -> Tuple[List[str], List[str]]:
    """
    Infère les colonnes numériques et catégorielles.

    Returns:
        numeric_features, categorical_features
    """
    cols = df.columns.tolist()

    excluded = set(filter(None, [target_col, id_col]))
    features = [c for c in cols if c not in excluded]

    numeric_features = df[features].select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        c for c in features if c not in numeric_features
    ]

    return numeric_features, categorical_features