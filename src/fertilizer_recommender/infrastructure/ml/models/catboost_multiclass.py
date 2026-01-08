"""
catboost_multiclass.py

Pourquoi ce fichier existe ?
- CatBoost gère NATIVEMENT les variables catégorielles.
- Pas besoin de OneHot → souvent plus performant.

À quoi ça sert réellement ?
- Fournir un modèle multiclass CatBoost compatible avec nos pipelines.
- Être interchangeable avec les autres modèles (LogReg, LGBM…).

Est-ce critique ?
Oui, CatBoost est souvent SOTA sur ce type de dataset tabulaire.
"""

from __future__ import annotations
from catboost import CatBoostClassifier


class CatBoostMulticlass:
    def __init__(self, **kwargs):
        self.model = CatBoostClassifier(
            **kwargs,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes_