"""
baseline_logreg.py

Pourquoi ce fichier existe ?
- Fournir un modèle simple, rapide, interprétable.
- Avoir une baseline fiable avant les modèles complexes.

À quoi ça sert réellement ?
- Vérifier que le pipeline fonctionne de bout en bout.
- Donner un point de comparaison (MAP@3).

Est-ce critique ?
Modèle remplaçable, mais pipeline indispensable.
"""

from __future__ import annotations
from sklearn.linear_model import LogisticRegression


class BaselineLogisticRegression:
    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes_
