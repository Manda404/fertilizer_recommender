"""
lightgbm_multiclass.py

Pourquoi ce fichier existe ?
- LightGBM est extrêmement rapide et performant.
- Très bon compromis vitesse / score.

À quoi ça sert ?
- Tester un autre algorithme sans changer l’application.

Est-ce critique ?
Optionnel mais fortement recommandé pour le benchmarking.
"""

from __future__ import annotations
import lightgbm as lgb


class LightGBMMulticlass:
    def __init__(self, **kwargs):
        self.model = lgb.LGBMClassifier(
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