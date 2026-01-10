"""
xgboost_multiclass.py

Pourquoi ce fichier existe ?
- XGBoost est un standard Kaggle pour le tabulaire.
- Très performant en multiclass avec softprob.

Choix d’architecture IMPORTANT :
- Tous les hyperparamètres passent via **kwargs
- Le code est totalement découplé du YAML

Très utile ?
OUI. Indispensable pour benchmark sérieux.
"""

from __future__ import annotations
import xgboost as xgb


class XGBoostMulticlass:
    def __init__(
        self,
        #num_class: int,
        **kwargs,
    ):
        self.model = xgb.XGBClassifier(
            #num_class=num_class,
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