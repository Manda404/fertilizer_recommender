"""
inference_pipeline.py

Pourquoi ce fichier existe ?
- Séparer clairement l'entraînement de l'inférence.
- Éviter toute modification accidentelle de l’état du modèle en prédiction.

À quoi ça sert réellement ?
- Appliquer le preprocessing + modèle déjà entraîné.
- Garantir une inférence stable et reproductible.

Très utile ?
OUI. Indispensable en production et pour les submissions Kaggle.
"""

from __future__ import annotations


class InferencePipeline:
    def __init__(self, transformer, model):
        self.transformer = transformer
        self.model = model

    def predict_proba(self, X_df):
        """
        Prédiction probabiliste sans aucun fit.
        """
        X = self.transformer.transform(X_df)
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes_