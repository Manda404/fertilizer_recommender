"""
train_final_model.py

Pourquoi ce fichier existe ?
- Entraîner le modèle FINAL (full train).
- Le sauvegarder immédiatement après entraînement.

À quoi ça sert réellement ?
- Produire LE modèle qui sera utilisé pour Kaggle ou la prod.
- Séparer clairement CV (étape 3) et entraînement final.

Est-ce critique ?
OUI. C’est la version “gold” du modèle.
"""

from __future__ import annotations
from fertilizer_recommender.domain.interfaces.model_repository import ModelRepository


class TrainFinalModelUseCase:
    def __init__(self, pipeline, model_repository: ModelRepository):
        self.pipeline = pipeline
        self.model_repository = model_repository

    def execute(self, X_df, y, model_name: str):
        self.pipeline.fit(X_df, y)
        self.model_repository.save(self.pipeline, model_name)
        return self.pipeline