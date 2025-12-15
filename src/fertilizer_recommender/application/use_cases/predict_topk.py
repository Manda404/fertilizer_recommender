"""
predict_topk.py

Pourquoi ce fichier existe ?
- Séparer prédiction et entraînement.
- Garantir une sortie conforme Kaggle.

À quoi ça sert réellement ?
- Prédire les TOP-K labels.
- Réutilisable pour évaluation et submission.

Est-ce critique ?
OUI.
"""

from __future__ import annotations
from fertilizer_recommender.domain.services.ranking_service import predict_top_k


class PredictTopKUseCase:
    def __init__(self, pipeline, k: int):
        self.pipeline = pipeline
        self.k = k

    def execute(self, X_df):
        proba = self.pipeline.predict_proba(X_df)
        return predict_top_k(
            proba=proba,
            class_labels=self.pipeline.classes_,
            k=self.k,
        )
