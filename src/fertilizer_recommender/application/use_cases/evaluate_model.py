"""
evaluate_model.py

Pourquoi ce fichier existe ?
- Séparer l'évaluation de l'entraînement.
- Centraliser les métriques et règles d’évaluation.

À quoi ça sert réellement ?
- Évaluer un pipeline ML sur un dataset donné.
- Retourner des métriques exploitables (MAP@3).

Très utile ?
OUI. Sans ce use case, ton évaluation est dispersée et fragile.
"""

from __future__ import annotations
from typing import Dict

from fertilizer_recommender.domain.services.metric_service import map_at_k
from fertilizer_recommender.domain.services.ranking_service import predict_top_k


class EvaluateModelUseCase:
    def __init__(self, pipeline, top_k: int):
        self.pipeline = pipeline
        self.top_k = top_k

    def execute(self, X_df, y_true) -> Dict[str, float]:
        proba = self.pipeline.predict_proba(X_df)
        topk = predict_top_k(
            proba=proba,
            class_labels=self.pipeline.classes_,
            k=self.top_k,
        )

        score = map_at_k(y_true, topk, k=self.top_k)

        return {
            f"map@{self.top_k}": score
        }