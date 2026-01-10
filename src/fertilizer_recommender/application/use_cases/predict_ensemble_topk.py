"""
predict_ensemble_topk.py

Pourquoi ce fichier existe ?
- L’application ne doit pas savoir comment fonctionne l’ensemble.
- Elle demande simplement une prédiction top-K.

À quoi ça sert réellement ?
- Prédire TOP-3 à partir d’un ensemble de modèles.
- Réutilisable pour CV, test, submission.

Très utile ?
OUI.
"""

from __future__ import annotations
from fertilizer_recommender.domain.services.ranking_service import predict_top_k


class PredictEnsembleTopKUseCase:
    def __init__(self, ensemble, top_k: int):
        self.ensemble = ensemble
        self.top_k = top_k

    def execute(self, X_df):
        proba = self.ensemble.predict_proba(X_df)
        return predict_top_k(
            proba=proba,
            class_labels=self.ensemble.classes_,
            k=self.top_k,
        )