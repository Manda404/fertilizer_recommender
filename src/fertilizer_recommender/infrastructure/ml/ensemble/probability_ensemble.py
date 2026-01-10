"""
probability_ensemble.py

Pourquoi ce fichier existe ?
- Implémenter un ensemble de modèles entraînés.
- Abstraire la logique multi-modèles.

À quoi ça sert réellement ?
- Charger plusieurs pipelines
- Calculer predict_proba pour chacun
- Retourner une probabilité agrégée

Très utile ?
OUI. C’est la brique “Top Kaggle”.
"""

from __future__ import annotations
from typing import List
import numpy as np

from fertilizer_recommender.domain.services.ensemble_service import average_probabilities


class ProbabilityEnsemble:
    def __init__(self, pipelines: List):
        self.pipelines = pipelines

    def predict_proba(self, X_df) -> np.ndarray:
        probas = [pipeline.predict_proba(X_df) for pipeline in self.pipelines]
        return average_probabilities(probas)

    @property
    def classes_(self):
        return self.pipelines[0].classes_