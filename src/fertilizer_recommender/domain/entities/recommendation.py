"""
recommendation.py

Pourquoi ce fichier existe ?
- Formaliser une recommandation métier (Top-K fertilisants).
- Ne pas retourner des listes anonymes de strings.

À quoi ça sert réellement ?
- Structurer la sortie des prédictions.
- Faciliter la génération de submission / API / reporting.

Très utile ?
OUI. C’est la sortie métier principale du système.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from fertilizer_recommender.domain.entities.fertilizer_label import FertilizerLabel


@dataclass(frozen=True)
class Recommendation:
    """
    Recommandation métier Top-K.
    """
    labels: List[FertilizerLabel]

    def as_strings(self) -> List[str]:
        return [label.name for label in self.labels]

    def as_kaggle_string(self) -> str:
        """
        Format exigé par Kaggle : "label1 label2 label3"
        """
        return " ".join(self.as_strings())