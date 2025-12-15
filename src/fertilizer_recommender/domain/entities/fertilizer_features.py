"""
fertilizer_features.py

Pourquoi ce fichier existe ?
- Définir le schéma MÉTIER des données attendues par le système.
- Séparer clairement “ce que l’on attend” de “comment on le charge”.

À quoi ça sert réellement ?
- Valider qu’un dataset est compatible avec le projet.
- Éviter les bugs silencieux (colonne manquante, faute de frappe, etc.).

Est-ce critique ?
OUI. Sans schéma explicite, ton modèle devient fragile et non maintenable.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FertilizerFeaturesSchema:
    """
    Schéma métier des colonnes attendues (hors target).
    """

    numeric_features: List[str]
    categorical_features: List[str]

    @property
    def all_features(self) -> List[str]:
        return self.numeric_features + self.categorical_features
