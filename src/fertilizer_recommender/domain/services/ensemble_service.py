"""
ensemble_service.py

Pourquoi ce fichier existe ?
- Définir la règle métier d’agrégation des prédictions.
- L’ensemble est une logique de décision, pas un détail ML.

À quoi ça sert réellement ?
- Combiner plusieurs matrices de probabilités en une seule.
- Garantir un comportement cohérent pour MAP@3.

Très utile ?
OUI. Une mauvaise règle d’ensemble = perte de score Kaggle.
"""

from __future__ import annotations
import numpy as np
from typing import List


def average_probabilities(probabilities: List[np.ndarray]) -> np.ndarray:
    """
    Moyenne simple des probabilités (blending).

    Args:
        probabilities: liste de matrices (n_samples, n_classes)

    Returns:
        np.ndarray: matrice moyennée
    """
    if not probabilities:
        raise ValueError("Aucune probabilité fournie pour l'ensemble.")

    return np.mean(probabilities, axis=0)