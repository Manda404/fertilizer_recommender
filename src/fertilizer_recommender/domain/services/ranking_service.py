"""
ranking_service.py

Pourquoi ce fichier existe ?
- Kaggle n’évalue PAS la classe prédite, mais un TOP-3 ordonné.
- La logique de ranking est une règle métier, pas du ML.

À quoi ça sert réellement ?
- Convertir des probabilités en recommandations top-K.
- Garantir un comportement cohérent partout (train, eval, submit).

Est-ce critique ?
OUI. Une erreur ici = score Kaggle faux.
"""

from __future__ import annotations
from typing import List, Sequence
import numpy as np


def predict_top_k(
    proba: np.ndarray,
    class_labels: Sequence[str],
    k: int = 3,
) -> List[List[str]]:
    """
    Convertit une matrice de probabilités en top-K labels ordonnés.

    Args:
        proba: shape (n_samples, n_classes)
        class_labels: mapping index -> label
        k: nombre de prédictions

    Returns:
        List[List[str]]: top-K labels par ligne
    """
    topk_indices = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    return [[class_labels[i] for i in row] for row in topk_indices]
