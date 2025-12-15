"""
metric_service.py

Pourquoi ce fichier existe ?
- La compétition Kaggle évalue avec MAP@3 : c'est une règle métier du projet.
- La métrique doit être dans le domaine (domain), car elle définit “ce qu'est une bonne prédiction”.

À quoi ça sert réellement ?
- Évaluer en local exactement comme Kaggle.
- Comparer des modèles / features / pipelines sans dépendre de Kaggle.

Ce fichier est-il “très utile” ?
Oui, critique : si tu calcules mal MAP@3, tu optimises dans le mauvais sens.
"""

from __future__ import annotations
from typing import List, Sequence


def average_precision_at_k(y_true: str, y_pred_topk: Sequence[str], k: int = 3) -> float:
    """
    AP@K pour un seul exemple.
    - y_true : la classe correcte
    - y_pred_topk : liste ordonnée de prédictions (top-k)

    Règle Kaggle:
    - On marque 1/rank si y_true est présent dans les k premières prédictions.
    - Sinon 0.
    - Si doublons, le premier match suffit (les suivants sont ignorés).

    Exemple:
    y_true="A"
    y_pred=["B","A","C"] -> AP@3 = 1/2 = 0.5
    """
    seen = set()
    for i, pred in enumerate(y_pred_topk[:k], start=1):
        if pred in seen:
            continue
        seen.add(pred)
        if pred == y_true:
            return 1.0 / i
    return 0.0


def map_at_k(y_true: Sequence[str], y_pred_topk: Sequence[Sequence[str]], k: int = 3) -> float:
    """
    MAP@K sur un dataset.

    Args:
        y_true: liste des labels vrais
        y_pred_topk: liste de listes (top-k prédictions pour chaque exemple)
        k: cutoff (3 pour la compétition)

    Returns:
        float: moyenne des AP@K
    """
    if len(y_true) != len(y_pred_topk):
        raise ValueError("y_true et y_pred_topk doivent avoir la même longueur.")

    scores = [average_precision_at_k(t, p, k=k) for t, p in zip(y_true, y_pred_topk)]
    return sum(scores) / len(scores) if scores else 0.0