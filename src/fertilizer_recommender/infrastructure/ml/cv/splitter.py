"""
splitter.py

Pourquoi ce fichier existe ?
- Standardiser la création des splits CV (StratifiedKFold, seed, shuffle).
- Garder sklearn en infrastructure.

À quoi ça sert ?
- Fournir les indices (train_idx, val_idx) de manière reproductible.

Très utile ?
Oui : reproductibilité + cohérence des évaluations.
"""

from __future__ import annotations
from sklearn.model_selection import StratifiedKFold


def make_stratified_kfold(n_splits: int, seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
