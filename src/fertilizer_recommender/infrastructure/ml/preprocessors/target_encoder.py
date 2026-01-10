"""
target_encoder.py

Pourquoi ce fichier existe ?
- Centraliser l’encodage de la variable cible.
- Éviter les mappings implicites et non maîtrisés.

À quoi ça sert réellement ?
- Transformer des labels string en entiers.
- Garantir un mapping stable et réversible.

Très utile ?
Important pour modèles avancés / stacking.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np


class TargetEncoder:
    def __init__(self):
        self.label_to_index: Dict[str, int] = {}
        self.index_to_label: Dict[int, str] = {}

    def fit(self, y: List[str]):
        unique_labels = sorted(set(y))
        self.label_to_index = {label: i for i, label in enumerate(unique_labels)}
        self.index_to_label = {i: label for label, i in self.label_to_index.items()}
        return self

    def transform(self, y: List[str]) -> np.ndarray:
        return np.array([self.label_to_index[label] for label in y])

    def inverse_transform(self, y_idx: np.ndarray) -> List[str]:
        return [self.index_to_label[int(i)] for i in y_idx]