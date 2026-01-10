"""
seed.py

Pourquoi ce fichier existe ?
- Centraliser la gestion des seeds aléatoires.
- Garantir des résultats reproductibles.

À quoi ça sert réellement ?
- Fixer toutes les sources de hasard connues.
- Éviter des résultats non déterministes.

Très utile ?
OUI. Indispensable en ML sérieux.
"""

from __future__ import annotations
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)