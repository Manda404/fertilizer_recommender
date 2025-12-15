"""
dataset_repository.py

Pourquoi ce fichier existe ?
- Définir un CONTRAT pour charger des datasets.
- Le domaine ne doit JAMAIS savoir si les données viennent d’un CSV,
  d’une base SQL, d’un Data Lake ou de Kaggle API.

À quoi ça sert réellement ?
- Permettre de remplacer la source de données sans toucher
  ni au domaine, ni aux use cases.

Est-ce critique ?
OUI. C’est le cœur de la Clean Architecture.
"""

from __future__ import annotations
from typing import Protocol, Any


class DatasetRepository(Protocol):
    def load_train_dataset(self) -> Any:
        """Charge le dataset d'entraînement"""
        ...

    def load_test_dataset(self) -> Any:
        """Charge le dataset de test"""
        ...
