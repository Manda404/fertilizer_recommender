"""
model_repository.py

Pourquoi ce fichier existe ?
- Le domaine ne doit pas savoir COMMENT un modèle est sauvegardé.
- Pickle, joblib, S3, MLflow Registry → ce n’est PAS son problème.

À quoi ça sert réellement ?
- Définir un CONTRAT pour sauvegarder / charger un modèle ML.

Est-ce critique ?
OUI. Sans ce port, ton application est couplée au stockage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ModelRepository(ABC):
    @abstractmethod
    def save(self, model: Any, name: str) -> None:
        """Sauvegarde un modèle"""
        raise NotImplementedError

    @abstractmethod
    def load(self, name: str) -> Any:
        """Charge un modèle"""
        raise NotImplementedError