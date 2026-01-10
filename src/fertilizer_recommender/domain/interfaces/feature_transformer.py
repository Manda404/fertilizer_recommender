"""
feature_transformer.py

Pourquoi ce fichier existe ?
- Définir un contrat pour toute transformation de features.
- Découpler l'application des implémentations ML concrètes.

À quoi ça sert réellement ?
- Permettre l'injection de FeaturePipeline, sklearn, custom FE…
- Garantir une API uniforme (fit / transform).

Très utile ?
OUI. C’est une brique clé de la Clean Architecture ML.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class FeatureTransformer(ABC):
    @abstractmethod
    def fit(self, X: Any) -> "FeatureTransformer":
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: Any) -> Any:
        raise NotImplementedError

    def fit_transform(self, X: Any) -> Any:
        self.fit(X)
        return self.transform(X)