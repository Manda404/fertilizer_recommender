"""
model_repository_impl.py

Pourquoi ce fichier existe ?
- Implémentation concrète du port ModelRepository.
- Gérer la persistance locale des pipelines ML.

À quoi ça sert réellement ?
- Sauvegarder le pipeline complet (preprocessing + modèle).
- Le recharger à l’identique pour l’inférence ou la submission.

Est-ce critique ?
OUI. C’est ce qui rend ton travail réutilisable.
"""

from __future__ import annotations
from pathlib import Path
import joblib

from fertilizer_recommender.domain.interfaces.model_repository import ModelRepository


class JoblibModelRepository(ModelRepository):
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, name: str) -> None:
        path = self.models_dir / f"{name}.joblib"
        joblib.dump(model, path)

    def load(self, name: str):
        path = self.models_dir / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Modèle introuvable: {path}")
        return joblib.load(path)