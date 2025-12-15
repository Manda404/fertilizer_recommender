"""
validation_service.py

Pourquoi ce fichier existe ?
- Centraliser les règles de validation métier des datasets.
- Éviter de dupliquer des checks partout (notebooks, scripts…).

À quoi ça sert réellement ?
- Vérifier colonnes, target, types attendus.
- Lever des erreurs explicites si le dataset est invalide.

Est-ce critique ?
OUI. C’est ton filet de sécurité.
"""

from __future__ import annotations
from typing import Sequence


class DatasetValidationError(ValueError):
    """Erreur levée si le dataset ne respecte pas le schéma métier."""


def validate_columns(
    dataset_columns: Sequence[str],
    expected_features: Sequence[str],
    target_col: str | None = None,
) -> None:
    """
    Vérifie que toutes les colonnes attendues sont présentes.

    Args:
        dataset_columns: colonnes du dataset
        expected_features: features attendues
        target_col: colonne cible (optionnelle)

    Raises:
        DatasetValidationError
    """
    missing = set(expected_features) - set(dataset_columns)
    if missing:
        raise DatasetValidationError(f"Colonnes manquantes: {missing}")

    if target_col and target_col not in dataset_columns:
        raise DatasetValidationError(f"Colonne cible absente: {target_col}")
