"""
prepare_dataset.py

Pourquoi ce fichier existe ?
- Orchestrer le chargement + validation du dataset.
- Le notebook ou la CLI n’appelle JAMAIS directement le repository.

À quoi ça sert réellement ?
- Charger train/test
- Vérifier qu’ils respectent le schéma métier
- Retourner des datasets “sains”

Est-ce critique ?
OUI. C’est le point d’entrée data du projet.
"""

from __future__ import annotations
from typing import Tuple, Any

from fertilizer_recommender.domain.entities.fertilizer_features import FertilizerFeaturesSchema
from fertilizer_recommender.domain.interfaces.dataset_repository import DatasetRepository
from fertilizer_recommender.domain.services.validation_service import validate_columns


class PrepareDatasetUseCase:
    def __init__(
        self,
        dataset_repository: DatasetRepository,
        schema: FertilizerFeaturesSchema,
        target_col: str,
    ):
        self.dataset_repository = dataset_repository
        self.schema = schema
        self.target_col = target_col

    def execute(self) -> Tuple[Any, Any]:
        train_df = self.dataset_repository.load_train_dataset()
        test_df = self.dataset_repository.load_test_dataset()

        validate_columns(
            dataset_columns=train_df.columns,
            expected_features=self.schema.all_features,
            target_col=self.target_col,
        )

        validate_columns(
            dataset_columns=test_df.columns,
            expected_features=self.schema.all_features,
        )

        return train_df, test_df