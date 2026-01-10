"""
composition_root.py

Pourquoi ce fichier existe ?
- Centraliser l’assemblage des dépendances.
- Respecter l’Inversion de Dépendance (Clean Architecture).

À quoi ça sert réellement ?
- Construire les use cases prêts à l’emploi.
- Fournir des objets configurés pour CLI / API / notebooks.

Très utile ?
OUI. C’est la “colle” de toute l’architecture.
"""

from __future__ import annotations
from pathlib import Path

from fertilizer_recommender.infrastructure.utils.config_loader import load_yaml_config
from fertilizer_recommender.infrastructure.utils.seed import set_global_seed
from fertilizer_recommender.infrastructure.repositories.dataset_repository_impl import CsvDatasetRepository
from fertilizer_recommender.infrastructure.repositories.model_repository_impl import JoblibModelRepository
from fertilizer_recommender.infrastructure.tracking.mlflow_tracker import MLflowExperimentTracker

from fertilizer_recommender.domain.entities.fertilizer_features import FertilizerFeaturesSchema
from fertilizer_recommender.application.use_cases.prepare_dataset import PrepareDatasetUseCase


def build_prepare_dataset_use_case(config_path: str = "configs/training.yaml"):
    cfg = load_yaml_config(config_path)

    set_global_seed(cfg["project"]["seed"])

    schema = FertilizerFeaturesSchema(
        numeric_features=[
            "Temperature", "Humidity", "Moisture",
            "Nitrogen", "Potassium", "Phosphorous",
        ],
        categorical_features=["Soil Type", "Crop Type"],
    )

    dataset_repo = CsvDatasetRepository(
        data_dir=Path(cfg["paths"]["data_raw_dir"]),
        train_file=cfg["data"]["train_file"],
        test_file=cfg["data"]["test_file"],
    )

    return PrepareDatasetUseCase(
        dataset_repository=dataset_repo,
        schema=schema,
        target_col=cfg["data"]["target_col"],
    )