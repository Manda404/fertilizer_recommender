"""
dataset_repository_impl.py

Pourquoi ce fichier existe ?
- Implémenter concrètement le port DatasetRepository.
- Faire le lien entre le domaine et pandas.

À quoi ça sert réellement ?
- Charger train/test sans exposer pandas au domaine ou à l’application.

Est-ce critique ?
OUI. C’est l’adaptateur principal des données.
"""

from __future__ import annotations
from pathlib import Path
from pandas import DataFrame

from fertilizer_recommender.domain.interfaces.dataset_repository import DatasetRepository
from fertilizer_recommender.infrastructure.data_sources.csv_loader import load_csv


class CsvDatasetRepository(DatasetRepository):
    def __init__(self, data_dir: Path, train_file: str, test_file: str):
        self.data_dir = data_dir
        self.train_file = train_file
        self.test_file = test_file

    def load_train_dataset(self) -> DataFrame:
        return load_csv(self.data_dir / self.train_file)

    def load_test_dataset(self) -> DataFrame:
        return load_csv(self.data_dir / self.test_file)
