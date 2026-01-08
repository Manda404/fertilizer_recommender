"""
feature_engineering.py

Pourquoi ce fichier existe ?
- Centraliser toutes les créations de features dérivées.
- Éviter le chaos des features créées dans les notebooks.

À quoi ça sert réellement ?
- Ajouter des signaux non linéaires forts aux modèles.
- Rendre explicites des relations métier implicites.

Très utile ?
✅ OUI. Sur Kaggle tabulaire, c’est souvent le facteur clé de performance.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


class FeatureEngineer:
    def __init__(
        self,
        enable_ratios: bool = True,
        enable_interactions: bool = True,
        enable_transforms: bool = True,
    ):
        self.enable_ratios = enable_ratios
        self.enable_interactions = enable_interactions
        self.enable_transforms = enable_transforms
        self.logger = logger

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting feature engineering")
        df = df.copy()
        initial_n_features = df.shape[1]

        # =========================
        # 1. Ratios NPK
        # =========================
        if self.enable_ratios:
            self.logger.info("Generating NPK ratio features")
            eps = 1e-6

            df["N_to_P"] = df["Nitrogen"] / (df["Phosphorous"] + eps)
            df["N_to_K"] = df["Nitrogen"] / (df["Potassium"] + eps)
            df["P_to_K"] = df["Phosphorous"] / (df["Potassium"] + eps)

            df["NPK_sum"] = (
                df["Nitrogen"] + df["Phosphorous"] + df["Potassium"]
            )

            df["N_ratio"] = df["Nitrogen"] / (df["NPK_sum"] + eps)
            df["P_ratio"] = df["Phosphorous"] / (df["NPK_sum"] + eps)
            df["K_ratio"] = df["Potassium"] / (df["NPK_sum"] + eps)

            self.logger.info("NPK ratio features added (7 features)")

        # =========================
        # 2. Interactions climat / sol
        # =========================
        if self.enable_interactions:
            self.logger.info("Generating climate × soil interaction features")

            df["Temp_x_Humidity"] = df["Temperature"] * df["Humidity"]
            df["Humidity_x_Moisture"] = df["Humidity"] * df["Moisture"]
            df["Temp_x_Moisture"] = df["Temperature"] * df["Moisture"]

            self.logger.info("Climate/soil interaction features added (3 features)")

        # =========================
        # 3. Transformations non linéaires (robustes)
        # =========================
        if self.enable_transforms:
            self.logger.info("Generating non-linear transformed features")

            df["log_Nitrogen"] = np.log1p(df["Nitrogen"])
            df["log_Phosphorous"] = np.log1p(df["Phosphorous"])
            df["log_Potassium"] = np.log1p(df["Potassium"])

            df["sqrt_Moisture"] = np.sqrt(df["Moisture"].clip(lower=0))
            df["sqrt_Rainfall"] = np.sqrt(df["Rainfall"].clip(lower=0))

            self.logger.info("Non-linear transformed features added (5 features)")

        added_features = df.shape[1] - initial_n_features
        self.logger.info(
            f"Feature engineering completed — {added_features} features added "
            f"(total: {df.shape[1]})"
        )

        return df