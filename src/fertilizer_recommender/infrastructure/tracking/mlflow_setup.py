# infrastructure/tracking/mlflow_setup.py

import os

import mlflow
from loguru import logger
from fertilizer_recommender.infrastructure.utils.exceptions import (
    MLflowConfigurationError,
    MLflowSetupError,
)
from mlflow.tracking import MlflowClient


class MLflowConfigurator:
    """
    Initialise MLflow à partir de l'environnement système.

    - Vérifie la configuration requise.
    - Configure l'URI de tracking et l'URI des artefacts.
    - Vérifie l'accès au backend MLflow.
    """

    def __init__(self):
        self.logger = logger

        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.artifact_uri = os.getenv("MLFLOW_ARTIFACT_URI")

        if not self.tracking_uri or not self.artifact_uri:
            missing = ", ".join(
                var for var in ["MLFLOW_TRACKING_URI", "MLFLOW_ARTIFACT_URI"]
                if not os.getenv(var)
            )
            raise MLflowConfigurationError(f"Variables d'environnement manquantes : {missing}")

        self.logger.debug(f"Tracking URI : {self.tracking_uri}")
        self.logger.debug(f"Artifact URI : {self.artifact_uri}")

    def configure(self) -> tuple[MlflowClient, str]:
        """Configure MLflow et retourne le client + l'URI normalisée des artefacts."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)

            artifact_uri = (
                self.artifact_uri
                if self.artifact_uri.startswith(("file:", "http", "s3", "gs"))
                else f"file:{os.path.abspath(self.artifact_uri)}"
            )

            client = MlflowClient(self.tracking_uri)
            client.search_experiments()  # test connexion

            self.logger.info("MLflow configuré avec succès.")
            return client, artifact_uri

        except Exception as exc:
            raise MLflowSetupError("Échec d'initialisation MLflow.") from exc