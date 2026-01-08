"""
mlflow_setup.py
===============

Configuration technique pure de MLflow.

📌 Rôle
-------
- Lire les variables d’environnement
- Configurer le tracking URI
- Normaliser l’URI des artefacts
- Initialiser le client MLflow

⚠️ Aucun concept métier ici.
⚠️ Aucun port.
"""
"""
DOCTRINE — MLFLOW SETUP (INFRASTRUCTURE)
=======================================

RÔLE ARCHITECTURAL
------------------
Ce fichier appartient à la couche INFRASTRUCTURE.

Il est responsable UNIQUEMENT de la configuration technique
de MLflow à partir de l’environnement système.

RESPONSABILITÉ UNIQUE
---------------------
- Lire les variables d’environnement
- Configurer le tracking URI
- Normaliser l’artifact location
- Initialiser le client MLflow

INTERDICTIONS ABSOLUES
---------------------
- Aucune logique métier
- Aucune décision fonctionnelle
- Aucun appel aux use cases
- Aucune dépendance vers le domaine ou l’application
"""

import os
import mlflow
from mlflow.tracking import MlflowClient


class MLflowConfigurator:
    """
    Configure MLflow à partir des variables d’environnement.
    """

    def __init__(self):
        self.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        self.artifact_uri = os.environ.get("MLFLOW_ARTIFACT_URI")

        if not self.tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI manquant.")
        if not self.artifact_uri:
            raise ValueError("MLFLOW_ARTIFACT_URI manquant.")

    def configure(self) -> tuple[MlflowClient, str]:
        """
        Configure MLflow et retourne :
        - le client MLflow
        - l'artifact_location normalisée
        """
        mlflow.set_tracking_uri(self.tracking_uri)

        artifact_uri = self.artifact_uri
        if not artifact_uri.startswith(("file:", "http", "s3", "gs")):
            artifact_uri = f"file:{os.path.abspath(artifact_uri)}"

        client = MlflowClient(tracking_uri=self.tracking_uri)
        return client, artifact_uri
