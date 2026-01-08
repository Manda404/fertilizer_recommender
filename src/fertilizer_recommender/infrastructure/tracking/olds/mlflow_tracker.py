"""
mlflow_tracker.py

Pourquoi ce fichier existe ?
- Implémentation concrète du port ExperimentTracker avec MLflow.
- Toute dépendance à mlflow reste confinée à l'infrastructure.

À quoi ça sert ?
- start_run, log_params, log_metrics, log_artifact, set_experiment.

Très utile ?
Oui. C'est l’adaptateur MLOps central.
"""

from __future__ import annotations

import mlflow
from typing import Any, Mapping
from fertilizer_recommender.domain.interfaces.experiment_tracker import ExperimentTracker
from fertilizer_recommender.infrastructure.tracking.mlflow_setup import MLflowConfigurator
from loguru import logger

""""
class MLflowExperimentTracker(ExperimentTracker):
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)

    def set_experiment(self, experiment_name: str) -> None:
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> Any:
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)
"""


class MLflowExperimentTracker(ExperimentTracker):
    """
    Implémentation MLflow du port de tracking d’expériences.
    """

    def __init__(self):
        self.logger = logger
        self.client, self.artifact_location = MLflowConfigurator().configure()

    def setup_experiment(self, name: str) -> str:
        existing = self.client.get_experiment_by_name(name)

        if existing:
            exp_id = existing.experiment_id
            logger.info(f"Expérience existante : {name}")
        else:
            exp_id = self.client.create_experiment(
                name=name,
                artifact_location=self.artifact_location,
            )
            logger.info(
                f"Expérience créée : {name} "
                f"(artifact_location={self.artifact_location})"
            )

        mlflow.set_experiment(experiment_id=exp_id)
        return exp_id

    def start_run(self, run_name: str | None = None) -> None:
        mlflow.start_run(run_name=run_name)

    def log_params(self, params: Mapping[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    def end_run(self) -> None:
        """
        Termine la run MLflow active (si existante).
        """
        if mlflow.active_run() is not None:
            mlflow.end_run()