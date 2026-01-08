# src/fertilizer_recommender/infrastructure/tracking/mlflow_tracker.py

from __future__ import annotations
import mlflow
from typing import Mapping, Any

from loguru import logger
from fertilizer_recommender.domain.interfaces.experiment_tracker import ExperimentTracker
from fertilizer_recommender.infrastructure.tracking.mlflow_setup import MLflowConfigurator


class MLflowExperimentTracker(ExperimentTracker):
    """
    Adaptateur MLflow du port ExperimentTrackingPort.

    - Gestion du cycle de vie des expériences et runs.
    - Enregistrement des métriques, paramètres et artefacts.
    """

    def __init__(self):
        self.logger = logger
        configurator = MLflowConfigurator()
        self.client, self.artifact_uri = configurator.configure()
        self.logger.info("MLflowExperimentTracker prêt.")

    # -------------------------
    # Expériences
    # -------------------------
    def setup_experiment(self, name: str) -> str:
        """
        Active une expérience MLflow.
        Si l'expérience existe mais est supprimée, elle est restaurée automatiquement.
        """

        experiment = self.client.get_experiment_by_name(name)

        # Cas 1 : l'expérience existe mais est supprimée
        if experiment and experiment.lifecycle_stage == "deleted":
            self.logger.warning("Expérience supprimée détectée. Restauration en cours.")
            self.client.restore_experiment(experiment.experiment_id)
            experiment = self.client.get_experiment_by_name(name)

        # Cas 2 : l'expérience est active → on la réutilise
        if experiment:
            exp_id = experiment.experiment_id

        # Cas 3 : l'expérience n'existe pas → création
        else:
            exp_id = self.client.create_experiment(
                name=name,
                artifact_location=self.artifact_uri,
            )
            self.logger.info(f"Expérience créée : {name}")

        self.logger.info(f"Expérience active : {name}")
        mlflow.set_experiment(experiment_id=exp_id)
        return exp_id

    # -------------------------
    # Runs
    # -------------------------
    def start_run(self, run_name: str | None = None) -> None:
        """Démarre une run MLflow en fermant la précédente si nécessaire."""
        if mlflow.active_run() is not None:
            self.logger.warning("Run déjà active détectée. Fermeture automatique.")
            mlflow.end_run()

        mlflow.start_run(run_name=run_name)
        self.logger.info(f"Run démarrée : {run_name}")

    def end_run(self) -> None:
        """Ferme la run active."""
        active = mlflow.active_run()
        if active:
            self.logger.info(f"Fermeture de la run : {active.info.run_id}")
            mlflow.end_run()

    # -------------------------
    # Logging
    # -------------------------
    def log_params(self, params: Mapping[str, Any]) -> None:
        self.logger.debug(f"Paramètres enregistrés : {params}")
        mlflow.log_params(params)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self.logger.debug(f"Métriques enregistrées : {metrics}")
        mlflow.log_metrics(metrics)

    def log_artifact(self, path: str) -> None:
        self.logger.debug(f"Artefact enregistré : {path}")
        mlflow.log_artifact(path)