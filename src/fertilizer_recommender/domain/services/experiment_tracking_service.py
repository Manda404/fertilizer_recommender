# src/fertilizer_recommender/domain/services/experiment_tracking_service.py
from contextlib import contextmanager
from loguru import logger
from typing import Any, Mapping
from fertilizer_recommender.domain.interfaces.experiment_tracker import (
    ExperimentTracker
)

class ExperimentTrackingService:
    """
    Service applicatif orchestrant le tracking d'expériences ML.

    - Utilisé par les cas d'usage (entraînement, évaluation, etc.)
    - Ne dépend d'aucune technologie (pas de MLflow ici)
    - Pilote le cycle complet : start → log → end
    """

    def __init__(self, tracker: ExperimentTracker):
        self._tracker = tracker
        self._logger = logger
        self._logger.info("ExperimentTrackingService initialisé.")

    # -----------------------------------------
    # Cycle de vie des expériences
    # -----------------------------------------
    def start_experiment(self, *, experiment_name: str, run_name: str) -> None:
        """Démarre une nouvelle expérience et une run associée."""
        self._logger.info(
            f"Initialisation de l'expérience '{experiment_name}' (run='{run_name}')"
        )
        # Sécurité : fermeture d’une run précédente
        self._tracker.end_run()  # sécurité (si run précédente ouverte)
        self._tracker.setup_experiment(experiment_name)
        self._tracker.start_run(run_name)

    # -----------------------------------------
    # Logging contextuel
    # -----------------------------------------
    def log_training_context(self, *, model_name: str, params: Mapping[str, Any]) -> None:
        """Log des informations relatives à l'entraînement."""
        self._logger.debug(f"Enregistrement du contexte d'entraînement pour {model_name}")
        self._tracker.log_params({"model": model_name, **params})

    def log_evaluation(self, metrics: Mapping[str, float]) -> None:
        """Log des métriques d'évaluation."""
        self._logger.info(f"Métriques d'évaluation : {metrics}")
        self._tracker.log_metrics(metrics)

    def log_artifact(self, path: str) -> None:
        """Log d'un artefact produit par le modèle (fichier)."""
        self._logger.debug(f"Enregistrement d'un artefact : {path}")
        self._tracker.log_artifact(path)

    # -----------------------------------------
    # Fermeture de la session
    # -----------------------------------------
    def close(self) -> None:
        """Fin de l'expérience en cours."""
        self._logger.info("Fermeture de l'expérience.")
        self._tracker.end_run()


    # -----------------------------------------
    # API CONTEXTUELLE (recommandée)
    # -----------------------------------------
    @contextmanager
    def experiment(self, *, experiment_name: str, run_name: str):
        """
        Point d’entrée UNIQUE pour les use cases.
        """
        self.start_experiment(
            experiment_name=experiment_name,
            run_name=run_name,
        )
        try:
            yield
        finally:
            self.close()