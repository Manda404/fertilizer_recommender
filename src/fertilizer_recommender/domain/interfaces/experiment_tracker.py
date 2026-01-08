# src/fertilizer_recommender/domain/interfaces/experiment_tracker.py
"""
experiment_tracker.py

Pourquoi ce fichier existe ?
- En Clean Architecture, l'application ne doit pas dépendre de MLflow.
- On définit donc un PORT (contrat) que l'infrastructure implémentera (MLflow, WandB, etc.).

À quoi ça sert réellement ?
- L'application "log" des métriques/params/artifacts sans connaître l'outil concret.

Très utile ?
Oui. Sans ce port, tu couples ton cœur applicatif à MLflow => dette technique directe.
"""

from __future__ import annotations
from typing import Any, Mapping, Protocol


class ExperimentTracker(Protocol):
    """
    Port (contrat) pour le tracking d’expériences ML.

    Toute implémentation DOIT respecter ce contrat
    pour pouvoir être utilisée par l’application.
    """

    def setup_experiment(self, name: str) -> str:
        """
        Crée ou récupère une expérience.

        Parameters
        ----------
        name : str
            Nom logique de l’expérience (ex: "health_lifestyle_diabetes").

        Returns
        -------
        str
            Identifiant unique de l’expérience.
        """
        ...

    def start_run(self, run_name: str | None = None) -> None:
        """
        Démarre une nouvelle run de tracking.

        Parameters
        ----------
        run_name : str | None
            Nom optionnel de la run.
        """
        ...

    def log_params(self, params: Mapping[str, Any]) -> None:
        """
        Log des paramètres (hyperparamètres, config).

        Parameters
        ----------
        params : Mapping[str, Any]
            Dictionnaire clé / valeur.
        """
        ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """
        Log des métriques numériques.

        Parameters
        ----------
        metrics : Mapping[str, float]
            Exemple : {"auc": 0.87, "f1": 0.78}
        """
        ...

    def log_artifact(self, path: str) -> None:
        """
        Log d’un artefact (fichier).

        Parameters
        ----------
        path : str
            Chemin du fichier à sauvegarder.
        """
        ...

    def end_run(self) -> None:
        """
        Termine la run active.
        """
        ...
