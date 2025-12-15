"""
logger.py

Pourquoi ce fichier existe ?
- Fournir un logger UNIQUE pour tout le projet (CLI, notebooks, pipelines ML).
- Avoir des logs lisibles, structurés et exploitables (debug + MLOps).
- Éviter toute configuration de logging dispersée dans le code.

Choix technique :
- loguru (meilleur DX que logging standard).
- Configuration centralisée.
- Compatible notebooks, scripts, CI, MLflow artifacts.

Ce fichier est-il critique ?
Oui. Sans logs propres, un projet ML devient rapidement incontrôlable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from loguru import logger
from typing import Final, Optional
from fertilizer_recommender.infrastructure.utils.root_finder import get_repository_root

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
DEFAULT_LOG_FILE_NAME: Final[str] = "fertilizer_recommender.log"

_LOGGER_INITIALIZED: bool = False


def setup_logger(
    *,
    project_name: str = "fertilizer_recommender",
    environment: str = "local",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file_name: str = DEFAULT_LOG_FILE_NAME,
) -> None:
    """
    Configure le logger global Loguru pour tout le projet.

    Design principles
    -----------------
    - Logger global unique (singleton Loguru)
    - Configuration explicite et centralisée
    - Compatible notebooks, CLI, batch jobs, MLflow
    - Logs console (lisibles) + fichiers (audit / debug)
    - Aucun impact sur le Domain layer

    Usage
    -----
    >>> from fertilizer_recommender.infrastructure.observability.logger import setup_logger
    >>> from loguru import logger
    >>> setup_logger(environment="notebook", level="DEBUG")
    >>> logger.info("Training started")

    Parameters
    ----------
    project_name : str
        Nom du projet (exposé dans les logs)
    environment : str
        Environnement d'exécution (local, dev, prod, kaggle, etc.)
    level : str
        Niveau de logs ("DEBUG", "INFO", "WARNING", ...)
    log_dir : Path | None
        Dossier où écrire les logs fichiers
        (par défaut: <repo_root>/logs)
    log_file_name : str
        Nom du fichier de log
    """
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return

    # -----------------------------------------------------------------
    # Resolve log directory
    # -----------------------------------------------------------------
    if log_dir is None:
        log_dir = get_repository_root() / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file: Path = log_dir / log_file_name

    # -----------------------------------------------------------------
    # Reset logger (CRITICAL)
    # -----------------------------------------------------------------
    logger.remove()

    # -----------------------------------------------------------------
    # Bind static context (available in every log)
    # -----------------------------------------------------------------
    logger.configure(
        extra={
            "project": project_name,
            "environment": environment,
        }
    )

    # -----------------------------------------------------------------
    # Console sink (human-readable)
    # -----------------------------------------------------------------
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{extra[project]}</cyan> | "
            "<magenta>{extra[environment]}</magenta> | "
            "{module}:{function}:{line} | "
            "<level>{message}</level>"
        ),
        enqueue=True,
    )

    # -----------------------------------------------------------------
    # File sink (audit / post-mortem / MLOps)
    # -----------------------------------------------------------------
    logger.add(
        log_file,
        level="DEBUG",
        rotation="20 MB",
        retention="30 days",
        compression="zip",
        serialize=True,   # JSON logs (MLflow / ELK friendly)
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    _LOGGER_INITIALIZED = True
