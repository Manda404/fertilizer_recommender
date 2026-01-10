"""
tracing.py

Pourquoi ce fichier existe ?
- Ajouter du traçage léger sans polluer le code métier.
- Mesurer le temps d'exécution des étapes clés.

À quoi ça sert réellement ?
- Identifier les goulots d’étranglement.
- Préparer une intégration future avec OpenTelemetry.

Très utile ?
Utile en prod / debug avancé.
"""

from __future__ import annotations
import time
from contextlib import contextmanager
from fertilizer_recommender.infrastructure.observability.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def trace(step_name: str):
    start = time.time()
    logger.info(f"[TRACE] Start: {step_name}")
    yield
    duration = time.time() - start
    logger.info(f"[TRACE] End: {step_name} | {duration:.3f}s")