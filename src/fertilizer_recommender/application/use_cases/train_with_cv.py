"""
train_with_cv.py

Pourquoi ce fichier existe ?
- Orchestrer la CV (splits -> fit -> predict -> MAP@3) + tracking, sans dépendances MLflow.
- Le notebook/CLI appelle un seul use case, pas 20 bouts de code.

À quoi ça sert ?
- Lance une CV complète
- Calcule MAP@3 par fold + moyenne
- Log MLflow via ExperimentTracker (port)

Très utile ?
Oui. C’est le “cerveau” de ton expérimentation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, List

import numpy as np
from loguru import logger

from fertilizer_recommender.domain.services.metric_service import map_at_k
from fertilizer_recommender.domain.services.ranking_service import predict_top_k
from fertilizer_recommender.domain.services.experiment_tracking_service import (
    ExperimentTrackingService
)


@dataclass(frozen=True)
class CVResult:
    fold_scores: List[float]
    mean_score: float


class TrainWithCVUseCase:
    """
    Orchestration complète d'une cross-validation + tracking d'expérience.
    """

    def __init__(
        self,
        experiment_service: ExperimentTrackingService,
        splitter_factory: Callable[[], Any],
        pipeline_factory: Callable[[], Any],
        top_k: int = 3,
    ):
        self.experiment_service = experiment_service
        self.splitter_factory = splitter_factory
        self.pipeline_factory = pipeline_factory
        self.top_k = top_k
        self.logger = logger

    def execute(
        self,
        X_df,
        y,
        *,
        experiment_name: str,
        run_name: str,
        params: Dict[str, Any],
    ) -> CVResult:
        splitter = self.splitter_factory()
        y_array = np.array(y)

        fold_scores: List[float] = []

        # Important ici:
        with self.experiment_service.experiment(
            experiment_name=experiment_name,
            run_name=run_name,
        ):
            self.experiment_service.log_training_context(
                model_name=params.get("model", "unknown_model"),
                params=params,
            )

            for fold, (tr_idx, va_idx) in enumerate(
                splitter.split(X_df, y_array), start=1
            ):
                X_tr = X_df.iloc[tr_idx]
                y_tr = y_array[tr_idx]
                X_va = X_df.iloc[va_idx]
                y_va = y_array[va_idx]

                self.logger.info(
                    f"[Fold {fold}] Démarrage "
                    f"(train={len(tr_idx)} obs, val={len(va_idx)} obs)"
                )

                self.logger.info(f"[Fold {fold}] Entraînement du modèle")
                pipeline = self.pipeline_factory()
                pipeline.fit(X_tr, y_tr)

                logger.info(f"[Fold {fold}] Prédiction et calcul du top-{self.top_k}")
                proba = pipeline.predict_proba(X_va)
                topk = predict_top_k(
                    proba=proba,
                    class_labels=pipeline.classes_,
                    k=self.top_k,
                )

                score = map_at_k(y_va.tolist(), topk, k=self.top_k)
                fold_scores.append(score)

                self.logger.success(
                    f"[Fold {fold}] Score MAP@{self.top_k} = {score:.4f}"
                )
                self.experiment_service.log_evaluation(
                    {f"map_{self.top_k}_fold{fold}": score}
                )

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            self.experiment_service.log_evaluation(
                {f"map_{self.top_k}_mean": mean_score}
            )

        return CVResult(
            fold_scores=fold_scores,
            mean_score=mean_score,
        )