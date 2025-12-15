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
from typing import Callable, Dict, Any, List, Tuple

import numpy as np

from fertilizer_recommender.domain.services.metric_service import map_at_k
from fertilizer_recommender.domain.services.ranking_service import predict_top_k
from fertilizer_recommender.domain.interfaces.experiment_tracker import ExperimentTracker


@dataclass(frozen=True)
class CVResult:
    fold_scores: List[float]
    mean_score: float


class TrainWithCVUseCase:
    def __init__(
        self,
        tracker: ExperimentTracker,
        splitter_factory: Callable[[], Any],
        pipeline_factory: Callable[[], Any],
        top_k: int = 3,
    ):
        self.tracker = tracker
        self.splitter_factory = splitter_factory
        self.pipeline_factory = pipeline_factory
        self.top_k = top_k

    def execute(
        self,
        X_df,
        y,
        run_name: str,
        params: Dict[str, Any],
        tags: Dict[str, str] | None = None,
    ) -> CVResult:
        splitter = self.splitter_factory()
        y_array = np.array(y)

        fold_scores: List[float] = []

        with self.tracker.start_run(run_name=run_name, tags=tags):
            self.tracker.log_params(params)

            for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_df, y_array), start=1):
                X_tr = X_df.iloc[tr_idx]
                y_tr = y_array[tr_idx]
                X_va = X_df.iloc[va_idx]
                y_va = y_array[va_idx]

                pipeline = self.pipeline_factory()
                pipeline.fit(X_tr, y_tr)

                proba = pipeline.predict_proba(X_va)
                topk = predict_top_k(proba=proba, class_labels=pipeline.classes_, k=self.top_k)

                score = map_at_k(y_va.tolist(), topk, k=self.top_k)
                fold_scores.append(score)

                self.tracker.log_metrics({f"map@{self.top_k}_fold": score}, step=fold)

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            self.tracker.log_metrics({f"map@{self.top_k}_mean": mean_score})

        return CVResult(fold_scores=fold_scores, mean_score=mean_score)
