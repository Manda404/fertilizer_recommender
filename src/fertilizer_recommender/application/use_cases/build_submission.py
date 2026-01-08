"""
build_submission.py

Pourquoi ce fichier existe ?
- Kaggle attend un format TRÈS précis.
- Toute la logique de formatting doit être centralisée.

À quoi ça sert réellement ?
- Prendre un modèle entraîné
- Prédire TOP-3 sur test
- Générer submission.csv

Est-ce critique ?
OUI. Une virgule mal placée = submission rejetée.
"""

from __future__ import annotations
import pandas as pd

from fertilizer_recommender.application.use_cases.predict_topk import PredictTopKUseCase


class BuildSubmissionUseCase:
    def __init__(self, model_repository, id_col: str, top_k: int):
        self.model_repository = model_repository
        self.id_col = id_col
        self.top_k = top_k

    def execute(self, model_name: str, test_df: pd.DataFrame, output_path: str):
        pipeline = self.model_repository.load(model_name)

        predictor = PredictTopKUseCase(pipeline, self.top_k)
        topk_preds = predictor.execute(test_df)

        submission = pd.DataFrame({
            "id": test_df[self.id_col],
            "Fertilizer Name": [" ".join(preds) for preds in topk_preds],
        })

        submission.to_csv(output_path, index=False)
        return submission