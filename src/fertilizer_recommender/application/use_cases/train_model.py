"""
train_model.py

Pourquoi ce fichier existe ?
- Orchestrer l'entraînement sans exposer le ML à la présentation.

À quoi ça sert réellement ?
- Entraîner un pipeline ML complet.
- Retourner un objet entraîné prêt à prédire.

Est-ce critique ?
OUI. Toute la logique ML passe ici.
"""

from __future__ import annotations


class TrainModelUseCase:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def execute(self, X_df, y):
        return self.pipeline.fit(X_df, y)