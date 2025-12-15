"""
training_pipeline.py

Pourquoi ce fichier existe ?
- Assembler preprocessing + modèle.
- Offrir une API simple au use case d'entraînement.

À quoi ça sert réellement ?
- Fit, predict_proba, accès aux classes.
- Faciliter le swap de modèles plus tard.

Est-ce critique ?
✅ OUI. C’est la brique ML centrale.
"""

from __future__ import annotations


class TrainingPipeline:
    def __init__(self, transformer, model):
        self.transformer = transformer
        self.model = model

    def fit(self, X_df, y):
        X = self.transformer.fit_transform(X_df)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X_df):
        X = self.transformer.transform(X_df)
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes_