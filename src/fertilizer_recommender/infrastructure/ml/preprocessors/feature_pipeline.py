"""
feature_pipeline.py

POURQUOI CE FICHIER EXISTE
-------------------------
Ce fichier encapsule TOUTE la logique de transformation des features AVANT
l'entraînement ou l'inférence du modèle.

Il permet d'enchaîner :
1) Feature engineering métier (ratios, interactions, etc.)
2) Preprocessing ML (scaling, encoding, etc.)

sans que :
- les use cases
- les modèles
- les notebooks

aient à connaître les détails techniques.

À QUOI ÇA SERT RÉELLEMENT
-------------------------
- Garantir que train / CV / inference / submission utilisent EXACTEMENT
  les mêmes transformations.
- Éviter toute divergence entre entraînement et prédiction.
- Centraliser la logique de transformation des données.

NIVEAU D'IMPORTANCE
-------------------
CRITIQUE.
Sans ce fichier, tu introduis du data leakage ou des incohérences silencieuses.
"""

from __future__ import annotations


class FeaturePipeline:
    """
    Pipeline de transformation des features.

    Il respecte le contrat suivant :
    - fit() : apprend uniquement à partir des données (train)
    - transform() : applique les transformations sans apprendre
    - fit_transform() : raccourci standard

    Cette classe est volontairement SIMPLE :
    elle orchestre, elle ne calcule rien.
    """

    def __init__(self, feature_engineer, transformer):
        """
        Args:
            feature_engineer:
                Objet responsable du feature engineering métier
                (ex: ratios NPK, interactions climat-sol).

            transformer:
                Transformer ML (ex: SklearnFeatureTransformer)
                responsable de :
                - scaling
                - encoding
                - vectorisation finale
        """
        self.feature_engineer = feature_engineer
        self.transformer = transformer

    def fit(self, X_df):
        """
        Apprend les transformations à partir du dataset d'entraînement.

        Étapes :
        1) Création des features dérivées (feature_engineering)
        2) Apprentissage des paramètres du transformer ML
           (scaler, encoder, etc.)

        IMPORTANT :
        - Cette méthode ne doit JAMAIS être appelée en inference.
        """
        X_fe = self.feature_engineer.transform(X_df)
        self.transformer.fit(X_fe)
        return self

    def transform(self, X_df):
        """
        Applique les transformations apprises à un dataset.

        Étapes :
        1) Recréation EXACTE des features dérivées
        2) Transformation ML (scaling, encoding)

        IMPORTANT :
        - AUCUN apprentissage ici
        - 100 % déterministe
        """
        X_fe = self.feature_engineer.transform(X_df)
        return self.transformer.transform(X_fe)

    def fit_transform(self, X_df):
        """
        Raccourci standard utilisé uniquement à l'entraînement.

        Équivalent à :
        - fit()
        - puis transform()
        """
        X_fe = self.feature_engineer.transform(X_df)
        return self.transformer.fit_transform(X_fe)