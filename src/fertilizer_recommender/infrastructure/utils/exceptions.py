"""
src/health_lifestyle_diabetes/exceptions.py

Exceptions personnalisées de l'application.

Objectifs :
-----------
1. Centraliser la gestion des erreurs.
2. Séparer clairement les erreurs techniques (Infrastructure)
   des erreurs métier (Domaine).
3. Garantir l'indépendance des couches (Clean Architecture).
4. Fournir des messages d'erreur explicites, traçables et maintenables.

Toutes les exceptions de l'application doivent hériter de `BaseAppError`.
"""

# ============================================================
# ===============   1. Exception Racine (Base)   ==============
# ============================================================


class BaseAppError(Exception):
    """
    Exception racine de l'application.

    Toutes les exceptions personnalisées doivent hériter de cette classe.
    Elle permet :
    - un catch global cohérent
    - un logging centralisé
    - une abstraction propre des erreurs
    """

    pass


# ============================================================
# ===============   2. Exceptions Infrastructure   ============
# ============================================================
# Responsabilité : technique uniquement (I/O, fichiers, config)
# Aucune règle métier ne doit apparaître ici.
# ============================================================

# ----------------------------
# Configuration & Logging
# ----------------------------


class ConfigLoadingError(BaseAppError):
    """
    Erreur lors du chargement ou du parsing d'une configuration
    (YAML, JSON, .env, etc.).
    """

    pass


class LoggerInitializationError(BaseAppError):
    """
    Erreur lors de l'initialisation du système de logging.
    """

    pass


# ----------------------------
# Accès aux données (Dataset)
# ----------------------------


class DatasetLoadingError(BaseAppError):
    """
    Erreur lors du chargement d'un dataset.
    (fichier manquant, corrompu ou format invalide)
    """

    pass


class DatasetSavingError(BaseAppError):
    """
    Erreur lors de la sauvegarde d'un dataset sur disque.
    """

    pass


# ----------------------------
# Persistance des modèles ML
# ----------------------------


class ModelLoadingError(BaseAppError):
    """
    Erreur lors du chargement d'un modèle ML.
    (fichier manquant, version incompatible, désérialisation impossible)
    """

    pass


class ModelSavingError(BaseAppError):
    """
    Erreur lors de la sauvegarde d'un modèle ML.
    """

    pass


# ============================================================
# ===============   3. Exceptions Domaine & Métier   ==========
# ============================================================
# Responsabilité : règles métier, logique ML, cohérence fonctionnelle
# Indépendantes de toute technologie.
# ============================================================

# ----------------------------
# Validation des données
# ----------------------------


class DatasetValidationError(BaseAppError):
    """
    Erreur lorsque les données d'entraînement ou de test
    ne respectent pas le schéma attendu.
    """

    pass


class FeatureValidationError(BaseAppError):
    """
    Erreur de validation des features en entrée de prédiction.
    (colonne manquante, valeur hors bornes, type invalide)
    """

    pass


# ----------------------------
# Cycle de vie du modèle ML
# ----------------------------


class ModelTrainingError(BaseAppError):
    """
    Erreur survenue pendant l'entraînement du modèle.
    (convergence impossible, données incohérentes, échec d'optimisation)
    """

    pass


class ModelPredictionError(BaseAppError):
    """
    Erreur lors de l'exécution d'une prédiction.
    (features mal formées, pipeline cassé, output invalide)
    """

    pass


# ----------------------------
# Métier Santé / Lifestyle
# ----------------------------


class InvalidPatientProfileError(BaseAppError):
    """
    Erreur métier : le profil patient est invalide ou incomplet
    au regard des règles métier (âge, IMC, données critiques manquantes).
    """

    pass


class FeatureEngineeringError(BaseAppError):
    """
    Erreur métier lors de la création ou transformation des features.
    (logique métier violée, calcul invalide)
    """

    pass


class PredictionServiceError(BaseAppError):
    """
    Erreur métier lors de l'exécution du service de prédiction final.
    (enchaînement de use cases incorrect, dépendances manquantes)
    """

    pass



# ----------------------------
# MLflow & Tracking d'expériences
# ----------------------------
class MLflowConfigurationError(BaseAppError):
    """Erreur liée à une mauvaise configuration MLflow."""
    pass

class MLflowSetupError(BaseAppError):
    """
    Erreur survenue lors de l'initialisation ou de la connexion à MLflow
    (DB corrompue, migration Alembic invalide, backend indisponible, etc.).
    """
    pass


class ExtractorMetricsError(BaseAppError):
    """
    Erreur survenue lors de l'initialisation ou de la connexion à MLflow
    (DB corrompue, migration Alembic invalide, backend indisponible, etc.).
    """
    pass


class StreamingDataError(BaseAppError):
    """
    Erreur lors du streaming des données.
    """
    pass


class XGBoostTrainingError(BaseAppError):
    """
    Erreur spécifique à l'entraînement du modèle XGBoost.
    """
    pass


class CatBoostTrainingError(BaseAppError):
    """
    Erreur spécifique à l'entraînement du modèle CatBoost.
    """
    pass


class LightGBMTrainingError(BaseAppError):
    """
    Erreur spécifique à l'entraînement du modèle LightGBM.
    """
    pass



# ============================================================
# ===============   4. Export Public (Best Practice)   ========
# ============================================================

__all__ = [
    # Base
    "BaseAppError",
    # Infrastructure
    "ConfigLoadingError",
    "LoggerInitializationError",
    "DatasetLoadingError",
    "DatasetSavingError",
    "ModelLoadingError",
    "ModelSavingError",
    # Domaine & Métier
    "DatasetValidationError",
    "FeatureValidationError",
    "ModelTrainingError",
    "ModelPredictionError",
    "InvalidPatientProfileError",
    "FeatureEngineeringError",
    "PredictionServiceError",
    # MLflow
    "MLflowConfigurationError",
    "MLflowSetupError",
    "ExtractorMetricsError",
    "StreamingDataError",
    "XGBoostTrainingError",
    "CatBoostTrainingError",
    "LightGBMTrainingError",
]