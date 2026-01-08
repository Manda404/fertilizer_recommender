"""
src/health_lifestyle_diabetes/exceptions.py
-------------------------------------------

Définition d'exceptions personnalisées utilisées dans l'application.

Ces exceptions servent à :
1. Centraliser la gestion des erreurs.
2. Fournir des erreurs métier (Domaine) ou techniques (Infrastructure) claires.
3. Maintenir l'indépendance des couches (respect de la Clean Architecture).
"""

# ============================================================
# ===============   1. Exceptions Racines (Base)   ===========
# ============================================================


class BaseAppError(Exception):
    """
    Exception racine pour toutes les erreurs personnalisées de l'application.
    Toutes les autres exceptions doivent hériter de celle-ci.
    """

    pass


# ============================================================
# ===============   2. Exceptions Infrastructure   ============
# ============================================================


class ConfigLoadingError(BaseAppError):
    """Erreur liée au chargement ou au parsing d'un fichier de configuration (e.g., YAML, .env)."""

    pass


class LoggerInitializationError(BaseAppError):
    """Erreur survenue lors de l'initialisation du système de logging."""

    pass


# --- Sous-catégorie: Data Access (Repository) ---


class DatasetLoadingError(BaseAppError):
    """Erreur lors du chargement d'un dataset (fichier manquant, corrompu, format incorrect)."""

    pass


class DatasetSavingError(BaseAppError):
    """Erreur lors de la sauvegarde d'un dataset sur disque."""

    pass


# --- Sous-catégorie: Modèles ML (Persistence) ---


class ModelLoadingError(BaseAppError):
    """Erreur lors du chargement d'un modèle ML (fichier manquant, incompatible)."""

    pass


class ModelSavingError(BaseAppError):
    """Erreur lors de la sauvegarde d'un modèle ML."""

    pass


# ============================================================
# ===============   3. Exceptions Domaine & Métier   ==========
# ============================================================


class DatasetValidationError(BaseAppError):
    """
    Erreur lorsqu'un dataset de training/testing ne respecte pas un schéma attendu.
    (Applicable lors du training ou de l'analyse des données brutes).
    """

    pass


class FeatureValidationError(BaseAppError):
    """
    Erreur levée en cas de non-conformité des features lors de l'INPUT de PRÉDICTION.
    (e.g., âge hors bornes, colonne manquante).
    """


class ModelTrainingError(BaseAppError):
    """Erreur survenue pendant l'entraînement du modèle (e.g., convergence impossible)."""

    pass


class ModelPredictionError(BaseAppError):
    """Erreur survenue lors de l'appel à la prédiction (e.g., données mal formées après FE)."""

    pass


class InvalidPatientProfileError(BaseAppError):
    """Erreur métier : les données d'un profil patient sont invalides ou incomplètes (règle métier abstraite)."""

    pass


class FeatureEngineeringError(BaseAppError):
    """Erreur métier lors de la création ou de la transformation des features."""

    pass


class PredictionServiceError(BaseAppError):
    """Erreur métier lors de l'exécution du service final de prédiction."""

    pass


# ============================================================
# ===============   Exportation (Best Practice)   ============
# ============================================================

__all__ = [
    "BaseAppError",
    "ConfigLoadingError",
    "LoggerInitializationError",
    "DatasetLoadingError",
    "DatasetSavingError",
    "ModelLoadingError",
    "ModelSavingError",
    "DatasetValidationError",
    "FeatureValidationError",
    "ModelTrainingError",
    "ModelPredictionError",
    "InvalidPatientProfileError",
    "FeatureEngineeringError",
    "PredictionServiceError",
]
