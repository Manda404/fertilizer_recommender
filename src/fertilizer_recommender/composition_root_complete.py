"""
composition_root.py

RÔLE FONDAMENTAL
================
Ce fichier est le POINT UNIQUE où l’on :
- charge les configurations
- fixe la reproductibilité (seed)
- choisit les implémentations concrètes (infra)
- assemble les dépendances
- retourne des use cases PRÊTS À L’EMPLOI

AUCUNE logique métier.
AUCUN calcul ML.
AUCUN pandas / sklearn direct.

➡️ Si ce fichier disparaît, toute l’architecture s’écroule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

# =========================
# Utils & config
# =========================
from fertilizer_recommender.infrastructure.utils.config_loader import load_yaml_config
from fertilizer_recommender.infrastructure.utils.seed import set_global_seed

# =========================
# Repositories & tracking
# =========================
from fertilizer_recommender.infrastructure.repositories.dataset_repository_impl import CsvDatasetRepository
from fertilizer_recommender.infrastructure.repositories.model_repository_impl import JoblibModelRepository
from fertilizer_recommender.infrastructure.tracking.mlflow_tracker import MLflowExperimentTracker

# =========================
# Domain entities
# =========================
from fertilizer_recommender.domain.entities.fertilizer_features import FertilizerFeaturesSchema

# =========================
# Use cases
# =========================
from fertilizer_recommender.application.use_cases.prepare_dataset import PrepareDatasetUseCase
from fertilizer_recommender.application.use_cases.train_with_cv import TrainWithCVUseCase
from fertilizer_recommender.application.use_cases.train_final_model import TrainFinalModelUseCase
from fertilizer_recommender.application.use_cases.build_submission import BuildSubmissionUseCase
from fertilizer_recommender.application.use_cases.evaluate_model import EvaluateModelUseCase

# =========================
# ML building blocks
# =========================
from fertilizer_recommender.infrastructure.ml.cv.splitter import make_stratified_kfold
from fertilizer_recommender.infrastructure.ml.preprocessors.sklearn_transformer import SklearnFeatureTransformer
from fertilizer_recommender.infrastructure.ml.preprocessors.feature_engineering import FeatureEngineer
from fertilizer_recommender.infrastructure.ml.preprocessors.feature_pipeline import FeaturePipeline
from fertilizer_recommender.infrastructure.ml.pipelines.training_pipeline import TrainingPipeline

from fertilizer_recommender.infrastructure.ml.models.baseline_logreg import BaselineLogisticRegression
from fertilizer_recommender.infrastructure.ml.models.catboost_multiclass import CatBoostMulticlass
from fertilizer_recommender.infrastructure.ml.models.lightgbm_multiclass import LightGBMMulticlass
from fertilizer_recommender.infrastructure.ml.models.xgboost_multiclass import XGBoostMulticlass


# ======================================================
# 1. Chargement des configs + seed
# ======================================================

def load_all_configs(
    training_cfg: str = "configs/training.yaml",
    models_cfg: str = "configs/models.yaml",
    features_cfg: str = "configs/features.yaml",
    mlflow_cfg: str = "configs/mlflow.yaml",
):
    cfg_train = load_yaml_config(training_cfg)
    cfg_models = load_yaml_config(models_cfg)
    cfg_features = load_yaml_config(features_cfg)
    cfg_mlflow = load_yaml_config(mlflow_cfg)

    set_global_seed(cfg_train["project"]["seed"])

    return cfg_train, cfg_models, cfg_features, cfg_mlflow


# ======================================================
# 2. Schéma des features (métier)
# ======================================================

def build_feature_schema() -> FertilizerFeaturesSchema:
    return FertilizerFeaturesSchema(
        numeric_features=[
            "Temperature",
            "Humidity",
            "Moisture",
            "Nitrogen",
            "Potassium",
            "Phosphorous",
        ],
        categorical_features=[
            "Soil Type",
            "Crop Type",
        ],
    )


# ======================================================
# 3. Dataset repository
# ======================================================

def build_dataset_repository(cfg_train) -> CsvDatasetRepository:
    return CsvDatasetRepository(
        data_dir=Path(cfg_train["paths"]["data_raw_dir"]),
        train_file=cfg_train["data"]["train_file"],
        test_file=cfg_train["data"]["test_file"],
    )


# ======================================================
# 4. PrepareDataset use case
# ======================================================

def build_prepare_dataset_use_case(cfg_train) -> PrepareDatasetUseCase:
    schema = build_feature_schema()
    repo = build_dataset_repository(cfg_train)

    return PrepareDatasetUseCase(
        dataset_repository=repo,
        schema=schema,
        target_col=cfg_train["data"]["target_col"],
    )


# ======================================================
# 5. Feature pipeline (FE + preprocessing)
# ======================================================

def build_feature_pipeline(cfg_train, cfg_features) -> FeaturePipeline:
    schema = build_feature_schema()

    feature_engineer = FeatureEngineer(
        enable_ratios=cfg_features["feature_engineering"]["enable_ratios"],
        enable_interactions=cfg_features["feature_engineering"]["enable_interactions"],
    )

    numeric_features = (
        schema.numeric_features
        + [
            "N_to_P", "N_to_K", "P_to_K",
            "NPK_sum", "N_ratio", "P_ratio", "K_ratio",
            "Temp_x_Humidity", "Humidity_x_Moisture", "Temp_x_Moisture",
        ]
    )

    transformer = SklearnFeatureTransformer(
        numeric_features=numeric_features,
        categorical_features=schema.categorical_features,
    )

    return FeaturePipeline(
        feature_engineer=feature_engineer,
        transformer=transformer,
    )


# ======================================================
# 6. Pipeline factories (modèles interchangeables)
# ======================================================

def make_pipeline_factory(
    model_name: str,
    cfg_train,
    cfg_models,
    cfg_features,
    n_classes: int,
) -> Callable[[], TrainingPipeline]:

    def factory() -> TrainingPipeline:
        feature_pipeline = build_feature_pipeline(cfg_train, cfg_features)

        if model_name == "logreg":
            model = BaselineLogisticRegression(
                random_state=cfg_train["project"]["seed"]
            )

        elif model_name == "catboost":
            model = CatBoostMulticlass(
                random_state=cfg_train["project"]["seed"],
                **cfg_models["catboost"],
            )

        elif model_name == "lightgbm":
            model = LightGBMMulticlass(
                num_class=n_classes,
                random_state=cfg_train["project"]["seed"],
                **cfg_models["lightgbm"],
            )

        elif model_name == "xgboost":
            model = XGBoostMulticlass(
                num_class=n_classes,
                random_state=cfg_train["project"]["seed"],
                **cfg_models["xgboost"],
            )

        else:
            raise ValueError(f"Modèle inconnu: {model_name}")

        return TrainingPipeline(
            transformer=feature_pipeline,
            model=model,
        )

    return factory


# ======================================================
# 7. TrainWithCV use case
# ======================================================

def build_train_with_cv_use_case(
    model_name: str,
    X,
):
    cfg_train, cfg_models, cfg_features, cfg_mlflow = load_all_configs()

    tracker = MLflowExperimentTracker(cfg_mlflow["mlflow"]["tracking_uri"])
    tracker.set_experiment(cfg_mlflow["mlflow"]["experiment_name"])

    def splitter_factory():
        return make_stratified_kfold(
            n_splits=cfg_train["training"]["n_splits"],
            seed=cfg_train["project"]["seed"],
        )

    pipeline_factory = make_pipeline_factory(
        model_name=model_name,
        cfg_train=cfg_train,
        cfg_models=cfg_models,
        cfg_features=cfg_features,
        n_classes=len(set(X[cfg_train["data"]["target_col"]])),
    )

    return TrainWithCVUseCase(
        tracker=tracker,
        splitter_factory=splitter_factory,
        pipeline_factory=pipeline_factory,
        top_k=cfg_train["training"]["top_k"],
    )


# ======================================================
# 8. Entraînement final + persistance
# ======================================================

def build_train_final_model_use_case(model_name: str):
    cfg_train, cfg_models, cfg_features, _ = load_all_configs()

    model_repo = JoblibModelRepository(
        models_dir=Path(cfg_train["paths"]["models_dir"])
    )

    pipeline_factory = make_pipeline_factory(
        model_name=model_name,
        cfg_train=cfg_train,
        cfg_models=cfg_models,
        cfg_features=cfg_features,
        n_classes=None,  # pas nécessaire ici
    )

    pipeline = pipeline_factory()

    return TrainFinalModelUseCase(
        pipeline=pipeline,
        model_repository=model_repo,
    )


# ======================================================
# 9. Évaluation & submission
# ======================================================

def build_evaluate_model_use_case(pipeline):
    cfg_train, _, _, _ = load_all_configs()
    return EvaluateModelUseCase(
        pipeline=pipeline,
        top_k=cfg_train["training"]["top_k"],
    )


def build_submission_use_case():
    cfg_train, _, _, _ = load_all_configs()

    model_repo = JoblibModelRepository(
        models_dir=Path(cfg_train["paths"]["models_dir"])
    )

    return BuildSubmissionUseCase(
        model_repository=model_repo,
        id_col=cfg_train["data"]["id_col"],
        top_k=cfg_train["training"]["top_k"],
    )