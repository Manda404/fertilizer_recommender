
---
# 🌱 Fertilizer Recommender  
**Predicting Optimal Fertilizers – Kaggle Competition**

---

## 🎯 Objectif du projet

Ce projet vise à **prédire le meilleur engrais** à recommander en fonction :
- des conditions météorologiques,
- du sol,
- du type de culture,

dans le cadre de la compétition Kaggle  
**“Predicting Optimal Fertilizers”** (métrique : **MAP@3**).

> ⚠️ Ce dépôt ne se limite pas à produire un score Kaggle.  
> Il démontre une **approche professionnelle, MLOps-ready**, basée sur la **Clean Architecture**, reproductible et maintenable.

---

## 🧠 Principes clés

### Clean Architecture appliquée au Machine Learning

Les dépendances suivent **strictement** le sens :

```

presentation → application → domain
infrastructure → domain (implémente les ports)

````

- `domain` : règles métier pures (aucune lib externe)
- `application` : orchestration des cas d’usage
- `infrastructure` : pandas, sklearn, CatBoost, MLflow, I/O
- `presentation` : CLI / notebooks / API

👉 **Le domaine ne dépend de rien.**

---

## 📂 Structure du projet

```text
fertilizer_recommender/
│
├── data/
│   ├── raw/            # Données Kaggle originales (IMMUTABLES)
│   ├── processed/      # Datasets internes (train/test labellisés)
│   └── submissions/    # Fichiers submission Kaggle
│
├── artifacts/
│   ├── models/         # Modèles entraînés
│   ├── reports/        # Rapports d’évaluation
│   └── plots/          # Visualisations
│
├── configs/             # Configurations YAML (reproductibilité)
│
├── src/
│   └── fertilizer_recommender/
│       ├── domain/         # Cœur métier pur
│       ├── application/    # Use cases (orchestration)
│       ├── infrastructure/ # Implémentations concrètes
│       ├── presentation/   # Interfaces utilisateur
│       └── composition_root.py
│
├── tests/               # Tests unitaires et E2E
├── pyproject.toml
└── README.md
````