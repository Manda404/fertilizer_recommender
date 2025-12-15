
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

---

## 🔄 Cycle de vie des données (très important)

Le **test.csv Kaggle n’est jamais utilisé pendant l’entraînement**.

```text
data/raw/train.csv   (Kaggle original)
        │
        ▼
PrepareDatasetUseCase
        │
        ├── data/processed/train.csv   ← entraînement (avec cible)
        └── data/processed/test.csv    ← validation interne (avec cible)
```

### Pourquoi cette approche ?

* éviter toute fuite de données
* permettre le calcul réel de la métrique MAP@3
* figer les datasets comme **artefacts ML**
* rendre le pipeline auditable et reproductible

---

## ⚙️ Use cases principaux

### 1️⃣ PrepareDatasetUseCase

* Charge `data/raw/train.csv`
* Effectue un split **stratifié**
* Sauvegarde deux CSV labellisés :

  * `processed/train.csv`
  * `processed/test.csv`

👉 Aucun `X/y` manipulé ici :
le split se fait **au niveau DataFrame complet**.

---

### 2️⃣ TrainModelUseCase

* Charge `processed/train.csv`
* Applique le preprocessing
* Entraîne un modèle multiclasses
* Calcule la métrique métier **MAP@3**
* Sauvegarde le modèle entraîné

👉 Le modèle et le preprocessing sont **remplaçables sans modifier le use case**.

---

### 3️⃣ EvaluateModelUseCase *(à venir)*

* Évalue le modèle sur `processed/test.csv`
* Produit MAP@3 + rapports
* Génère des artefacts d’analyse

---

### 4️⃣ BuildSubmissionUseCase *(à venir)*

* Charge le modèle entraîné
* Prédit le **Top-3** d’engrais
* Génère `submission.csv` au format Kaggle

---

## 📐 Métrique métier – MAP@3

La métrique officielle Kaggle est implémentée :

* en **Python pur**
* dans le **domain**
* sans dépendance externe

👉 Cela garantit :

* la testabilité
* la transparence
* l’alignement exact avec Kaggle

---

## 🧪 Tests

```text
tests/
├── domain/        # règles métier
├── application/   # use cases
├── infrastructure/
└── e2e/           # pipeline complet
```

Les tests couvrent :

* la validité métier
* la stabilité des use cases
* l’exécution complète du pipeline ML

---

## 🔁 Reproductibilité & MLOps

* Configurations YAML (seed, paths, modèles)
* Datasets figés (`processed/`)
* Artefacts versionnés (`artifacts/`)
* Architecture extensible vers :

  * MLflow
  * Model Registry
  * CI/CD

---

## 🚀 Lancer le projet (local)

```bash
# Préparer les datasets
python -m fertilizer_recommender.presentation.cli.prepare_dataset

# Entraîner le modèle
python -m fertilizer_recommender.presentation.cli.train

# Générer une submission Kaggle
python -m fertilizer_recommender.presentation.cli.submit
```

---

## 🧩 Pourquoi ce projet est différent

✔️ Clean Architecture réelle (pas cosmétique)
✔️ Séparation stricte des responsabilités
✔️ Pipeline ML explicable et auditable
✔️ Approche proche des standards entreprise
✔️ Conçu pour être maintenu, testé et étendu

---

## 👤 Auteur

Projet développé dans une optique **Data Science / MLOps professionnelle**,
avec une attention particulière portée à :

* la qualité du design
* la reproductibilité
* la lisibilité long terme

---

> *“Un bon modèle fait un score.
> Un bon système ML survit dans le temps.”*

```

---

## Ce README est :

- cohérent avec **tout ce que tu as construit**
- défendable en entretien
- lisible par un non-Kaggleur
- parfaitement aligné Clean Architecture

👉 **Étape suivante** possible :
- README technique par couche (`domain/README.md`)
- Diagramme d’architecture
- Ajout MLflow dans le README
- Section “Design Decisions” (très senior)

Dis-moi ce que tu veux enrichir ensuite 👌
```
