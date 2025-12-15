"""
config_loader.py

Pourquoi ce fichier existe ?
- Charger les fichiers YAML de manière standardisée.
- Éviter de relire YAML différemment dans chaque notebook ou script.
- Préparer la transition vers des dataclasses (plus robuste qu'un dict libre).

À quoi ça sert réellement ?
- Tous les points d'entrée (CLI, notebooks) lisent training.yaml via cette fonction.
- Permet de centraliser la gestion d'erreurs (fichier absent, YAML invalide).

Ce fichier est-il “très utile” ?
Oui. C'est une brique de reproductibilité et de maintenabilité.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigError(RuntimeError):
    """Erreur levée quand une config est introuvable ou invalide."""


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """
    Charge un fichier YAML et retourne un dictionnaire Python.

    Args:
        path: chemin vers le YAML (ex: configs/training.yaml)

    Returns:
        dict: contenu du YAML

    Raises:
        ConfigError: si fichier introuvable ou YAML invalide
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config introuvable: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML invalide ({path}): {e}") from e

    if data is None:
        raise ConfigError(f"Config vide: {path}")

    return data
