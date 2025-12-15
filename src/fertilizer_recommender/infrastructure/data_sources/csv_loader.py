"""
csv_loader.py

Pourquoi ce fichier existe ?
- Isoler pandas dans l’infrastructure.
- Avoir un point unique pour lire des CSV.

À quoi ça sert réellement ?
- Charger train.csv / test.csv depuis le disque.
- Faciliter les évolutions futures (compression, encoding, etc.).

Est-ce critique ?
Important mais remplaçable.
"""

from __future__ import annotations
from pathlib import Path
from pandas import DataFrame, read_csv


def load_csv(path: Path) -> DataFrame:
    """
    Charge un CSV avec pandas.

    Args:
        path: chemin du fichier CSV

    Returns:
        pd.DataFrame
    """
    return read_csv(path)
