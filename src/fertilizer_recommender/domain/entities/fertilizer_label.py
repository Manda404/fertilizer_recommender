"""
fertilizer_label.py

Pourquoi ce fichier existe ?
- Représenter un label de fertilisant comme une entité métier.
- Éviter l'utilisation brute de chaînes de caractères dans le domaine.

À quoi ça sert réellement ?
- Clarifier ce qu’est une “classe” dans le problème.
- Préparer des règles futures (rareté, regroupement, priorisation).

Très utile ?
Pas critique aujourd’hui, mais IMPORTANT pour un projet propre et évolutif.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class FertilizerLabel:
    """
    Entité métier représentant un fertilisant.
    """
    name: str

    def __str__(self) -> str:
        return self.name