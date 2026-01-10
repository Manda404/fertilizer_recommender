"""
local_storage.py

Pourquoi ce fichier existe ?
- Centraliser la lecture / écriture de fichiers locaux.
- Éviter les accès disque dispersés dans le code.

À quoi ça sert réellement ?
- Sauvegarde de CSV, JSON, artefacts simples.
- Point d’extension futur vers du cloud storage.

Très utile ?
Important mais non critique aujourd’hui.
"""

from __future__ import annotations
from pathlib import Path


class LocalStorage:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_text(self, content: str, filename: str) -> None:
        path = self.base_dir / filename
        path.write_text(content, encoding="utf-8")

    def load_text(self, filename: str) -> str:
        path = self.base_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return path.read_text(encoding="utf-8")