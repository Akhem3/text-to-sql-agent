"""Chemins et constantes du projet."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # racine du repo (ce fichier : src/text2sql/config.py)
DB_PATH = ROOT / "data" / "raw" / "banque_entreprise.db"
DATASET_PATH = ROOT / "data" / "processed" / "hf_dataset"
MLX_DATA_DIR = ROOT / "data" / "processed" / "mlx"
SHOTS_DIR = ROOT / "data" / "shots"
RESULTS_DIR = ROOT / "results"

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 512

# Une seule source pour l'adresse du serveur : en local, llama-server tourne sur la même machine ;
# dans le conteneur, Compose la remplace par host.docker.internal. L'option --url garde la priorité.
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")
