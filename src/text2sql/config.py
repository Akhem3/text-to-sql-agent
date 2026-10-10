"""Chemins et constantes du projet."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # racine du repo (ce fichier : src/text2sql/config.py)
DB_PATH = ROOT / "data" / "raw" / "banque_entreprise.db"
DATASET_PATH = ROOT / "data" / "processed" / "hf_dataset"
RESULTS_DIR = ROOT / "results"

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 512
