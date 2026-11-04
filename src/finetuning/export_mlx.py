"""Exporte le train au format chat de mlx-lm : train.jsonl (entraînement) et valid.jsonl (perte de suivi).

    uv run python -m src.finetuning.export_mlx

Le valid.jsonl de mlx-lm est tiré du train, et non du split validation : ce dernier reste réservé
au score final, sinon choisir un adaptateur sur ce split biaiserait la comparaison avec la baseline.
"""

import json
from pathlib import Path

from datasets import load_from_disk

from src.finetuning.build_dataset import SEED, split_par_groupe
from src.text2sql.config import DATASET_PATH, MLX_DATA_DIR
from src.text2sql.db import get_categorical_values, get_schema
from src.text2sql.prompts import build_schema_context, to_sft_record

N_DEV = 50


def preparer_exemples(train: list[dict], schema: str) -> tuple[list[dict], list[dict]]:
    """Sépare le train en (entraînement, dev) et renvoie les deux listes au format chat de mlx-lm."""
    train_training, train_dev = split_par_groupe(train, N_DEV, SEED)
    record_training = [to_sft_record(exemple, schema) for exemple in train_training]
    record_dev = [to_sft_record(exemple, schema) for exemple in train_dev]
    return record_training, record_dev


def ecrire_jsonl(path: Path, records: list[dict]) -> None:
    # ensure_ascii=False : garde les accents lisibles dans le fichier, utile pour le relire à la main.
    lignes = (json.dumps(record, ensure_ascii=False) for record in records)
    path.write_text("\n".join(lignes) + "\n", encoding="utf-8")


def main() -> None:
    train = list(load_from_disk(str(DATASET_PATH))["train"])
    schema = build_schema_context(get_schema(), get_categorical_values())

    entrainement, dev = preparer_exemples(train, schema)

    MLX_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ecrire_jsonl(MLX_DATA_DIR / "train.jsonl", entrainement)
    ecrire_jsonl(MLX_DATA_DIR / "valid.jsonl", dev)
    print(f"{len(entrainement)} exemples d'entraînement, {len(dev)} de dev, écrits dans {MLX_DATA_DIR}")


if __name__ == "__main__":
    main()
