"""Tests de l'export au format mlx-lm (train.jsonl et valid.jsonl).

Les fichiers exportés ne sont pas versionnés et ne se régénèrent pas seuls : changer le prompt ou le
dataset sans relancer l'export ferait entraîner le modèle sur des données périmées, sans aucune erreur.
Ces tests comparent donc les fichiers présents sur le disque à leurs sources actuelles.

La fuite vers validation et test n'est pas retestée ici : l'export doit contenir exactement le train,
et tests/test_dataset.py vérifie déjà que le train ne partage rien avec ces deux splits.

Lancer depuis la racine du repo :
    uv run python -m pytest

Les tests sur les fichiers sont ignorés si l'export, le dataset ou la base sont absents.
"""

import json
from collections import Counter

import pytest
from datasets import load_from_disk

from src.finetuning.build_dataset import normaliser_sql
from src.finetuning.export_mlx import preparer_exemples
from src.text2sql.config import DATASET_PATH, DB_PATH, MLX_DATA_DIR
from src.text2sql.db import get_categorical_values, get_schema
from src.text2sql.prompts import build_messages, build_schema_context

PREFIXE_QUESTION = "Question : "

necessite_export = pytest.mark.skipif(
    not (
        DATASET_PATH.exists()
        and DB_PATH.exists()
        and (MLX_DATA_DIR / "train.jsonl").exists()
        and (MLX_DATA_DIR / "valid.jsonl").exists()
    ),
    reason="export, dataset ou base absents (dossier data/ non versionné ?)",
)


def _lire(nom: str) -> list[dict]:
    with open(MLX_DATA_DIR / f"{nom}.jsonl", encoding="utf-8") as f:
        return [json.loads(ligne) for ligne in f]


def _question(record: dict) -> str:
    return record["messages"][-2]["content"].removeprefix(PREFIXE_QUESTION)


def _sql(record: dict) -> str:
    return record["messages"][-1]["content"]


def test_preparer_exemples_garde_les_paraphrases_ensemble():
    # Assez de groupes pour que N_DEV ne vide pas tout le pool vers le dev.
    train = [
        {"question": f"Question {i} variante {v}", "sql": f"SELECT {i};", "famille": f"F{i % 2}"}
        for i in range(100)
        for v in "ab"
    ]

    entrainement, dev = preparer_exemples(train, "SCHEMA")

    assert entrainement and dev
    assert Counter(_question(r) for r in entrainement + dev) == Counter(ex["question"] for ex in train)
    assert not {_sql(r) for r in entrainement} & {_sql(r) for r in dev}


@necessite_export
@pytest.mark.parametrize("nom", ["train", "valid"])
def test_prompt_identique_a_l_evaluation(nom):
    # Un écart ici ferait évaluer le modèle sur un autre prompt que celui de son entraînement.
    schema = build_schema_context(get_schema(), get_categorical_values())
    differents = [_question(r) for r in _lire(nom) if r["messages"][:-1] != build_messages(_question(r), schema)]
    assert not differents, f"{len(differents)} prompts de {nom}.jsonl diffèrent de l'évaluation : relancer l'export"


@necessite_export
def test_export_contient_exactement_le_train():
    exporte = Counter((_question(r), _sql(r)) for r in _lire("train") + _lire("valid"))
    attendu = Counter((ex["question"], ex["sql"]) for ex in load_from_disk(str(DATASET_PATH))["train"])
    assert exporte == attendu, "l'export ne correspond plus au train : relancer l'export"


@necessite_export
def test_aucun_sql_commun_entre_entrainement_et_dev():
    # Sinon la perte de dev baisserait par simple récitation et masquerait le surapprentissage.
    entrainement = {normaliser_sql(_sql(r)) for r in _lire("train")}
    communs = [_question(r) for r in _lire("valid") if normaliser_sql(_sql(r)) in entrainement]
    assert not communs, f"{len(communs)} requêtes du dev déjà présentes à l'entraînement : {communs}"
