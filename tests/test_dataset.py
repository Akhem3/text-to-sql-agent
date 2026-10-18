"""Tests d'intégrité du dataset Text-to-SQL (train / validation / test).

Vérifie qu'aucune requête n'est partagée entre deux splits (fuite de données)
et que chaque requête de référence renvoie au moins une ligne sur la base :
un gold vide accepterait comme correcte n'importe quelle prédiction vide.

Lancer depuis la racine du repo :
    poetry run python -m pytest

Les tests sont ignorés si la base de données ou le dataset sont absents.
"""

import pytest
from datasets import load_from_disk

from src.finetuning.build_dataset import normaliser_sql
from src.text2sql.config import DATASET_PATH, DB_PATH
from src.text2sql.db import run_sql

pytestmark = pytest.mark.skipif(
    not (DATASET_PATH.exists() and DB_PATH.exists()),
    reason="base ou dataset absents (dossier data/ non versionné ?)",
)

SPLITS = ["train", "validation", "test"]


@pytest.fixture(scope="module")
def ds():
    return load_from_disk(str(DATASET_PATH))


@pytest.mark.parametrize("a, b", [("train", "validation"), ("train", "test"), ("validation", "test")])
def test_aucun_sql_commun_entre_splits(ds, a, b):
    sql_a = {normaliser_sql(ex["sql"]) for ex in ds[a]}
    communs = [ex["question"] for ex in ds[b] if normaliser_sql(ex["sql"]) in sql_a]
    assert not communs, f"{len(communs)} requêtes de {b} déjà présentes dans {a} : {communs}"


@pytest.mark.parametrize("split", SPLITS)
def test_chaque_gold_renvoie_au_moins_une_ligne(ds, split):
    vides = [ex["question"] for ex in ds[split] if not run_sql(ex["sql"])]
    assert not vides, f"{len(vides)} requêtes de {split} renvoient 0 ligne : {vides}"

