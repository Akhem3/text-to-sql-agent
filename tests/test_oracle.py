"""Tests de validation de l'évaluation du modèle Text-to-SQL.

Vérifie que la fonction evaluate mesure correctement la validité et
l'exactitude des requêtes SQL à l'aide d'un faux modèle qui renvoie
directement les requêtes SQL de référence (gold).

Les tests couvrent les splits validation et test du dataset.

Lancer depuis la racine du repo :
    poetry run python -m pytest

Les tests sont ignorés si la base de données ou le dataset sont absents.
"""

import pytest
from datasets import load_from_disk

from src.text2sql.config import DATASET_PATH, DB_PATH
from src.text2sql.db import get_schema
from src.text2sql.evaluate import evaluate

pytestmark = pytest.mark.skipif(
    not (DATASET_PATH.exists() and DB_PATH.exists()),
    reason="base ou dataset absents (dossier data/ non versionné ?)",
)


class Oracle:
    """Faux modèle qui renvoie le gold : il valide l'outil de mesure, pas le modèle."""

    def __init__(self, examples):
        self.sqls = iter(e["sql"] for e in examples)

    def generate(self, messages):
        return next(self.sqls)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_oracle_scores_100_percent(split):
    ds = load_from_disk(str(DATASET_PATH))
    df = evaluate(Oracle(ds[split]), ds[split], get_schema())
    assert df["valide"].all()
    assert df["correct"].all(), df[~df["correct"]]["question"].tolist()