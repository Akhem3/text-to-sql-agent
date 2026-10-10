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