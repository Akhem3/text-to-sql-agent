import sqlite3

import pytest

from src.text2sql.db import get_schema, run_sql
from src.text2sql.evaluate import evaluate, results_match
from src.text2sql.prompts import build_messages, pick_shots, to_sft_record
from src.text2sql.sql_utils import extract_sql


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE clients (client_id INTEGER PRIMARY KEY AUTOINCREMENT, region TEXT);
        INSERT INTO clients (region) VALUES ('Bretagne'), ('Corse'), ('Corse');
        """
    )
    conn.commit()
    conn.close()
    return path


EXAMPLES = [
    {"question": "Combien de clients ?", "sql": "SELECT COUNT(*) FROM clients;",
     "famille": "comptage", "difficulte": "facile"},
    {"question": "Quelles régions ?", "sql": "SELECT DISTINCT region FROM clients ORDER BY region;",
     "famille": "distinct", "difficulte": "facile"},
]


class FakeBackend:
    def __init__(self, answers):
        self.answers = list(answers)

    def generate(self, messages):
        return self.answers.pop(0)


def test_extract_sql():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1;"
    assert extract_sql("Voici la requête : SELECT 1; et voilà") == "SELECT 1;"
    assert extract_sql("WITH t AS (SELECT 1) SELECT * FROM t") == "WITH t AS (SELECT 1) SELECT * FROM t;"


def test_schema_excludes_internal_tables(db_path):
    schema = get_schema(db_path)
    assert "CREATE TABLE clients" in schema
    assert "sqlite_sequence" not in schema


def test_run_sql_ok_error_and_readonly(db_path):
    assert run_sql("SELECT COUNT(*) FROM clients", db_path) == [(3,)]
    assert run_sql("SELECT * FROM table_inconnue", db_path) is None
    assert run_sql("DELETE FROM clients", db_path) is None
    assert run_sql("SELECT COUNT(*) FROM clients", db_path) == [(3,)]  # rien n'a été supprimé


def test_run_sql_timeout(db_path):
    slow = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT COUNT(*) FROM r"
    assert run_sql(slow, db_path, timeout_s=0.2) is None


def test_results_match():
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert results_match([(1.00001,)], [(1.0,)], ordered=True)
    assert not results_match(None, [(1,)], ordered=False)


def test_build_messages_with_shots():
    shots = [{"question": "Q1", "sql": "SELECT 1;"}]
    messages = build_messages("Q2", "CREATE TABLE t (a);", shots)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "CREATE TABLE t (a);" in messages[0]["content"]
    assert messages[-1]["content"] == "Question : Q2"


def test_sft_record_matches_inference_prompt():
    record = to_sft_record(EXAMPLES[0], "SCHEMA")
    inference = build_messages(EXAMPLES[0]["question"], "SCHEMA")
    assert record["messages"][:-1] == inference
    assert record["messages"][-1] == {"role": "assistant", "content": EXAMPLES[0]["sql"]}


def test_pick_shots_distinct_families_and_reproducible():
    train = EXAMPLES * 3
    first, second = pick_shots(train, 2, seed=1), pick_shots(train, 2, seed=1)
    assert first == second
    assert len({s["famille"] for s in first}) == 2


def test_evaluate_perfect_and_broken_backend(db_path):
    schema = get_schema(db_path)
    perfect = FakeBackend(["```sql\n" + ex["sql"] + "\n```" for ex in EXAMPLES])
    df = evaluate(perfect, EXAMPLES, schema, db_path=db_path)
    assert df["correct"].all() and df["valide"].all()

    broken = FakeBackend(["SELECT n_importe_quoi FROM nulle_part;", "SELECT COUNT(*) FROM clients;"])
    df = evaluate(broken, EXAMPLES, schema, db_path=db_path)
    assert list(df["valide"]) == [False, True]
    assert list(df["correct"]) == [False, False]
