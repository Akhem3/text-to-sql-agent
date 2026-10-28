"""Tests unitaires des composants de l'agent Text-to-SQL.

Vérifie le bon fonctionnement des utilitaires SQL, de la récupération du
schéma SQLite, de l'exécution sécurisée des requêtes et de l'évaluation
des résultats.

Teste également la construction des prompts, la préparation des exemples
pour le fine-tuning (SFT) et la sélection reproductible des exemples
few-shot.

Les tests utilisent une base SQLite temporaire et des faux modèles afin
de ne pas dépendre d'un modèle réel ni de la base de données du projet.

Lancer depuis la racine du repo :
    poetry run python -m pytest
"""

import sqlite3

import pytest

from src.text2sql.db import get_categorical_values, get_schema, run_sql
from src.text2sql.evaluate import evaluate, ordre_impose, results_match
from src.text2sql.prompts import build_messages, build_schema_context, pick_shots, to_sft_record
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


def test_get_categorical_values_seuil(db_path):
    assert get_categorical_values(db_path) == {"clients.region": ["Bretagne", "Corse"]}
    assert get_categorical_values(db_path, max_distinct=1) == {}  # 2 régions distinctes > 1


def test_get_categorical_values_types_et_null(tmp_path):
    path = tmp_path / "cat.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE t (id INTEGER PRIMARY KEY, statut TEXT, actif BOOLEAN, nom TEXT, montant REAL);
        INSERT INTO t (statut, actif, nom, montant) VALUES
            ('Payée', 1, 'A', 1.0), ('En retard', 0, 'B', 2.0), ('Payée', 1, 'C', 3.0), (NULL, 1, 'D', 4.0);
        """
    )
    conn.commit()
    conn.close()
    # id (INTEGER) et montant (REAL) : mauvais type ; nom : 4 valeurs > 2 ; NULL exclu de statut ; valeurs triées.
    assert get_categorical_values(path, max_distinct=2) == {
        "t.statut": ["En retard", "Payée"],
        "t.actif": [0, 1],
    }


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


GOLD_2COL = [("a", 1), ("b", 2)]


@pytest.mark.parametrize("pred, gold, ordered, attendu_strict, attendu_relache", [
    # Colonnes permutées : même réponse, autre ordre de SELECT.
    ([(1, "a"), (2, "b")], GOLD_2COL, False, True, True),
    ([(1, "a"), (2, "b")], GOLD_2COL, True, True, True),
    ([(2, "b"), (1, "a")], GOLD_2COL, True, False, False),  # permutation OK mais lignes dans le mauvais ordre
    # Mêmes valeurs par colonne mais lignes mal appariées : ne doit PAS passer.
    ([(2, "a"), (1, "b")], GOLD_2COL, False, False, False),
    # Colonnes en trop : refusées en strict, tolérées en relâché.
    ([(10, "a", 1), (20, "b", 2)], GOLD_2COL, False, False, True),
    # Colonne manquante : toujours faux.
    ([("a",), ("b",)], GOLD_2COL, False, False, False),
    # Colonnes en double : chaque colonne du gold doit avoir sa propre colonne dans la prédiction.
    ([(1, 1)], [(1, 1)], False, True, True),
    ([(1, 2)], [(1, 1)], False, False, False),
    ([(1, 1, 2)], [(1, 2)], False, False, True),
    # Arrondi des flottants conservé après permutation.
    ([(1.00001, "x")], [("x", 1.0)], True, True, True),
    # NULL SQL (None) mélangé à des nombres : le tri des colonnes ne doit pas planter (key=repr).
    ([(None, "a"), (1, "b")], [("b", 1), ("a", None)], False, True, True),
    # Nombre de lignes différent / résultats vides.
    ([(1, "a")], GOLD_2COL, False, False, False),
    ([], [], False, True, True),
    ([], [(1,)], False, False, False),
])
def test_results_match_colonnes(pred, gold, ordered, attendu_strict, attendu_relache):
    assert results_match(pred, gold, ordered) == attendu_strict
    assert results_match(pred, gold, ordered, colonnes_en_trop=True) == attendu_relache


@pytest.mark.parametrize("sql, attendu", [
    ("SELECT region FROM clients;", False),
    ("SELECT region FROM clients ORDER BY region;", True),
    (("WITH r AS (SELECT region, ROW_NUMBER() OVER (PARTITION BY region ORDER BY client_id) AS rn "
      "FROM clients) SELECT region FROM r WHERE rn = 1;"), False),
    ("SELECT region FROM (SELECT region FROM clients ORDER BY region LIMIT 2);", False),
    ("SELECT a FROM (SELECT b FROM (SELECT c) ORDER BY b);", False),
    ("SELECT region FROM clients WHERE client_id IN (SELECT MAX(client_id) FROM clients) ORDER BY region;", True),
])
def test_ordre_impose(sql, attendu):
    assert ordre_impose(sql) == attendu


def test_build_messages_with_shots():
    shots = [{"question": "Q1", "sql": "SELECT 1;"}]
    messages = build_messages("Q2", "CREATE TABLE t (a);", shots)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "CREATE TABLE t (a);" in messages[0]["content"]
    assert messages[-1]["content"] == "Question : Q2"


def test_build_schema_context():
    schema = "CREATE TABLE t (a);"
    assert build_schema_context(schema, {}) == schema
    context = build_schema_context(schema, {"t.statut": ["En retard", "Payée", "L'Oréal"], "t.actif": [0, 1]})
    assert context.startswith(schema)
    assert "-- t.statut : 'En retard', 'Payée', 'L''Oréal'" in context  # littéraux SQL, apostrophe échappée
    assert "-- t.actif : 0, 1" in context


def test_system_prompt_contient_les_regles():
    system = build_messages("Q", "SCHEMA")[0]["content"]
    assert "strftime" in system
    assert "SELECT *" in system


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
    assert df["correct"].all() and df["valide"].all() and df["correct_relache"].all()

    broken = FakeBackend(["SELECT n_importe_quoi FROM nulle_part;", "SELECT COUNT(*) FROM clients;"])
    df = evaluate(broken, EXAMPLES, schema, db_path=db_path)
    assert list(df["valide"]) == [False, True]
    assert list(df["correct"]) == [False, False]
    assert list(df["correct_relache"]) == [False, False]

    # Bonne réponse + une colonne en trop : fausse en strict, juste en relâché.
    extra = FakeBackend([
        "SELECT COUNT(*), 0 FROM clients;",
        "SELECT region, COUNT(*) FROM clients GROUP BY region ORDER BY region;",
    ])
    df = evaluate(extra, EXAMPLES, schema, db_path=db_path)
    assert list(df["correct"]) == [False, False]
    assert list(df["correct_relache"]) == [True, True]
