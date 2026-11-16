"""Tests de l'évaluation de l'agent (sans serveur : faux backend et petite base SQLite).

Lancer depuis la racine du repo :
    uv run python -m pytest tests/test_evaluate_agent.py
"""

import sqlite3

import pytest

from src.serving.evaluate_agent import evaluer_agent, marquer_repetitions, resumer
from tests.fakes import FauxBackend

SCHEMA = "CREATE TABLE clients (client_id INTEGER PRIMARY KEY, region TEXT);"
N_CLIENTS = 25  # plus que les 20 lignes affichées par la CLI : de quoi tester la troncature
COMPTE_CORSE = "SELECT COUNT(*) FROM clients WHERE region = 'Corse';"
REQUETE_EN_ERREUR = "SELECT nom FROM clients;"  # colonne inexistante


# Pas la fixture db_path de conftest.py (3 clients) : il en faut plus de 20 pour la troncature.
# Nom distinct plutôt qu'un masquage de db_path, qui voudrait dire deux choses selon le fichier.
@pytest.fixture()
def db_25_clients(tmp_path):
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE clients (client_id INTEGER PRIMARY KEY AUTOINCREMENT, region TEXT)")
    conn.executemany(
        "INSERT INTO clients (region) VALUES (?)", [("Corse" if i % 5 == 0 else "Bretagne",) for i in range(N_CLIENTS)]
    )
    conn.commit()
    conn.close()
    return path


def _exemple(sql: str, question: str = "Combien de clients en Corse ?") -> dict:
    return {"question": question, "sql": sql, "famille": "test", "difficulte": "facile"}


def _evaluer(reponses, examples, db_path, **options):
    return evaluer_agent(FauxBackend(reponses), examples, SCHEMA, db_path, **options)


# --- marquer_repetitions ----------------------------------------------------


@pytest.mark.parametrize(
    ("sql_tentatives", "attendu"),
    [
        (["SELECT a;", "SELECT a;", "SELECT b;"], [False, True, False]),
        (["SELECT a;", "SELECT b;", "SELECT a;"], [False, False, True]),
        (["SELECT a;"], [False]),
        ([], []),
        # Mêmes requêtes pour SQLite : seuls les blancs diffèrent.
        (["SELECT a\nFROM t;", "SELECT  a FROM   t;"], [False, True]),
        # Filtres différents : la casse d'un littéral change le résultat.
        (["SELECT a FROM t WHERE s = 'Payée';", "SELECT a FROM t WHERE s = 'payée';"], [False, False]),
    ],
)
def test_marquer_repetitions(sql_tentatives, attendu):
    assert marquer_repetitions(sql_tentatives) == attendu


# --- evaluer_agent ----------------------------------------------------------


def test_succes_du_premier_coup(db_25_clients):
    ligne = _evaluer([COMPTE_CORSE], [_exemple(COMPTE_CORSE)], db_25_clients).iloc[0]
    assert ligne["n_tentatives"] == 1
    assert ligne["correct_t1"] and ligne["correct"]
    assert ligne["repetitions"] == [False]


def test_correction_reussie(db_25_clients):
    ligne = _evaluer([REQUETE_EN_ERREUR, COMPTE_CORSE], [_exemple(COMPTE_CORSE)], db_25_clients).iloc[0]
    assert ligne["n_tentatives"] == 2
    assert not ligne["valide_t1"] and not ligne["correct_t1"]
    assert ligne["valide"] and ligne["correct"]
    # L'erreur de la 1re tentative est conservée, la 2e n'en a pas.
    assert "nom" in ligne["erreurs_tentatives"][0]
    assert ligne["erreurs_tentatives"][1] is None


def test_repetitions_comptees(db_25_clients):
    ligne = _evaluer([REQUETE_EN_ERREUR] * 3, [_exemple(COMPTE_CORSE)], db_25_clients, max_tentatives=3).iloc[0]
    assert ligne["n_tentatives"] == 3
    assert ligne["repetitions"] == [False, True, True]
    assert not ligne["valide"] and not ligne["correct"]


def test_resultat_long_non_tronque(db_25_clients):
    # Si l'évaluation tronquait comme la CLI (20 lignes), 20 lignes seraient comparées aux 25 de la
    # référence et la requête, pourtant juste, serait comptée fausse.
    tout = "SELECT client_id FROM clients;"
    ligne = _evaluer([tout], [_exemple(tout, question="Liste des clients")], db_25_clients).iloc[0]
    assert ligne["correct"]


def test_reference_invalide(db_25_clients):
    with pytest.raises(ValueError, match="Gold SQL invalide"):
        _evaluer([COMPTE_CORSE], [_exemple("SELECT x FROM table_inexistante;")], db_25_clients)


# --- resumer ----------------------------------------------------------------


def test_resumer_compte_relances_et_repetitions(db_25_clients, capsys):
    # Question 1 : même erreur 3 fois (2 relances, toutes deux des répétitions). Question 2 : réussie.
    reponses = [REQUETE_EN_ERREUR] * 3 + [COMPTE_CORSE]
    df = _evaluer(reponses, [_exemple(COMPTE_CORSE), _exemple(COMPTE_CORSE)], db_25_clients)
    resumer(df)
    assert "Relances : 2 | dont répétitions : 2" in capsys.readouterr().out
