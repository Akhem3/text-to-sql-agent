"""Tests de la logique pure de l'agent (sans serveur ni LangGraph).

Lancer depuis la racine du repo :
    uv run python -m pytest tests/test_agent.py
"""

import sqlite3

import pytest

from src.serving.agent import AUCUNE_REQUETE, Tentative, construire_messages, executer
from src.text2sql.db import ResultatSQL
from src.text2sql.prompts import build_messages

SCHEMA = "CREATE TABLE clients (client_id INTEGER PRIMARY KEY, region TEXT);"
QUESTION = "Combien de clients en Corse ?"


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


def _ratee(sql: str, erreur: str) -> Tentative:
    return Tentative(sql, ResultatSQL(None, erreur=erreur))


# --- executer ---------------------------------------------------------------


def test_executer_ok(db_path):
    t = executer("SELECT COUNT(*) AS n FROM clients WHERE region = 'Corse';", db_path)
    assert t.sql == "SELECT COUNT(*) AS n FROM clients WHERE region = 'Corse';"
    assert t.resultat.lignes == [(2,)]
    assert t.resultat.colonnes == ["n"]
    assert t.resultat.erreur is None


def test_executer_extrait_le_sql(db_path):
    # Même nettoyage que l'évaluation : le modèle sort parfois un bloc Markdown malgré la consigne.
    t = executer("```sql\nSELECT region FROM clients LIMIT 1;\n```", db_path)
    assert t.sql == "SELECT region FROM clients LIMIT 1;"
    assert t.resultat.erreur is None


@pytest.mark.parametrize("sortie", ["", "   \n", ";"])
def test_executer_sortie_vide_est_une_erreur(db_path, sortie):
    # Sans ce contrôle, « ; » s'exécute sans erreur et renvoie [] : une réponse vide passerait pour juste.
    t = executer(sortie, db_path)
    assert t.resultat.lignes is None
    assert t.resultat.erreur == AUCUNE_REQUETE


def test_executer_erreur_sqlite_conservee(db_path):
    t = executer("SELECT nom FROM clients;", db_path)
    assert t.resultat.lignes is None
    assert "no such column" in t.resultat.erreur


def test_executer_transmet_max_lignes(db_path):
    t = executer("SELECT * FROM clients;", db_path, max_lignes=2)
    assert len(t.resultat.lignes) == 2
    assert t.resultat.tronque is True


# --- construire_messages ----------------------------------------------------


def test_premier_appel_identique_a_l_evaluation():
    # Les 43 % mesurent exactement ce prompt, et c'est celui que le LoRA a vu à l'entraînement.
    assert construire_messages(QUESTION, SCHEMA) == build_messages(QUESTION, SCHEMA)


@pytest.mark.parametrize("n_ratees", [1, 2])
def test_correction_prolonge_le_prompt_d_evaluation(n_ratees):
    ratees = [_ratee(f"SELECT faux_{i} FROM clients;", f"no such column: faux_{i}") for i in range(n_ratees)]
    base = build_messages(QUESTION, SCHEMA)
    messages = construire_messages(QUESTION, SCHEMA, ratees)

    # Même début que le premier appel : pas de nouveau prompt système, rien de réécrit.
    assert messages[: len(base)] == base
    suite = messages[len(base) :]
    assert len(suite) == 2 * n_ratees
    for i, t in enumerate(ratees):
        assistant, user = suite[2 * i], suite[2 * i + 1]
        assert assistant == {"role": "assistant", "content": t.sql}
        assert user["role"] == "user"
        # Le modèle ne peut corriger que ce qu'on lui dit : le message SQLite doit arriver intact.
        assert t.resultat.erreur in user["content"]


def test_correction_finit_par_un_tour_user():
    # C'est le tour user qui porte l'erreur : sans lui, le modèle ne sait pas que son SQL a échoué
    # et le répète (deux tours assistant d'affilée, en plus, sont hors distribution).
    messages = construire_messages(QUESTION, SCHEMA, [_ratee("SELECT x;", "no such column: x")])
    assert messages[-1]["role"] == "user"


def test_correction_refuse_une_tentative_reussie():
    # Sinon le modèle recevrait « Erreur SQLite : None » et « corrigerait » un SQL juste, sans que rien
    # ne plante : un bug de routage dans le graphe ne se verrait qu'à la baisse du score.
    ratee = _ratee("SELECT x;", "no such column: x")
    reussie = Tentative("SELECT 1;", ResultatSQL([(1,)], ["1"]))
    with pytest.raises(ValueError, match="sans erreur"):
        construire_messages(QUESTION, SCHEMA, [ratee, reussie])
