"""Tests de la logique pure de l'agent (sans serveur ni LangGraph).

Lancer depuis la racine du repo :
    uv run python -m pytest tests/test_agent.py
"""

import sqlite3

import httpx
import pytest

from src.serving.agent import (
    AUCUNE_REQUETE,
    Tentative,
    construire_graphe,
    construire_messages,
    executer,
    formater_tableau,
    router,
)
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


# --- router -----------------------------------------------------------------

OK = Tentative("SELECT 1;", ResultatSQL([(1,)], ["1"]))
VIDE = Tentative("SELECT 1 WHERE 0;", ResultatSQL([], ["1"]))
KO = _ratee("SELECT x;", "no such column: x")


@pytest.mark.parametrize(
    ("tentatives", "max_tentatives", "attendu"),
    [
        ([OK], 3, "fin"),
        # Pas de relance sur un résultat vide : « aucun client ne correspond » est souvent la bonne réponse,
        # et l'ancien agent poussait alors le modèle à inventer un filtre qui renvoie quelque chose.
        ([VIDE], 3, "fin"),
        ([KO], 3, "tenter"),
        ([KO, KO], 3, "tenter"),
        ([KO, KO, KO], 3, "fin"),  # budget épuisé : la 3e tentative ratée est la dernière
        ([KO, OK], 3, "fin"),  # corrigée à la 2e
        ([KO], 1, "fin"),  # max_tentatives=1 : pas de correction du tout (mode évaluation)
    ],
)
def test_router(tentatives, max_tentatives, attendu):
    assert router(tentatives, max_tentatives) == attendu


# --- formater_tableau -------------------------------------------------------


def test_formater_tableau_aligne_les_colonnes():
    resultat = ResultatSQL([("Bretagne", 1), ("Corse", 2)], ["region", "n"])
    assert formater_tableau(resultat) == "region   | n\n---------+--\nBretagne | 1\nCorse    | 2"


def test_formater_tableau_cellule_plus_large_que_l_en_tete():
    # La largeur vient du contenu le plus long, pas seulement de l'en-tête.
    resultat = ResultatSQL([(1, 1234.5)], ["id", "montant"])
    assert formater_tableau(resultat) == "id | montant\n---+--------\n1  | 1234.5"


def test_formater_tableau_affiche_null():
    resultat = ResultatSQL([("Corse", None)], ["region", "email"])
    assert formater_tableau(resultat) == "region | email\n-------+------\nCorse  | NULL"


# --- graphe -----------------------------------------------------------------


class FauxBackend:
    """Renvoie des réponses écrites à l'avance et garde les messages reçus, pour vérifier ce que le modèle a vu."""

    def __init__(self, reponses):
        self.reponses = list(reponses)
        self.recus = []

    def generate(self, messages):
        self.recus.append(messages)
        # IndexError si le graphe appelle le modèle plus souvent que prévu : le test échoue bruyamment.
        return self.reponses.pop(0)


def _lancer(backend, db_path, **options):
    graphe = construire_graphe(backend, db_path, **options)
    return graphe.invoke({"question": QUESTION, "schema": SCHEMA, "tentatives": []})


def test_graphe_succes_du_premier_coup(db_path):
    backend = FauxBackend(["SELECT COUNT(*) FROM clients;"])
    etat = _lancer(backend, db_path)
    assert len(etat["tentatives"]) == 1
    assert etat["tentatives"][-1].resultat.lignes == [(3,)]
    # Le seul appel est celui que l'évaluation a mesuré.
    assert backend.recus == [build_messages(QUESTION, SCHEMA)]


def test_graphe_corrige_apres_erreur(db_path):
    backend = FauxBackend(["SELECT nom FROM clients;", "SELECT COUNT(*) FROM clients;"])
    etat = _lancer(backend, db_path)
    premiere, seconde = etat["tentatives"]
    assert "no such column" in premiere.resultat.erreur
    assert seconde.resultat.lignes == [(3,)]
    # Le 2e appel contient bien le SQL fautif et l'erreur : c'est tout l'intérêt de la boucle.
    assert backend.recus[1] == construire_messages(QUESTION, SCHEMA, [premiere])


def test_graphe_s_arrete_apres_max_tentatives(db_path):
    backend = FauxBackend(["SELECT nom FROM clients;"] * 5)
    etat = _lancer(backend, db_path, max_tentatives=3)
    assert len(backend.recus) == 3
    assert len(etat["tentatives"]) == 3
    assert etat["tentatives"][-1].resultat.erreur is not None


def test_graphe_resultat_vide_sans_relance(db_path):
    backend = FauxBackend(["SELECT * FROM clients WHERE region = 'Alsace';"])
    etat = _lancer(backend, db_path)
    assert len(backend.recus) == 1
    assert etat["tentatives"][-1].resultat.lignes == []


def test_graphe_transmet_max_lignes(db_path):
    # Paramètre passé de construire_graphe jusqu'à execute_sql : s'il se perdait en route, rien ne planterait.
    backend = FauxBackend(["SELECT * FROM clients;"])
    etat = _lancer(backend, db_path, max_lignes=2)
    assert etat["tentatives"][-1].resultat.tronque is True


def test_graphe_erreur_backend_remonte(db_path):
    # Un serveur éteint n'est pas une erreur SQL : la renvoyer au modèle n'aurait aucun sens.
    class BackendEteint:
        def generate(self, messages):
            raise httpx.ConnectError("serveur injoignable")

    with pytest.raises(httpx.ConnectError):
        _lancer(BackendEteint(), db_path)
