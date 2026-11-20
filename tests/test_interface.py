"""Tests de l'interface : `repondre` sans Gradio (tourne en CI), puis construction de l'écran (groupe ui).

Lancer depuis la racine du repo :
    uv run python -m pytest tests/test_interface.py
"""

import pytest

from src.serving.agent import construire_graphe
from src.serving.interface import TABLEAU_VIDE, construire_interface, repondre
from tests.fakes import FauxBackend

SCHEMA = "CREATE TABLE clients (client_id INTEGER PRIMARY KEY, region TEXT);"
QUESTION = "Combien de clients en Corse ?"
FAUX_SQL = "SELECT nom FROM clients;"
BON_SQL = "SELECT region, COUNT(*) AS n FROM clients GROUP BY region ORDER BY region;"


def _repondre(reponses, db_path, question=QUESTION, max_tentatives=3):
    backend = FauxBackend(reponses)
    graphe = construire_graphe(backend, db_path, max_tentatives=max_tentatives)
    return backend, list(repondre(graphe, SCHEMA, question, max_tentatives))


def test_succes_du_premier_coup(db_path):
    _, sorties = _repondre([BON_SQL], db_path)
    assert len(sorties) == 1
    markdown, tableau, statut = sorties[0]
    assert BON_SQL in markdown
    assert tableau == {"headers": ["region", "n"], "data": [("Bretagne", 1), ("Corse", 2)]}
    assert statut.startswith("Exécutée du premier coup · 2 lignes · ")
    assert statut.endswith(" s")


def test_correction_annoncee_puis_resultat(db_path):
    _, sorties = _repondre([FAUX_SQL, BON_SQL], db_path)
    assert len(sorties) == 2
    # Étape intermédiaire : l'erreur est visible, le tableau reste vide, la correction est annoncée.
    markdown, tableau, statut = sorties[0]
    assert "no such column" in markdown
    assert tableau == TABLEAU_VIDE
    assert statut == "Tentative 1 échouée, correction en cours…"
    assert sorties[1][2].startswith("Exécutée après correction (tentative 2) · 2 lignes")


def test_echec_n_annonce_pas_de_correction_a_la_derniere_tentative(db_path):
    _, sorties = _repondre([FAUX_SQL] * 3, db_path, max_tentatives=3)
    statuts = [statut for _, _, statut in sorties]
    assert statuts[:2] == [
        "Tentative 1 échouée, correction en cours…",
        "Tentative 2 échouée, correction en cours…",
    ]
    assert statuts[2].startswith("Échec après 3 tentatives")
    assert sorties[2][1] == TABLEAU_VIDE


@pytest.mark.parametrize("question", ["", "   \n"])
def test_question_vide_n_appelle_pas_le_modele(db_path, question):
    backend, sorties = _repondre([], db_path, question=question)
    assert sorties == [("", TABLEAU_VIDE, "Pose une question.")]
    assert backend.recus == []


def test_l_ecran_se_construit(db_path):
    gr = pytest.importorskip("gradio")
    demo = construire_interface(FauxBackend([]), SCHEMA, db_path)
    assert isinstance(demo, gr.Blocks)
