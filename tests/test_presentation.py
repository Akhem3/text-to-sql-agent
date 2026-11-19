"""Tests de la logique d'affichage de l'interface (sans Gradio ni serveur).

Lancer depuis la racine du repo :
    uv run python -m pytest tests/test_presentation.py
"""

import pytest

from src.serving.agent import Tentative, construire_graphe
from src.serving.presentation import resumer, suivre_agent, tentatives_en_markdown
from src.text2sql.db import ResultatSQL
from tests.fakes import FauxBackend

SCHEMA = "CREATE TABLE clients (client_id INTEGER PRIMARY KEY, region TEXT);"
QUESTION = "Combien de clients en Corse ?"
FAUX_SQL = "SELECT nom FROM clients;"
BON_SQL = "SELECT COUNT(*) FROM clients;"


def _ratee(sql: str = FAUX_SQL, erreur: str = "no such column: nom") -> Tentative:
    return Tentative(sql, ResultatSQL(None, erreur=erreur))


def _reussie(lignes: list[tuple], tronque: bool = False) -> Tentative:
    return Tentative(BON_SQL, ResultatSQL(lignes, colonnes=["n"], tronque=tronque))


# --- suivre_agent -------------------------------------------------------------


def _suivre(reponses, db_path, **options):
    backend = FauxBackend(reponses)
    graphe = construire_graphe(backend, db_path, **options)
    return backend, suivre_agent(graphe, QUESTION, SCHEMA)


def test_suivre_succes_du_premier_coup(db_path):
    _, etapes = _suivre([BON_SQL], db_path)
    etapes = list(etapes)
    assert len(etapes) == 1
    assert [t.sql for t in etapes[0]] == [BON_SQL]


def test_suivre_une_etape_par_tentative(db_path):
    _, etapes = _suivre([FAUX_SQL, BON_SQL], db_path)
    etapes = list(etapes)
    # L'historique grandit d'une tentative à chaque étape : c'est ce que l'écran redessine.
    assert [len(e) for e in etapes] == [1, 2]
    assert etapes[0][0].resultat.erreur is not None
    assert etapes[1][1].resultat.lignes == [(3,)]


def test_suivre_ne_renvoie_jamais_l_etat_de_depart(db_path):
    # En mode "values", LangGraph émet d'abord l'état initial (liste vide) : l'écran afficherait « 0 tentative ».
    _, etapes = _suivre([FAUX_SQL] * 3, db_path, max_tentatives=3)
    etapes = list(etapes)
    assert len(etapes) == 3
    assert all(etapes)


def test_suivre_rend_la_main_avant_la_correction(db_path):
    # Tout l'intérêt du générateur : la tentative 1 est disponible AVANT le 2e appel au modèle.
    # Avec graphe.invoke, l'écran resterait figé jusqu'à la fin de la boucle.
    backend, etapes = _suivre([FAUX_SQL, BON_SQL], db_path)
    premiere = next(etapes)
    assert len(premiere) == 1
    assert len(backend.recus) == 1


# --- tentatives_en_markdown -----------------------------------------------------


def test_markdown_montre_le_sql_et_l_erreur():
    md = tentatives_en_markdown([_ratee(), _reussie([(3,)])])
    assert "**Tentative 1**" in md and "**Tentative 2**" in md
    assert f"```sql\n{FAUX_SQL}\n```" in md
    assert f"```sql\n{BON_SQL}\n```" in md
    assert "❌ Erreur : `no such column: nom`" in md
    # La tentative retenue est marquée : sans cela, une correction réussie se lisait comme un échec.
    assert md.endswith("✅ Exécutée sans erreur")


def test_markdown_pas_d_erreur_sur_une_reussite():
    md = tentatives_en_markdown([_reussie([(3,)])])
    assert "❌" not in md
    assert "✅ Exécutée sans erreur" in md


# --- resumer --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tentatives", "attendu"),
    [
        ([_reussie([(3,)])], "Exécutée du premier coup · 1 ligne"),
        ([_reussie([(1,), (2,)])], "Exécutée du premier coup · 2 lignes"),
        (
            [_reussie([])],
            "Exécutée du premier coup · aucune ligne : la question sort peut-être du périmètre de la base",
        ),
        ([_ratee(), _reussie([(3,)])], "Exécutée après correction (tentative 2) · 1 ligne"),
        ([_reussie([(1,), (2,)], tronque=True)], "Exécutée du premier coup · 2 premières lignes (résultat tronqué)"),
        ([_ratee()], "Échec après 1 tentative"),
        ([_ratee()] * 3, "Échec après 3 tentatives"),
    ],
)
def test_resumer(tentatives, attendu):
    assert resumer(tentatives) == attendu
