"""Tests de l'API REST : le contrat de POST /ask (codes HTTP et JSON), sans llama-server ni uvicorn.

Le TestClient appelle l'application en mémoire ; FauxBackend joue le modèle.

Lancer depuis la racine du repo :
    uv run python -m pytest tests/test_api.py
"""

import httpx
import pytest

# Import direct, sans importorskip : la CI installe le groupe api. Si elle cessait de le faire, ces tests
# doivent casser bruyamment, pas être sautés en silence (CI verte, API jamais testée).
from fastapi.testclient import TestClient

from src.serving.api import creer_app
from tests.fakes import FauxBackend

SCHEMA = "CREATE TABLE clients (client_id INTEGER PRIMARY KEY, region TEXT);"
QUESTION = "Combien de clients par région ?"
FAUX_SQL = "SELECT nom FROM clients;"
BON_SQL = "SELECT region, COUNT(*) AS n FROM clients GROUP BY region ORDER BY region;"


class BackendEnPanne:
    """llama-server éteint : le premier appel au modèle échoue comme le ferait httpx."""

    def generate(self, messages):
        raise httpx.ConnectError("Connection refused")


def _client(backend, db_path, **options) -> TestClient:
    return TestClient(creer_app(backend, SCHEMA, db_path, **options))


def test_reussi_du_premier_coup(db_path):
    reponse = _client(FauxBackend([BON_SQL]), db_path).post("/ask", json={"question": QUESTION})

    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["statut"] == "executee"
    assert corps["sql"] == BON_SQL
    assert corps["tentatives"] == [{"sql": BON_SQL, "erreur": None}]
    assert corps["colonnes"] == ["region", "n"]
    # JSON n'a pas de tuple : les lignes de SQLite arrivent en listes.
    assert corps["lignes"] == [["Bretagne", 1], ["Corse", 2]]
    assert corps["tronque"] is False
    assert corps["duree_s"] >= 0


def test_corrige_a_la_deuxieme_tentative(db_path):
    reponse = _client(FauxBackend([FAUX_SQL, BON_SQL]), db_path).post("/ask", json={"question": QUESTION})

    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["statut"] == "executee"
    # Le client voit tout l'historique, pas seulement la requête retenue.
    assert [t["sql"] for t in corps["tentatives"]] == [FAUX_SQL, BON_SQL]
    assert "no such column" in corps["tentatives"][0]["erreur"]
    assert corps["tentatives"][1]["erreur"] is None
    assert corps["sql"] == BON_SQL
    assert corps["lignes"] == [["Bretagne", 1], ["Corse", 2]]


def test_echec_de_l_agent_reste_un_200(db_path):
    # La requête était valide et le service a fait son travail : « pas trouvé » est une réponse, pas une
    # erreur HTTP. Un 5xx annoncerait une panne, un 4xx une faute du client.
    client = _client(FauxBackend([FAUX_SQL] * 3), db_path, max_tentatives=3)
    reponse = client.post("/ask", json={"question": QUESTION})

    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["statut"] == "echec"
    assert len(corps["tentatives"]) == 3
    assert all(t["erreur"] for t in corps["tentatives"])
    assert corps["sql"] == FAUX_SQL
    assert corps["colonnes"] == []
    assert corps["lignes"] == []


def test_resultat_tronque(db_path):
    reponse = _client(FauxBackend([BON_SQL]), db_path, max_lignes=1).post("/ask", json={"question": QUESTION})

    corps = reponse.json()
    assert corps["lignes"] == [["Bretagne", 1]]
    assert corps["tronque"] is True


@pytest.mark.parametrize("corps", [{"question": ""}, {"question": "   \n"}, {}], ids=["vide", "espaces", "absente"])
def test_question_invalide_donne_422_sans_appeler_le_modele(db_path, corps):
    backend = FauxBackend([])
    reponse = _client(backend, db_path).post("/ask", json=corps)

    # 422 : faute du client, détectée par Pydantic avant même d'entrer dans la fonction de l'endpoint.
    assert reponse.status_code == 422
    assert backend.recus == []


def test_llama_server_injoignable_donne_503(db_path):
    reponse = _client(BackendEnPanne(), db_path).post("/ask", json={"question": QUESTION})

    # 503 : une dépendance est indisponible, le client peut réessayer plus tard. Pas 500 (ce n'est pas
    # un bug de l'API) ni 4xx (la requête était correcte).
    assert reponse.status_code == 503
    assert "llama-server" in reponse.json()["detail"]


def test_documentation_openapi_decrit_ask(db_path):
    reponse = _client(FauxBackend([]), db_path).get("/openapi.json")

    assert reponse.status_code == 200
    assert "post" in reponse.json()["paths"]["/ask"]
