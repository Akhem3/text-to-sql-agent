"""Tests unitaires du backend HTTP vers llama-server (API compatible OpenAI).

Aucun serveur n'est lancé : httpx.MockTransport remplace la couche réseau par une fonction qui
reçoit la requête et renvoie une réponse écrite à la main. On vérifie ainsi ce que le backend
envoie réellement, ce qu'un vrai serveur ne permet pas d'observer facilement.

Ce qui compte : les conditions de décodage doivent être celles des autres backends (greedy,
même plafond de tokens). Sinon, un écart de score avec le run MLX pourrait venir de
l'échantillonnage et non du GGUF, et la comparaison de l'étape 5 ne prouverait rien.

Lancer depuis la racine du repo :
    uv run python -m pytest tests/test_llm_server.py
"""

import json

import httpx
import pytest

from src.text2sql.config import MAX_NEW_TOKENS
from src.text2sql.llm import LlamaServerBackend

BASE_URL = "http://test-server:8080"
GGUF_SERVI = "llama31-lora-v1-iter440-Q4_K_M.gguf"
MESSAGES = [
    {"role": "system", "content": "Tu écris du SQL SQLite."},
    {"role": "user", "content": "Combien de clients ?"},
]


def reponse_chat(contenu: str) -> dict:
    """Corps minimal d'une réponse /v1/chat/completions au format OpenAI."""
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": contenu}, "finish_reason": "stop"}]}


def faux_serveur(contenu: str = "SELECT 1", statut_chat: int = 200):
    """Renvoie (client, requetes) : un client branché sur un faux llama-server qui journalise les requêtes reçues."""
    requetes: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requetes.append(request)
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": GGUF_SERVI, "object": "model"}]})
        if request.method == "POST" and request.url.path == "/v1/chat/completions":
            if statut_chat != 200:
                return httpx.Response(statut_chat, json={"error": {"message": "erreur simulée"}})
            return httpx.Response(200, json=reponse_chat(contenu))
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    return client, requetes


def test_init_interroge_le_serveur_et_enregistre_le_modele():
    # Le GGUF servi doit finir dans meta.json : sans cela, un run ne dit pas quel fichier a été évalué.
    client, requetes = faux_serveur()
    backend = LlamaServerBackend(BASE_URL, client=client)

    assert backend.model_id == GGUF_SERVI
    assert [(r.method, r.url.path) for r in requetes] == [("GET", "/v1/models")]


def test_corps_de_la_requete():
    client, requetes = faux_serveur()
    backend = LlamaServerBackend(BASE_URL, client=client)
    backend.generate(MESSAGES)

    chat = [r for r in requetes if r.url.path == "/v1/chat/completions"]
    assert len(chat) == 1
    corps = json.loads(chat[0].content)
    # Messages bruts : c'est le serveur qui applique le chat template, pas le client.
    assert corps["messages"] == MESSAGES
    assert corps["temperature"] == 0
    assert corps["max_tokens"] == MAX_NEW_TOKENS


def test_max_new_tokens_du_constructeur_respecte():
    # Une valeur différente du défaut : sinon, lire la constante globale au lieu de l'attribut passerait inaperçu.
    client, requetes = faux_serveur()
    backend = LlamaServerBackend(BASE_URL, max_new_tokens=64, client=client)
    backend.generate(MESSAGES)

    corps = json.loads(requetes[-1].content)
    assert corps["max_tokens"] == 64


def test_extraction_de_la_reponse():
    # Même nettoyage que HFBackend et MLXBackend (.strip()), pour que les sorties soient comparables.
    client, _ = faux_serveur(contenu="  SELECT COUNT(*) FROM clients;\n")
    backend = LlamaServerBackend(BASE_URL, client=client)

    assert backend.generate(MESSAGES) == "SELECT COUNT(*) FROM clients;"


@pytest.mark.parametrize("statut", [400, 500, 503])
def test_erreur_http_leve_une_exception(statut):
    # Une erreur serveur ne doit jamais devenir une chaîne vide : evaluate la compterait comme
    # un SQL invalide, et le score baisserait sans que personne ne voie que le serveur a planté.
    client, _ = faux_serveur(statut_chat=statut)
    backend = LlamaServerBackend(BASE_URL, client=client)

    with pytest.raises(httpx.HTTPStatusError):
        backend.generate(MESSAGES)
