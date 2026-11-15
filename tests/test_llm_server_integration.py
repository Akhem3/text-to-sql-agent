"""Test d'intégration : le prompt construit par llama-server est-il celui de l'entraînement ?

Le modèle a été fine-tuné et évalué sur des prompts mis en forme par le chat template de HF.
llama-server applique le template lui-même, avec son propre moteur jinja et ses propres variables.
Un écart (BOS doublé, date du jour injectée, prompt système modifié) ne provoque aucune erreur :
le modèle répond quand même, simplement un peu moins bien. Seule une comparaison exacte le détecte.

Deux comparaisons, sur de vrais prompts d'évaluation (schéma compris) :
- le texte, pour un message d'échec lisible ;
- les identifiants de tokens, qui seuls voient un BOS doublé (le texte ne le montre pas).

Cas réel trouvé par ce test : llama-server injecte la date du jour dans « Today Date: », alors que
HF garde la valeur par défaut du template (26 Jul 2024), celle vue à l'entraînement. D'où l'option
--chat-template-kwargs, à garder dans toute commande de lancement (Docker Compose compris).

Ignoré si aucun serveur ne répond. Lancer le serveur :
    ~/tools/llama.cpp/build/bin/llama-server -m models/gguf/llama31-lora-v1-iter440-Q4_K_M.gguf \
        -ngl 99 -c 4096 --port 8080 --chat-template-kwargs '{"date_string": "26 Jul 2024"}'
puis, depuis la racine du repo :
    uv run python -m pytest tests/test_llm_server_integration.py -v
URL du serveur : variable d'environnement LLAMA_SERVER_URL (défaut http://localhost:8080).
"""

import difflib

import httpx
import pytest

from src.text2sql.config import DATASET_PATH, DB_PATH, DEFAULT_MODEL, LLAMA_SERVER_URL
from src.text2sql.db import get_categorical_values, get_schema
from src.text2sql.prompts import build_messages, build_schema_context

URL = LLAMA_SERVER_URL
N_QUESTIONS = 3


@pytest.fixture(scope="module")
def client():
    client = httpx.Client(base_url=URL, timeout=httpx.Timeout(30.0, connect=2.0))
    try:
        client.get("/v1/models").raise_for_status()
    except httpx.HTTPError:
        client.close()
        pytest.skip(f"llama-server injoignable sur {URL}")
    yield client
    client.close()


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    # Le tokenizer du modèle de base, pas celui du dossier fusionné : c'est lui qui a mis en forme
    # les données d'entraînement, et le dossier fusionné est un intermédiaire voué à disparaître.
    return AutoTokenizer.from_pretrained(DEFAULT_MODEL)


def _prompts_evaluation() -> list[list[dict]]:
    from datasets import load_from_disk

    validation = load_from_disk(str(DATASET_PATH))["validation"]
    schema = build_schema_context(get_schema(), get_categorical_values())
    return [build_messages(ex["question"], schema) for ex in validation.select(range(N_QUESTIONS))]


PROMPTS = _prompts_evaluation() if DATASET_PATH.exists() and DB_PATH.exists() else []


def _diff(attendu: str, obtenu: str) -> str:
    lignes = difflib.unified_diff(
        attendu.splitlines(), obtenu.splitlines(), "HF (entraînement)", "llama-server", lineterm="", n=1
    )
    return "\n".join(lignes)


@pytest.mark.skipif(not PROMPTS, reason="dataset ou base absents")
@pytest.mark.parametrize("messages", PROMPTS, ids=[f"question_{i}" for i in range(len(PROMPTS))])
def test_texte_du_prompt_identique(client, tokenizer, messages):
    attendu = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    reponse = client.post("/apply-template", json={"messages": messages})
    reponse.raise_for_status()
    # Le serveur retire le BOS du texte du template et le rajoute à la tokenisation (common/chat.cpp) :
    # on le remet ici pour comparer à HF. Le test sur les tokens vérifie qu'il n'est bien ajouté qu'une fois.
    obtenu = tokenizer.bos_token + reponse.json()["prompt"]

    assert obtenu == attendu, "Prompt différent de celui de l'entraînement :\n" + _diff(attendu, obtenu)


@pytest.mark.skipif(not PROMPTS, reason="dataset ou base absents")
@pytest.mark.parametrize("messages", PROMPTS, ids=[f"question_{i}" for i in range(len(PROMPTS))])
def test_tokens_du_prompt_identiques(client, tokenizer, messages):
    attendu = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)["input_ids"]

    prompt = client.post("/apply-template", json={"messages": messages})
    prompt.raise_for_status()
    # add_special=True : c'est ainsi que le serveur tokenise un prompt de chat avant génération.
    reponse = client.post("/tokenize", json={"content": prompt.json()["prompt"], "add_special": True})
    reponse.raise_for_status()
    obtenu = reponse.json()["tokens"]

    assert obtenu.count(tokenizer.bos_token_id) == 1, "BOS absent ou doublé"
    assert obtenu == attendu
