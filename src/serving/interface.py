"""Interface web de l'agent : la question, les tentatives au fil de l'eau, le tableau de résultats.

Lancer depuis la racine du repo (llama-server doit tourner, avec l'option date, voir le README) :
    uv run python -m src.serving.interface

Gradio n'est importé que dans construire_interface et main : `repondre` se teste sans le groupe `ui` (CI).
"""

import time
from collections.abc import Iterator
from pathlib import Path

import httpx

from src.serving.agent import construire_graphe, router
from src.serving.presentation import resumer, suivre_agent, tentatives_en_markdown
from src.text2sql.config import DB_PATH, LLAMA_SERVER_URL

# Questions du split de VALIDATION, jamais du test. Les 4 premières étaient justes du premier coup lors
# de la mesure de l'agent ; la dernière a été corrigée à la 3e tentative (boucle de correction visible).
EXEMPLES = [
    "Combien de factures par statut de paiement pour les clients inscrits avant 2025 ?",
    "Donne les 5 clients possédant le plus d'abonnements",
    "Pourcentage du total des revenus perçus apporté par chaque région",
    "quelles entreprises ont jamais souscrit ?",
    "liste les abonnements avec plus de factures que la moyenne de factures par abonnement",
]

# Des exemples choisis parce qu'ils marchent : le vrai score doit être affiché à côté.
AVERTISSEMENT = (
    "Une requête qui s'exécute sans erreur peut quand même être fausse : "
    "exactitude mesurée de 42 % sur le jeu de test (voir le README)."
)

TABLEAU_VIDE = {"headers": [], "data": []}


def repondre(graphe, schema: str, question: str, max_tentatives: int) -> Iterator[tuple[str, dict, str]]:
    """Pour chaque tentative de l'agent : (Markdown des tentatives, tableau, phrase d'état).

    `max_tentatives` doit être celui du graphe : il sert à savoir si une correction va suivre.
    """
    question = question.strip()
    if not question:
        yield "", TABLEAU_VIDE, "Pose une question."
        return

    debut = time.perf_counter()
    for tentatives in suivre_agent(graphe, question, schema):
        markdown = tentatives_en_markdown(tentatives)
        # Même règle que le graphe : l'écran ne peut pas annoncer autre chose que ce que fait l'agent.
        if router(tentatives, max_tentatives) == "tenter":
            yield markdown, TABLEAU_VIDE, f"Tentative {len(tentatives)} échouée, correction en cours…"
            continue

        final = tentatives[-1].resultat
        tableau = TABLEAU_VIDE if final.erreur is not None else {"headers": final.colonnes, "data": final.lignes}
        duree = f"{time.perf_counter() - debut:.1f}".replace(".", ",")
        yield markdown, tableau, f"{resumer(tentatives)} · {duree} s"


def construire_interface(backend, schema: str, db_path: Path = DB_PATH, max_tentatives: int = 3, max_lignes=100):
    """L'écran complet (gr.Blocks), prêt à être lancé. Le graphe est construit ici, une seule fois."""
    import gradio as gr

    graphe = construire_graphe(backend, db_path, max_tentatives=max_tentatives, max_lignes=max_lignes)

    def _repondre(question: str):
        try:
            yield from repondre(graphe, schema, question, max_tentatives)
        except httpx.HTTPError as e:
            # Serveur éteint ou en erreur : un bandeau lisible plutôt qu'une trace Python côté serveur.
            raise gr.Error(f"llama-server injoignable ou en erreur : {e}") from e

    # analytics_enabled=False : Gradio envoie sinon des statistiques d'usage à ses serveurs.
    with gr.Blocks(title="Agent Text-to-SQL", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Agent Text-to-SQL\n"
            "Pose une question en français sur la base (clients, abonnements, factures) : l'agent écrit "
            "le SQL, l'exécute, et se corrige si SQLite renvoie une erreur."
        )
        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(label="Question", placeholder="Ex. : nombre de clients par région", lines=2)
                bouton = gr.Button("Générer le SQL", variant="primary")
                gr.Examples(EXEMPLES, inputs=[question])
                tentatives = gr.Markdown()
            with gr.Column(scale=3):
                statut = gr.Markdown()
                tableau = gr.Dataframe(label="Résultat", interactive=False)
        gr.Markdown(f"*{AVERTISSEMENT}*")

        # Bouton et touche Entrée déclenchent la même fonction ; les sorties sont remplies dans l'ordre
        # des valeurs renvoyées par chaque `yield`.
        gr.on(
            triggers=[bouton.click, question.submit],
            fn=_repondre,
            inputs=[question],
            outputs=[tentatives, tableau, statut],
            # Gradio expose aussi chaque événement comme une API (/gradio_api/...) : un nom stable plutôt que « _repondre ».
            api_name="repondre",
        )
    return demo


def main() -> None:
    import argparse
    import sys

    from src.text2sql.db import get_categorical_values, get_schema
    from src.text2sql.llm import LlamaServerBackend
    from src.text2sql.prompts import build_schema_context

    parser = argparse.ArgumentParser(description="Interface web de l'agent Text-to-SQL (llama-server doit tourner).")
    parser.add_argument("--url", default=LLAMA_SERVER_URL, help="adresse de llama-server")
    parser.add_argument("--max-tentatives", type=int, default=3)
    parser.add_argument("--max-lignes", type=int, default=100, help="nombre maximal de lignes affichées")
    # 127.0.0.1 : accessible depuis ce Mac seulement ; 0.0.0.0 sera nécessaire dans un conteneur.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # Même contexte que evaluate.py et la CLI : le premier prompt est celui qui a été mesuré.
    schema = build_schema_context(get_schema(), get_categorical_values())
    try:
        backend = LlamaServerBackend(args.url)
    except httpx.ConnectError:
        sys.exit(f"llama-server injoignable sur {args.url} : lancez-le avant l'interface.")

    demo = construire_interface(backend, schema, max_tentatives=args.max_tentatives, max_lignes=args.max_lignes)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
