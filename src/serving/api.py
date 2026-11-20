"""API REST de l'agent : POST /ask reçoit une question, renvoie le SQL, les tentatives et le résultat en JSON.

Pour les programmes (scripts, dashboards, autres services) ; l'écran Gradio est fait pour les humains.
Les deux s'appuient sur le même graphe (agent.py) et sont servis ensemble par `main` :
    uv run python -m src.serving.api
puis http://127.0.0.1:8000/docs (API, testable en direct) et http://127.0.0.1:8000/ui (interface).
"""

import time
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

from src.serving.agent import construire_graphe
from src.text2sql.config import DB_PATH


class Question(BaseModel):
    # Espaces retirés AVANT le contrôle de longueur : « "   " » est refusée (422), comme une chaîne vide.
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TentativeAPI(BaseModel):
    sql: str
    erreur: str | None


class Reponse(BaseModel):
    # « echec » est une réponse normale (code 200) : l'agent n'a pas produit de SQL exécutable.
    statut: Literal["executee", "echec"]
    # Le SQL de la dernière tentative : celui qui a donné le résultat, ou le dernier essai raté.
    sql: str
    tentatives: list[TentativeAPI]
    colonnes: list[str]
    lignes: list[list[Any]]
    tronque: bool
    duree_s: float


def creer_app(
    backend, schema: str, db_path: Path = DB_PATH, max_tentatives: int = 3, max_lignes: int | None = 100
) -> FastAPI:
    """Application FastAPI prête à servir. Le backend est fourni par l'appelant : FauxBackend en test,
    LlamaServerBackend en production (injection de dépendance, comme construire_graphe)."""
    # Une seule fois, au démarrage : le graphe compilé est réutilisé par toutes les requêtes.
    graphe = construire_graphe(backend, db_path, max_tentatives=max_tentatives, max_lignes=max_lignes)
    app = FastAPI(
        title="Agent Text-to-SQL",
        description="Pose une question en français sur la base (clients, abonnements, factures).",
    )

    # `def` et non `async def` : l'agent fait des appels bloquants (httpx synchrone, sqlite). FastAPI exécute
    # donc ask dans un thread à part, et la boucle d'événements continue de servir les autres requêtes.
    @app.post("/ask")
    def ask(entree: Question) -> Reponse:
        debut = time.perf_counter()
        try:
            etat = graphe.invoke({"question": entree.question, "schema": schema, "tentatives": []})
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=503, detail="llama-server est injoignable ou en erreur : réessayez plus tard."
            ) from e
        tentatives = etat["tentatives"]
        final = tentatives[-1].resultat
        echec = final.erreur is not None
        return Reponse(
            statut="echec" if echec else "executee",
            sql=tentatives[-1].sql,
            tentatives=[TentativeAPI(sql=t.sql, erreur=t.resultat.erreur) for t in tentatives],
            colonnes=[] if echec else final.colonnes,
            lignes=[] if echec else final.lignes,
            tronque=final.tronque,
            duree_s=round(time.perf_counter() - debut, 2),
        )

    return app


def main() -> None:
    import argparse
    import sys

    import uvicorn

    from src.text2sql.config import LLAMA_SERVER_URL
    from src.text2sql.db import get_categorical_values, get_schema
    from src.text2sql.llm import LlamaServerBackend
    from src.text2sql.prompts import build_schema_context

    parser = argparse.ArgumentParser(
        description="API REST et interface de l'agent Text-to-SQL (llama-server doit tourner)."
    )
    parser.add_argument("--url", default=LLAMA_SERVER_URL, help="adresse de llama-server")
    parser.add_argument("--max-tentatives", type=int, default=3)
    parser.add_argument("--max-lignes", type=int, default=100, help="nombre maximal de lignes renvoyées")
    # 127.0.0.1 : accessible depuis ce Mac seulement ; 0.0.0.0 sera nécessaire dans un conteneur.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sans-interface", action="store_true", help="API seule, sans Gradio sur /ui")
    args = parser.parse_args()

    # Même contexte que evaluate.py et la CLI : le premier prompt est celui qui a été mesuré.
    schema = build_schema_context(get_schema(), get_categorical_values())
    try:
        backend = LlamaServerBackend(args.url)
    except httpx.ConnectError:
        sys.exit(f"llama-server injoignable sur {args.url} : lancez-le avant l'API.")

    options = {"max_tentatives": args.max_tentatives, "max_lignes": args.max_lignes}
    app = creer_app(backend, schema, **options)
    if not args.sans_interface:
        # Import ici : l'API seule (--sans-interface) n'a pas besoin du groupe ui.
        import gradio as gr

        from src.serving.interface import construire_interface

        # Un seul serveur, un seul port : /ask et /docs pour les programmes, /ui pour les humains.
        demo = construire_interface(backend, schema, **options)
        app = gr.mount_gradio_app(app, demo, path="/ui", server_name=args.host, server_port=args.port)

    # uvicorn est le serveur HTTP : il écoute le port et transmet chaque requête à l'application FastAPI.
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
