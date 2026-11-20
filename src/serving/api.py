"""API REST de l'agent : POST /ask reçoit une question, renvoie le SQL, les tentatives et le résultat en JSON.

Pour les programmes (scripts, dashboards, autres services) ; l'écran Gradio est fait pour les humains.
Les deux s'appuient sur le même graphe (agent.py).
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
