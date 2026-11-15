"""Agent Text-to-SQL : génération, exécution et correction sur erreur.

Les fonctions pures (executer, construire_messages, router) portent toute la logique ;
le graphe LangGraph, en bas du fichier, ne fait que les enchaîner.
"""

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from src.text2sql.config import DB_PATH, LLAMA_SERVER_URL
from src.text2sql.db import ResultatSQL, execute_sql, get_categorical_values, get_schema
from src.text2sql.llm import Backend, LlamaServerBackend
from src.text2sql.prompts import CORRECTION_TEMPLATE, build_messages, build_schema_context
from src.text2sql.sql_utils import extract_sql

# Une sortie vide donne « ; », que SQLite exécute sans erreur (zéro ligne) : sans ce message,
# l'agent présenterait un résultat vide comme une réponse.
AUCUNE_REQUETE = "aucune requête SQL trouvée dans la réponse"


@dataclass(frozen=True)
class Tentative:
    """Une génération du modèle : le SQL extrait et ce que SQLite en a fait."""

    sql: str
    resultat: ResultatSQL


def executer(sortie_brute: str, db_path: Path = DB_PATH, max_lignes: int | None = None) -> Tentative:
    """Extrait le SQL de la sortie du modèle et l'exécute."""
    # Même extraction que evaluate.py : l'agent doit exécuter ce que l'évaluation a mesuré.
    sql = extract_sql(sortie_brute)
    # extract_sql ajoute toujours un « ; » : une sortie vide (ou sans texte) devient exactement « ; ».
    if sql == ";":
        return Tentative(sql, ResultatSQL(None, erreur=AUCUNE_REQUETE))
    return Tentative(sql, execute_sql(sql, db_path, max_lignes=max_lignes))


def construire_messages(question: str, schema: str, tentatives: Sequence[Tentative] = ()) -> list[dict]:
    """Messages envoyés au modèle : le prompt d'évaluation, puis un échange par tentative ratée."""
    messages = build_messages(question, schema)
    for tentative in tentatives:
        if tentative.resultat.erreur is None:
            raise ValueError(f"tentative sans erreur, rien à corriger : {tentative.sql}")
        messages.append({"role": "assistant", "content": tentative.sql})
        messages.append({"role": "user", "content": CORRECTION_TEMPLATE.format(erreur=tentative.resultat.erreur)})
    return messages


def router(tentatives: Sequence[Tentative], max_tentatives: int) -> Literal["tenter", "fin"]:
    """Décide après chaque tentative : en refaire une (correction) ou s'arrêter."""
    if tentatives[-1].resultat.erreur is None:
        return "fin"
    elif len(tentatives) < max_tentatives:
        return "tenter"
    else:
        return "fin"


def formater_tableau(resultat: ResultatSQL) -> str:
    """Tableau texte aligné : en-tête, séparateur, puis une ligne par résultat.

    Exemple pour colonnes ["region", "n"] et lignes [("Bretagne", 1), ("Corse", 2)] :
        region   | n
        ---------+--
        Bretagne | 1
        Corse    | 2
    """
    # L'en-tête est traité comme une ligne : il compte dans les largeurs, et zip(*...) ne reçoit
    # jamais de liste vide (zip(*[]) ne donnerait aucune colonne, donc aucune largeur).
    cellules = [resultat.colonnes] + [["NULL" if v is None else str(v) for v in ligne] for ligne in resultat.lignes]
    largeurs = [max(len(c) for c in colonne) for colonne in zip(*cellules)]

    def aligner(ligne: list[str], sep: str = " | ") -> str:
        return sep.join(c.ljust(w) for c, w in zip(ligne, largeurs)).rstrip()

    separateur = aligner(["-" * w for w in largeurs], sep="-+-")
    return "\n".join([aligner(cellules[0]), separateur, *(aligner(ligne) for ligne in cellules[1:])])


class EtatAgent(TypedDict):
    question: str
    schema: str
    # Réducteur operator.add : chaque passage dans « tenter » renvoie [sa tentative] et LangGraph
    # l'ajoute à la liste au lieu de la remplacer. L'historique complet sert à construire_messages.
    tentatives: Annotated[list[Tentative], operator.add]


def construire_graphe(
    backend: Backend, db_path: Path = DB_PATH, max_tentatives: int = 3, max_lignes: int | None = None
):
    """Graphe compilé : START → tenter → (router) → tenter | END."""
    # Import local : LangGraph n'est nécessaire qu'au moment de construire le graphe, pas pour les
    # fonctions pures ci-dessus (même principe que les backends de llm.py).
    from langgraph.graph import END, START, StateGraph

    # Les nœuds ne reçoivent que l'état : backend et options leur parviennent par fermeture (closure).
    def tenter(etat: EtatAgent) -> dict:
        messages = construire_messages(etat["question"], etat["schema"], etat["tentatives"])
        tentative = executer(backend.generate(messages), db_path, max_lignes)
        # Une liste d'un élément : le réducteur operator.add la concatène à l'historique.
        return {"tentatives": [tentative]}

    def choisir_suite(etat: EtatAgent) -> Literal["tenter", "fin"]:
        return router(etat["tentatives"], max_tentatives)

    graphe = StateGraph(EtatAgent)
    graphe.add_node("tenter", tenter)
    graphe.add_edge(START, "tenter")
    # La chaîne renvoyée par choisir_suite désigne le nœud suivant ; END termine l'exécution.
    graphe.add_conditional_edges("tenter", choisir_suite, {"tenter": "tenter", "fin": END})
    return graphe.compile()


def main() -> None:
    import argparse
    import sys

    import httpx

    parser = argparse.ArgumentParser(description="Pose une question à l'agent Text-to-SQL (llama-server doit tourner).")
    parser.add_argument("question")
    parser.add_argument(
        "--url", default=LLAMA_SERVER_URL, help="adresse de llama-server (défaut : $LLAMA_SERVER_URL, sinon localhost:8080)"
    )
    parser.add_argument("--max-tentatives", type=int, default=3)
    parser.add_argument("--max-lignes", type=int, default=20, help="nombre maximal de lignes affichées")
    args = parser.parse_args()

    # Même contexte que evaluate.py : le premier prompt doit être celui qui a été mesuré (43 % sur la validation).
    schema = build_schema_context(get_schema(), get_categorical_values())
    try:
        backend = LlamaServerBackend(args.url)
    except httpx.ConnectError:
        sys.exit(f"llama-server injoignable sur {args.url} : lancez-le avant l'agent.")

    graphe = construire_graphe(backend, max_tentatives=args.max_tentatives, max_lignes=args.max_lignes)
    etat = graphe.invoke({"question": args.question, "schema": schema, "tentatives": []})

    for i, tentative in enumerate(etat["tentatives"], start=1):
        print(f"\n[Tentative {i}]\n{tentative.sql}")
        if tentative.resultat.erreur is not None:
            print(f"Erreur : {tentative.resultat.erreur}")

    final = etat["tentatives"][-1].resultat
    if final.erreur is not None:
        # Code de sortie non nul : un script (ou un conteneur) appelant l'agent doit voir l'échec.
        sys.exit(f"\nÉchec après {len(etat['tentatives'])} tentative(s).")
    print()
    print(formater_tableau(final) if final.lignes else "(aucune ligne)")
    if final.tronque:
        print(f"… résultat tronqué aux {args.max_lignes} premières lignes")


if __name__ == "__main__":
    main()
