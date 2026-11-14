"""Agent Text-to-SQL : génération, exécution et correction sur erreur.

Ce module contient la logique pure (sans réseau ni LangGraph) : le graphe ne fera que l'assembler.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.text2sql.config import DB_PATH
from src.text2sql.db import ResultatSQL, execute_sql
from src.text2sql.prompts import CORRECTION_TEMPLATE, build_messages
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
