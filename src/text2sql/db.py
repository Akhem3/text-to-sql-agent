"""Accès SQLite en lecture seule."""

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from .config import DB_PATH


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Connexion en lecture seule : toute écriture est refusée par SQLite lui-même."""
    uri = Path(db_path).resolve().as_uri()
    return sqlite3.connect(f"{uri}?mode=ro", uri=True)


def get_schema(db_path: Path = DB_PATH) -> str:
    """Instructions CREATE TABLE de la base, sans les tables internes de SQLite."""
    query = "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    with closing(connect(db_path)) as conn:
        return "\n".join(row[0] for row in conn.execute(query))


def get_categorical_values(db_path: Path = DB_PATH, max_distinct: int = 40) -> dict[str, list]:
    """Valeurs distinctes des colonnes catégorielles : {"table.colonne": [valeurs triées]}.

    Catégorielle = type TEXT ou BOOLEAN et au plus `max_distinct` valeurs distinctes (hors NULL).
    Les valeurs sont triées pour que le prompt soit identique d'un run à l'autre.
    """
    tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    colonnes_query = "SELECT name FROM pragma_table_info(?) WHERE type IN ('TEXT', 'BOOLEAN')"
    valeurs = {}
    with closing(connect(db_path)) as conn:
        tables = [row[0] for row in conn.execute(tables_query)]
        for table in tables:
            colonnes = [row[0] for row in conn.execute(colonnes_query, (table,))]
            for col in colonnes:
                # Identifiants (et non valeurs) : f-string obligatoire, entre guillemets doubles.
                n = conn.execute(f'SELECT COUNT(DISTINCT "{col}") FROM "{table}"').fetchone()[0]
                if n <= max_distinct:
                    query = f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL ORDER BY "{col}"'
                    valeurs[f"{table}.{col}"] = [row[0] for row in conn.execute(query)]
    return valeurs


@dataclass(frozen=True)
class ResultatSQL:
    """Résultat d'exécution : `lignes` vaut None si et seulement si `erreur` est renseignée."""

    lignes: list[tuple] | None
    colonnes: list[str] = field(default_factory=list)
    erreur: str | None = None
    tronque: bool = False


def execute_sql(
    sql: str, db_path: Path = DB_PATH, timeout_s: float = 10.0, max_lignes: int | None = None
) -> ResultatSQL:
    """Exécute une requête en gardant ce dont l'agent a besoin : noms de colonnes et message d'erreur.

    `max_lignes` borne ce qui est lu : un SELECT sans filtre sur les factures ne doit pas remonter
    des dizaines de milliers de lignes jusqu'à l'interface. None = tout lire (cas de l'évaluation).
    """
    conn = connect(db_path)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 100_000)
    try:
        curseur = conn.execute(sql)
        colonnes = [d[0] for d in curseur.description or []]
        if max_lignes is None:
            return ResultatSQL(curseur.fetchall(), colonnes)
        lignes = curseur.fetchmany(max_lignes + 1)
        return ResultatSQL(lignes[:max_lignes], colonnes, tronque=len(lignes) > max_lignes)
    except (sqlite3.Error, sqlite3.Warning) as e:
        if time.monotonic() > deadline:
            # Le progress handler interrompt la requête et SQLite ne dit que « interrupted ».
            return ResultatSQL(None, erreur=f"délai dépassé ({timeout_s:g} s) : requête trop lente")
        return ResultatSQL(None, erreur=str(e))
    finally:
        conn.close()


def run_sql(sql: str, db_path: Path = DB_PATH, timeout_s: float = 10.0):
    """Exécute une requête.

    Renvoie la liste des lignes, ou None en cas d'erreur SQL, d'écriture refusée
    ou de dépassement du délai (requête qui boucle ou produit un produit cartésien).
    """
    return execute_sql(sql, db_path, timeout_s).lignes
