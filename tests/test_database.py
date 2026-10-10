"""Tests d'intégrité de la base SQLite utilisée par l'agent Text-to-SQL.

Lancer depuis la racine du repo :
    poetry run python -m pytest

"""
import os
import sqlite3
import unicodedata
from pathlib import Path

import pytest

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "raw" / "banque_entreprise.db"
DB_PATH = Path(os.environ.get("DB_PATH", DEFAULT_DB))

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason=f"base introuvable : {DB_PATH}")

EXPECTED_COLUMNS = {
    "clients": {"client_id", "nom_entreprise", "contact_email", "segment", "region", "date_inscription"},
    "abonnements": {"abonnement_id", "client_id", "type_forfait", "prix_mensuel", "statut_actif"},
    "factures": {"facture_id", "abonnement_id", "montant_paye", "date_facturation", "statut_paiement"},
}


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)  # lecture seule
    yield c
    c.close()


def scalar(conn, sql):
    return conn.execute(sql).fetchone()[0]


def distinct(conn, table, col):
    return {r[0] for r in conn.execute(f"SELECT DISTINCT {col} FROM {table}")}


# --- Structure ---------------------------------------------------------------

@pytest.mark.parametrize("table, cols", EXPECTED_COLUMNS.items())
def test_colonnes_conformes_au_schema(conn, table, cols):
    found = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    assert found == cols


@pytest.mark.parametrize("table", EXPECTED_COLUMNS)
def test_tables_non_vides(conn, table):
    assert scalar(conn, f"SELECT COUNT(*) FROM {table}") > 0


def test_sqlite_supporte_les_fonctions_de_fenetre():
    assert sqlite3.sqlite_version_info >= (3, 25, 0)


# --- Intégrité référentielle -------------------------------------------------

def test_pas_d_abonnement_orphelin(conn):
    n = scalar(conn, """
        SELECT COUNT(*) FROM abonnements a
        LEFT JOIN clients c ON a.client_id = c.client_id
        WHERE c.client_id IS NULL""")
    assert n == 0


def test_pas_de_facture_orpheline(conn):
    n = scalar(conn, """
        SELECT COUNT(*) FROM factures f
        LEFT JOIN abonnements a ON f.abonnement_id = a.abonnement_id
        WHERE a.abonnement_id IS NULL""")
    assert n == 0


# --- Qualité des valeurs -----------------------------------------------------

@pytest.mark.parametrize("table, col", [("clients", "date_inscription"), ("factures", "date_facturation")])
def test_dates_au_format_iso(conn, table, col):
    n = scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL OR date({col}) IS NULL OR date({col}) != {col}")
    assert n == 0


def test_prix_mensuels_positifs(conn):
    assert scalar(conn, "SELECT COUNT(*) FROM abonnements WHERE prix_mensuel IS NULL OR prix_mensuel <= 0") == 0


def test_montants_non_negatifs(conn):
    assert scalar(conn, "SELECT COUNT(*) FROM factures WHERE montant_paye IS NULL OR montant_paye < 0") == 0


def test_statut_actif_est_booleen(conn):
    assert distinct(conn, "abonnements", "statut_actif") <= {0, 1}


def test_statuts_de_paiement_attendus(conn):
    found = distinct(conn, "factures", "statut_paiement")
    assert None not in found
    assert {"Payée", "Échouée", "En retard"} <= found


# Les requêtes du dataset comparent du texte avec accents ('Échouée', 'Côte d''Azur') :
# si la base est en NFD alors que le dataset est en NFC, les filtres ne matchent jamais.
@pytest.mark.parametrize("table, col", [
    ("factures", "statut_paiement"),
    ("clients", "region"),
    ("clients", "segment"),
    ("abonnements", "type_forfait"),
])
def test_texte_normalise_nfc(conn, table, col):
    bad = [v for v in distinct(conn, table, col) if v and unicodedata.normalize("NFC", v) != v]
    assert not bad
