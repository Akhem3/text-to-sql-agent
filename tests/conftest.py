"""Fixtures partagées : pytest charge ce fichier tout seul, ses fixtures sont visibles dans tous les tests.

Les classes utilitaires (faux backend) sont dans fakes.py : on importe un module, pas conftest.
"""

import sqlite3

import pytest


@pytest.fixture()
def db_path(tmp_path):
    """Petite base : 3 clients, dont 2 en Corse. tmp_path en fait une base neuve à chaque test."""
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE clients (client_id INTEGER PRIMARY KEY AUTOINCREMENT, region TEXT);
        INSERT INTO clients (region) VALUES ('Bretagne'), ('Corse'), ('Corse');
        """
    )
    conn.commit()
    conn.close()
    return path
