"""Tests des shots écrits à la main pour l'ablation sur le style des colonnes (plan 2×2).

Pour un même jeu de questions, les versions « minimal » et « riche » ne doivent différer
que par la liste du SELECT. Sinon, l'écart de score entre deux runs ne pourrait plus être
attribué au seul style des colonnes (facteur confondant).

Lancer depuis la racine du repo :
    uv run python -m pytest

Les tests sont ignorés si la base est absente ou si les shots ne sont pas encore écrits.
"""

import re

import pytest

from src.text2sql.config import DB_PATH, SHOTS_DIR
from src.text2sql.db import run_sql
from src.text2sql.evaluate import ordre_impose, results_match
from src.text2sql.prompts import load_shots

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="base absente (dossier data/ non versionné ?)")

JEUX = ["s1", "s2"]


def _paire(jeu: str):
    minimal, riche = (SHOTS_DIR / f"{jeu}_{style}.json" for style in ("minimal", "riche"))
    if not (minimal.exists() and riche.exists()):
        pytest.skip(f"shots {jeu} pas encore écrits dans {SHOTS_DIR}")
    return load_shots(minimal), load_shots(riche)


def _corps(sql: str) -> str:
    """Tout ce qui suit le premier FROM, espaces normalisés : la partie qui ne doit pas bouger.

    Suppose qu'aucune sous-requête n'est dans le SELECT ; si c'est le cas, le premier FROM est
    le sien, les corps diffèrent et le test échoue : faux positif bruyant plutôt que silencieux.
    """
    return " ".join(re.split(r"\bFROM\b", sql, maxsplit=1, flags=re.IGNORECASE)[1].split())


@pytest.mark.parametrize("jeu", JEUX)
def test_memes_questions_meme_ordre(jeu):
    minimal, riche = _paire(jeu)
    assert [s["question"] for s in minimal] == [s["question"] for s in riche]


@pytest.mark.parametrize("jeu", JEUX)
def test_seul_le_select_change(jeu):
    minimal, riche = _paire(jeu)
    differents = [m["question"] for m, r in zip(minimal, riche) if _corps(m["sql"]) != _corps(r["sql"])]
    assert not differents, f"corps SQL modifié (pas seulement le SELECT) : {differents}"


@pytest.mark.parametrize("jeu", JEUX)
def test_chaque_shot_renvoie_des_lignes(jeu):
    # Non vide, et pas seulement valide : deux résultats vides passent results_match sans rien prouver.
    minimal, riche = _paire(jeu)
    vides = [s["question"] + " / " + s["sql"] for s in minimal + riche if not run_sql(s["sql"])]
    assert not vides, f"shots invalides ou sans ligne : {vides}"


@pytest.mark.parametrize("jeu", JEUX)
def test_minimal_contenu_dans_riche(jeu):
    minimal, riche = _paire(jeu)
    for m, r in zip(minimal, riche):
        lignes_m, lignes_r = run_sql(m["sql"]), run_sql(r["sql"])
        # La largeur d'une ligne suffit : test_chaque_shot_renvoie_des_lignes garantit qu'il y en a une.
        assert len(lignes_r[0]) > len(lignes_m[0]), f"le riche n'ajoute aucune colonne : {m['question']}"
        # Le riche joue `pred` (lui seul a des colonnes en trop). ordre_impose(m) ou (r) revient au
        # même, puisque l'ORDER BY est dans le corps, identique d'après test_seul_le_select_change.
        assert results_match(lignes_r, lignes_m, ordered=ordre_impose(m["sql"]), colonnes_en_trop=True), (
            f"les lignes du minimal ne se retrouvent pas dans le riche : {m['question']}"
        )
