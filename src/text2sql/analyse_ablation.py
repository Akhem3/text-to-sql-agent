"""Analyse de l'ablation minimal/riche : recalcule les chiffres cités dans le README.

Lancer depuis la racine du repo :
    poetry run python -m src.text2sql.analyse_ablation

Les comparaisons de PRINCIPALES et la métrique stricte ont été figées avant de voir les scores.
Tout le reste est secondaire ou exploratoire : à lire comme des pistes, pas comme des preuves.
"""

import json
import sqlite3
from math import comb

from .config import DB_PATH, RESULTS_DIR

RUNS = {
    "0shot": "hf_0shot_validation_prompt_v2",
    "s1_origine": "hf_5shot_s1_validation_prompt_v2",
    "s2_origine": "hf_5shot_s2_validation_prompt_v2",
    "s1_minimal": "hf_5shot_s1_minimal_validation_prompt_v2",
    "s1_riche": "hf_5shot_s1_riche_validation_prompt_v2",
    "s2_minimal": "hf_5shot_s2_minimal_validation_prompt_v2",
    "s2_riche": "hf_5shot_s2_riche_validation_prompt_v2",
}
PRINCIPALES = [
    ("style, questions s1", "s1_minimal", "s1_riche"),
    ("style, questions s2", "s2_minimal", "s2_riche"),
    ("questions, minimal", "s1_minimal", "s2_minimal"),
    ("questions, riche", "s1_riche", "s2_riche"),
]
REPERES = [
    ("0-shot vers s1 d'origine", "0shot", "s1_origine"),
    ("0-shot vers s2 d'origine", "0shot", "s2_origine"),
    ("s1 d'origine vers s2 d'origine", "s1_origine", "s2_origine"),
    ("s2 d'origine vers s2_riche", "s2_origine", "s2_riche"),
    ("s1 d'origine vers s1_riche", "s1_origine", "s1_riche"),
    ("0-shot vers s1_minimal", "0shot", "s1_minimal"),
    ("0-shot vers s2_minimal", "0shot", "s2_minimal"),
]


def charger(nom: str) -> list[dict]:
    lignes = (RESULTS_DIR / f"{nom}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(ligne) for ligne in lignes]


def mcnemar(a: list[dict], b: list[dict], cle: str) -> tuple[int, int, float]:
    """(gains de b sur a, pertes de b sur a, p exact bilatéral).

    Version exacte (binomiale) plutôt que chi² : l'approximation est peu fiable avec si peu de
    bascules. Seules les questions où les deux runs divergent portent de l'information.
    """
    gains = sum(not x[cle] and y[cle] for x, y in zip(a, b))
    pertes = sum(x[cle] and not y[cle] for x, y in zip(a, b))
    n = gains + pertes
    if n == 0:
        return gains, pertes, 1.0
    queue = sum(comb(n, i) for i in range(min(gains, pertes) + 1)) / 2**n
    return gains, pertes, min(1.0, 2 * queue)


def holm(p_valeurs: list[float]) -> list[float]:
    """Correction de Holm : tester quatre hypothèses augmente le risque d'un faux positif."""
    ordre = sorted(range(len(p_valeurs)), key=lambda i: p_valeurs[i])
    ajustees, maximum = [0.0] * len(p_valeurs), 0.0
    for rang, i in enumerate(ordre):
        maximum = max(maximum, (len(p_valeurs) - rang) * p_valeurs[i])
        ajustees[i] = min(1.0, maximum)
    return ajustees


def largeur(con: sqlite3.Connection, sql: str) -> int | None:
    # LIMIT 0 : le nombre de colonnes sans exécuter le corps, car une requête prédite peut être très lente.
    try:
        return len(con.execute(f"SELECT * FROM ({sql.strip().rstrip(';')}) LIMIT 0").description)
    except (sqlite3.Error, AttributeError):
        return None


def afficher_largeurs(data: dict[str, list[dict]]) -> None:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    for nom, run in data.items():
        paires = [(largeur(con, e["pred_sql"]), largeur(con, e["gold_sql"]), e["correct"]) for e in run]
        paires = [p for p in paires if p[0] is not None and p[1] is not None]
        moins = sum(n < g for n, g, _ in paires)
        egal = [c for n, g, c in paires if n == g]
        plus = sum(n > g for n, g, _ in paires)
        moyenne = sum(n for n, _, _ in paires) / len(paires)
        ecart = sum(abs(n - g) for n, g, _ in paires) / len(paires)
        print(
            f"{nom:11} moy={moyenne:.2f} écart_abs_gold={ecart:.2f} moins={moins:3} égal={len(egal):3} "
            f"plus={plus:3} | justes parmi égal={sum(egal)}/{len(egal)} ({sum(egal) / len(egal):.0%})"
        )


def main() -> None:
    data = {nom: charger(fichier) for nom, fichier in RUNS.items()}
    questions = [e["question"] for e in data["0shot"]]
    # McNemar est apparié : un décalage entre les fichiers fausserait tout sans lever d'erreur.
    assert all([e["question"] for e in run] == questions for run in data.values()), "questions non alignées"

    print("=== Scores (sur 100) ===")
    for nom, run in data.items():
        print(
            f"{nom:11} valide={sum(e['valide'] for e in run):3} EX={sum(e['correct'] for e in run):3} "
            f"relâchée={sum(e['correct_relache'] for e in run):3}"
        )

    for cle, titre in (("correct", "EX stricte (principale, figée)"), ("correct_relache", "EX relâchée (secondaire)")):
        print(f"\n=== {titre} ===")
        resultats = [mcnemar(data[a], data[b], cle) for _, a, b in PRINCIPALES]
        ajustees = holm([p for _, _, p in resultats])
        for (nom, a, b), (gains, pertes, p), p_holm in zip(PRINCIPALES, resultats, ajustees):
            print(f"{nom:20} {a} vers {b} : +{gains}/-{pertes} p={p:.4f} Holm={p_holm:.4f}")

    print("\n=== Repères (exploratoire, EX stricte, sans correction) ===")
    for nom, a, b in REPERES:
        gains, pertes, p = mcnemar(data[a], data[b], "correct")
        print(f"{nom:31} +{gains}/-{pertes} p={p:.4f}")

    print("\n=== Nombre de colonnes prédit vs gold (exploratoire, requêtes compilables) ===")
    afficher_largeurs(data)


if __name__ == "__main__":
    main()
