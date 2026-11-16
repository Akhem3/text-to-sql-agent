"""Évaluation de l'agent complet (boucle de correction comprise) sur un split du dataset.

evaluate.py mesure une seule génération ; ce script fait tourner le graphe déployé et mesure ce que
la boucle ajoute. La métrique est celle de evaluate.py (mêmes fonctions importées), pour que
« EX tentative 1 » et « EX agent » se comparent directement aux runs existants.

Exemple (depuis la racine du repo, llama-server lancé) :
    uv run python -m src.serving.evaluate_agent --split validation --name agent_q4km_lora_v1_iter440_validation
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.serving.agent import construire_graphe
from src.text2sql.config import DATASET_PATH, DB_PATH, LLAMA_SERVER_URL, RESULTS_DIR
from src.text2sql.db import get_categorical_values, get_schema, run_sql
from src.text2sql.evaluate import METRIC_VERSION, ordre_impose, results_match
from src.text2sql.llm import Backend
from src.text2sql.prompts import PROMPT_VERSION, build_schema_context


def marquer_repetitions(sql_tentatives: list[str]) -> list[bool]:
    """Pour chaque tentative, vrai si elle recopie une tentative précédente de la même question.

    Une répétition est une relance gâchée : à température 0, le modèle a reçu l'erreur et a quand
    même regénéré une requête déjà essayée.

    Exemples attendus :
        ["SELECT a;", "SELECT a;", "SELECT b;"]  → [False, True, False]
        ["SELECT a;", "SELECT b;", "SELECT a;"]  → [False, False, True]   (pas seulement la précédente)
        ["SELECT a;"]                            → [False]                (la 1re ne répète jamais rien)
    """
    deja_vues = set()
    marques = []
    for sql in sql_tentatives:
        cle = " ".join(sql.split())
        marques.append(cle in deja_vues)
        deja_vues.add(cle)
    return marques


def evaluer_agent(
    backend: Backend, examples, schema: str, db_path: Path = DB_PATH, max_tentatives: int = 3, limit: int | None = None
) -> pd.DataFrame:
    """Une ligne par question : toutes les tentatives, et la justesse de la première et de la dernière."""
    # max_lignes=None : un résultat tronqué serait comparé au résultat complet de la référence,
    # et compté faux alors que la requête est juste.
    graphe = construire_graphe(backend, db_path, max_tentatives=max_tentatives, max_lignes=None)
    records = []
    for ex in tqdm(list(examples)[:limit], desc="agent"):
        start = time.perf_counter()
        etat = graphe.invoke({"question": ex["question"], "schema": schema, "tentatives": []})
        latence = time.perf_counter() - start

        gold = run_sql(ex["sql"], db_path)
        if gold is None:
            raise ValueError(f"Gold SQL invalide, à corriger dans le dataset : {ex['question']!r}")
        ordered = ordre_impose(ex["sql"])
        tentatives = etat["tentatives"]
        premiere, finale = tentatives[0].resultat, tentatives[-1].resultat

        records.append({
            "question": ex["question"],
            "famille": ex["famille"],
            "difficulte": ex["difficulte"],
            "gold_sql": ex["sql"],
            # Listes alignées (une entrée par tentative) : de quoi rejouer l'analyse sans relancer le modèle.
            "sql_tentatives": [t.sql for t in tentatives],
            "erreurs_tentatives": [t.resultat.erreur for t in tentatives],
            "n_tentatives": len(tentatives),
            "repetitions": marquer_repetitions([t.sql for t in tentatives]),
            "valide_t1": premiere.erreur is None,
            "valide": finale.erreur is None,
            # correct_t1 doit retrouver le run evaluate.py du même GGUF : c'est le contrôle de déterminisme.
            "correct_t1": results_match(premiere.lignes, gold, ordered),
            "correct": results_match(finale.lignes, gold, ordered),
            "correct_relache": results_match(finale.lignes, gold, ordered, colonnes_en_trop=True),
            "latence_s": round(latence, 2),
        })
    return pd.DataFrame(records)


def resumer(df: pd.DataFrame) -> None:
    relancees = df[~df["valide_t1"]]
    print(
        f"\nSQL valide : {df['valide_t1'].mean():.1%} (tentative 1) → {df['valide'].mean():.1%} (agent)"
        f"\nExecution accuracy : {df['correct_t1'].mean():.1%} (tentative 1) → {df['correct'].mean():.1%} (agent)"
        f" | EX relâchée (agent) : {df['correct_relache'].mean():.1%}"
        f"\nQuestions relancées : {len(relancees)} | devenues valides : {relancees['valide'].sum()}"
        f" | devenues justes : {relancees['correct'].sum()}"
        # Les relances sont les tentatives 2 et suivantes : n_tentatives - 1 par question.
        f"\nRelances : {(df['n_tentatives'] - 1).sum()} | dont répétitions : {df['repetitions'].apply(sum).sum()}"
        f"\nLatence moyenne : {df['latence_s'].mean():.1f}s"
    )
    print("\nNombre de tentatives :")
    print(df["n_tentatives"].value_counts().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--url", default=LLAMA_SERVER_URL, help="adresse de llama-server (défaut : $LLAMA_SERVER_URL, sinon localhost:8080)"
    )
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--max-tentatives", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="n'évaluer que les N premiers exemples")
    parser.add_argument("--name", default=None, help="nom du run (fichiers de résultats)")
    args = parser.parse_args()

    from datasets import load_from_disk

    from src.text2sql.llm import LlamaServerBackend

    ds = load_from_disk(str(DATASET_PATH))
    # Même contexte que evaluate.py et que la CLI de l'agent : la tentative 1 doit être le prompt mesuré.
    schema = build_schema_context(get_schema(), get_categorical_values())
    backend = LlamaServerBackend(args.url)

    df = evaluer_agent(backend, ds[args.split], schema, max_tentatives=args.max_tentatives, limit=args.limit)
    resumer(df)

    name = args.name or f"agent_{args.split}"
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_json(RESULTS_DIR / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "url": args.url,
        "model_servi": backend.model_id,
        "split": args.split,
        "n_exemples": len(df),
        "max_tentatives": args.max_tentatives,
        "prompt_version": PROMPT_VERSION,
        "metric_version": METRIC_VERSION,
        "sql_valide_t1": round(float(df["valide_t1"].mean()), 4),
        "sql_valide": round(float(df["valide"].mean()), 4),
        "execution_accuracy_t1": round(float(df["correct_t1"].mean()), 4),
        "execution_accuracy": round(float(df["correct"].mean()), 4),
        "execution_accuracy_relachee": round(float(df["correct_relache"].mean()), 4),
    }
    (RESULTS_DIR / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nRésultats enregistrés dans {RESULTS_DIR / name}.jsonl")


if __name__ == "__main__":
    main()
