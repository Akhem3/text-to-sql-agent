"""Benchmark Text-to-SQL : execution accuracy sur un split du dataset.

Exemples (depuis la racine du repo) :
    python -m src.text2sql.evaluate --split validation --shots 0 --limit 10
    python -m src.text2sql.evaluate --split validation --shots 5
    python -m src.text2sql.evaluate --backend llamacpp --model models/llama-q4.gguf --split test
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import DATASET_PATH, DB_PATH, DEFAULT_MODEL, RESULTS_DIR
from .db import get_schema, run_sql
from .prompts import PROMPT_VERSION, build_messages, pick_shots
from .sql_utils import extract_sql


def _normalize(rows):
    return [tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in rows]


def results_match(pred, gold, ordered: bool) -> bool:
    """Compare deux résultats d'exécution. L'ordre ne compte que si le gold a un ORDER BY."""
    if pred is None:
        return False
    pred, gold = _normalize(pred), _normalize(gold)
    if ordered:
        return pred == gold
    return sorted(pred, key=repr) == sorted(gold, key=repr)


def evaluate(backend, examples, schema: str, shots=(), db_path: Path = DB_PATH, limit: int | None = None) -> pd.DataFrame:
    """Génère, exécute et compare. Une ligne par exemple, sortie brute du modèle conservée."""
    records = []
    for ex in tqdm(list(examples)[:limit], desc="évaluation"):
        start = time.perf_counter()
        raw_output = backend.generate(build_messages(ex["question"], schema, shots))
        latency = time.perf_counter() - start

        pred_sql = extract_sql(raw_output)
        pred = run_sql(pred_sql, db_path)
        gold = run_sql(ex["sql"], db_path)
        if gold is None:
            raise ValueError(f"Gold SQL invalide, à corriger dans le dataset : {ex['question']!r}")

        records.append({
            "question": ex["question"],
            "famille": ex["famille"],
            "difficulte": ex["difficulte"],
            "raw_output": raw_output,
            "pred_sql": pred_sql,
            "gold_sql": ex["sql"],
            "valide": pred is not None,
            "correct": results_match(pred, gold, ordered="ORDER BY" in ex["sql"].upper()),
            "latence_s": round(latency, 2),
        })
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> None:
    print(
        f"\nSQL valide : {df['valide'].mean():.1%} | Execution accuracy : {df['correct'].mean():.1%}"
        f" | latence moyenne : {df['latence_s'].mean():.1f}s"
    )
    for col in ("famille", "difficulte"):
        print(f"\nPar {col} :")
        print(df.groupby(col)["correct"].agg(["mean", "count"]).round(3).sort_values("mean").to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["hf", "llamacpp"], default="hf")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="id Hugging Face ou chemin d'un .gguf")
    parser.add_argument("--adapter", default=None, help="adaptateur LoRA (backend hf uniquement)")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--shots", type=int, default=0, help="nombre d'exemples few-shot tirés du train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="n'évaluer que les N premiers exemples")
    parser.add_argument("--name", default=None, help="nom du run (fichiers de résultats)")
    args = parser.parse_args()

    from datasets import load_from_disk

    from .llm import load_backend

    ds = load_from_disk(str(DATASET_PATH))
    schema = get_schema()
    shots = pick_shots(ds["train"], args.shots, args.seed) if args.shots else []
    backend = load_backend(args.backend, args.model, args.adapter)

    df = evaluate(backend, ds[args.split], schema, shots, limit=args.limit)
    summarize(df)

    name = args.name or f"{args.backend}_{args.shots}shot_{args.split}"
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_json(RESULTS_DIR / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "backend": args.backend,
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "n_exemples": len(df),
        "shots": args.shots,
        "shots_questions": [s["question"] for s in shots],
        "seed": args.seed,
        "prompt_version": PROMPT_VERSION,
        "sql_valide": round(float(df["valide"].mean()), 4),
        "execution_accuracy": round(float(df["correct"].mean()), 4),
    }
    (RESULTS_DIR / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nRésultats enregistrés dans {RESULTS_DIR / name}.jsonl")


if __name__ == "__main__":
    main()
