import json
from pathlib import Path

from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "hf_dataset"

TEST_FILE = "dataset_text_to_sql_test.md"
REPLACE_FILE = "dataset_text_to_sql_test_replace8.md"
FAMILLE_A_REMPLACER = "Évolution dans le temps"

N_VALID = 100
SEED = 42

FAMILLE_MAP = {
    "Détection d'absence (LEFT JOIN ... IS NULL / NOT EXISTS)": "Détection d'absence",
    "GROUP BY HAVING": "GROUP BY + HAVING",
    "Jointures 2 tables": "Jointures à 2 tables",
    "Jointures 3 tables": "Jointures à 3 tables",
    "Top-N par groupe (CTE + ROW_NUMBER)": "Top-N par groupe",
}

def clean_sql(sql: str) -> str:
    return sql.replace("\\n", "\n").replace("\\t", "\t").strip()

def read_jsonl_md(path: Path) -> list[dict]:
    """Lit un fichier .md contenant un objet JSON par ligne."""
    rows = []

    for i, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()

        # Ignore les lignes qui ne contiennent pas de JSON
        if not line.startswith("{"):
            continue

        try:
            row = json.loads(line)
            row["famille"] = FAMILLE_MAP.get(row["famille"], row["famille"])
            rows.append(row)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{path.name}:{i} JSON invalide : {e}"
            ) from e

    return rows


def build() -> DatasetDict:
    """Construit les datasets train, validation et test."""

    pool, test, replacement = [], [], []

    files = sorted(RAW.glob("dataset_text_to_sql_*.md"))

    if not files:
        raise FileNotFoundError(f"Aucun fichier dataset trouvé dans : {RAW}")

    print(f"Fichiers trouvés : {len(files)}")

    for path in files:
        rows = read_jsonl_md(path)
        print(f"{path.name}: {len(rows)} exemples")

        if path.name == TEST_FILE:
            test.extend(rows)
        elif path.name == REPLACE_FILE:
            replacement.extend(rows)      # jamais dans le pool
        else:
            pool.extend(rows)

    # Remplacement dans le test
    if any(r["famille"] == FAMILLE_A_REMPLACER for r in replacement):
        raise ValueError("Le fichier de remplacement contient encore la famille à remplacer")

    n_before = len(test)
    test = [r for r in test if r["famille"] != FAMILLE_A_REMPLACER]
    n_removed = n_before - len(test)

    if n_removed != len(replacement):
        raise ValueError(
            f"{n_removed} cas retirés du test, {len(replacement)} de remplacement : "
            "les nombres doivent être égaux"
        )

    # Les familles de remplacement doivent exister dans le train
    unknown = {r["famille"] for r in replacement} - {r["famille"] for r in pool}
    if unknown:
        raise ValueError(f"Familles absentes du train : {unknown}")

    test.extend(replacement)

    print()
    print(f"Test : {n_removed} cas remplacés par {len(replacement)} nouveaux")
    print(f"Total train + validation : {len(pool)}")
    print(f"Total test               : {len(test)}")

    # Vérifications
    if not pool:
        raise ValueError("Aucun exemple pour train/validation.")

    if not test:
        raise ValueError("Aucun exemple dans le dataset de test.")

    if len(pool) <= N_VALID:
        raise ValueError(
            f"Pas assez d'exemples pour créer la validation : "
            f"{len(pool)} disponibles, {N_VALID} demandés."
        )
    
    test_q = {r["question"].strip().lower() for r in test}
    test_sql = {" ".join(r["sql"].lower().split()) for r in test}

    overlap = [
        r for r in pool
        if r["question"].strip().lower() in test_q
        or " ".join(r["sql"].lower().split()) in test_sql
    ]
    #Gestion des doublons dans le pool vs test
    if overlap:
        print(f"⚠ {len(overlap)} exemples retirés du pool (déjà présents dans le test) :")
        for r in overlap:
            print(f"   - [{r['famille']}] {r['question'][:70]}")
        overlap_ids = {id(r) for r in overlap}
        pool = [r for r in pool if id(r) not in overlap_ids]
        
        # Nettoyage des SQL (\n littéraux) avant la conversion en Dataset
        for r in pool + test:
            r["sql"] = clean_sql(r["sql"])

    # Conversion en Dataset Hugging Face
    full = Dataset.from_list(pool)

    # On garde le schéma d'origine (famille en string) pour le restaurer après
    string_features = full.features.copy()

    # Encodage temporaire en ClassLabel, uniquement pour la stratification
    full = full.class_encode_column("famille")
    names = full.features["famille"].names  # liste des vrais noms de famille

    split = full.train_test_split(
        test_size=N_VALID,
        stratify_by_column="famille",
        seed=SEED,
    )

    def decode_famille(ex):
        return {"famille": names[ex["famille"]]}

    # On retrouve le nom à partir de l'entier, et on remet le type string
    train = split["train"].map(decode_famille, features=string_features)
    validation = split["test"].map(decode_famille, features=string_features)

    # Dataset test (déjà en chaînes de caractères)
    test_dataset = Dataset.from_list(test)

    return DatasetDict(
        {
            "train": train,
            "validation": validation,
            "test": test_dataset,
        }
    )


def main() -> None:
    ds = build()

    print()
    print("Dataset final :")
    print(ds)

    # Aperçu d'un exemple pour vérifier que "famille" est bien du texte
    print()
    print("Exemple train[0] :")
    print(ds["train"][0])

    # Crée le dossier de sortie s'il n'existe pas
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarde du DatasetDict Hugging Face
    ds.save_to_disk(OUT)

    print()
    print(f"Dataset sauvegardé dans : {OUT}")


if __name__ == "__main__":
    main()