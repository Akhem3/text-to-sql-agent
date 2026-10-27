import json
import random
from collections import defaultdict
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

#Correction de cas
def clean_sql(sql: str) -> str:
    return sql.replace("\\n", "\n").replace("\\t", "\t").strip()


def normaliser_sql(sql: str) -> str:
    """Clé de comparaison : deux requêtes qui ne diffèrent que par la casse ou les blancs sont identiques."""
    return " ".join(sql.lower().split())

SQL_OVERRIDES = {
    "Quelles factures payées sont supérieures au montant moyen facturé pour leur forfait ?": """
WITH moyenne_forfait AS (
  SELECT a2.type_forfait, AVG(f2.montant_paye) AS moy
  FROM factures f2
  JOIN abonnements a2 ON f2.abonnement_id = a2.abonnement_id
  GROUP BY a2.type_forfait
)
SELECT f.facture_id, a.type_forfait, f.montant_paye
FROM factures f
JOIN abonnements a ON f.abonnement_id = a.abonnement_id
JOIN moyenne_forfait m ON m.type_forfait = a.type_forfait
WHERE f.statut_paiement = 'Payée'
  AND f.montant_paye > m.moy;""",
    "les 5 factures payees les plus cheres parmi celles superieures a la moyenne mensuelle de leur mois demission": """
WITH moyenne_mensuelle AS (
  SELECT strftime('%Y-%m', date_facturation) AS mois, AVG(montant_paye) AS moy
  FROM factures
  WHERE statut_paiement = 'Payée'
  GROUP BY mois
)
SELECT f1.facture_id, f1.montant_paye, f1.date_facturation
FROM factures f1
JOIN moyenne_mensuelle m ON m.mois = strftime('%Y-%m', f1.date_facturation)
WHERE f1.statut_paiement = 'Payée'
  AND f1.montant_paye > m.moy
ORDER BY f1.montant_paye DESC
LIMIT 5;""",
}

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
            row["sql"] = SQL_OVERRIDES.get(row["question"], clean_sql(row["sql"]))
            rows.append(row)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{path.name}:{i} JSON invalide : {e}"
            ) from e

    return rows


def split_par_groupe(pool: list[dict], n_valid: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Split train/validation stratifié par famille, où les paraphrases restent ensemble.

    Deux questions avec le même SQL normalisé forment un groupe, envoyé en entier d'un
    seul côté : sinon le modèle récite en validation un SQL appris par cœur en train,
    et le score est surestimé. La validation fait donc environ n_valid exemples, pas exactement.
    """
    rng = random.Random(seed)

    # SQL normalisé -> paraphrases
    groupes = defaultdict(list)
    for r in pool:
        groupes[normaliser_sql(r["sql"])].append(r)

    # famille -> groupes (la famille d'un groupe est celle de sa première question)
    par_famille = defaultdict(list)
    for groupe in groupes.values():
        par_famille[groupe[0]["famille"]].append(groupe)

    train, validation = [], []
    for famille in sorted(par_famille):  # ordre fixe : le résultat ne dépend que du seed
        paquets = par_famille[famille]
        rng.shuffle(paquets)
        n_famille = sum(len(p) for p in paquets)
        quota = round(n_valid * n_famille / len(pool))  # part de la famille dans la validation
        compteur = 0 
        for paquet in paquets:
            if compteur < quota:
                validation.extend(paquet)
                compteur += len(paquet)
            else:
                train.extend(paquet)

    return train, validation


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
    test_sql = {normaliser_sql(r["sql"]) for r in test}

    overlap = [
        r for r in pool
        if r["question"].strip().lower() in test_q
        or normaliser_sql(r["sql"]) in test_sql
    ]
    #Gestion des doublons dans le pool vs test
    if overlap:
        print(f"⚠ {len(overlap)} exemples retirés du pool (déjà présents dans le test) :")
        for r in overlap:
            print(f"   - [{r['famille']}] {r['question'][:70]}")
        overlap_ids = {id(r) for r in overlap}
        pool = [r for r in pool if id(r) not in overlap_ids]

    train, validation = split_par_groupe(pool, N_VALID, SEED)

    # Conversion en Dataset Hugging Face
    return DatasetDict(
        {
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(validation),
            "test": Dataset.from_list(test),
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