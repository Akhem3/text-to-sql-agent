"""Prompts : une seule source de vérité pour l'évaluation, le few-shot, le fine-tuning et le serving.

Si le format change ici, il change partout : le modèle fine-tuné est entraîné
exactement sur ce qu'il verra à l'inférence.
"""

import random
from collections import defaultdict

PROMPT_VERSION = "v2"

# Une règle par type d'erreur observé sur la baseline v1 : on pourra mesurer l'effet de chacune
# en comptant son erreur cible avant / après. 
REGLES = (
    # Requêtes invalides : EXTRACT(YEAR FROM ...) n'existe pas en SQLite.
    "Les dates sont du texte au format 'YYYY-MM-DD'. Utilise strftime, par exemple "
    "strftime('%Y', date_inscription) = '2025' ou strftime('%Y-%m', date_facturation) = '2025-03'. "
    "EXTRACT, YEAR() et MONTH() n'existent pas en SQLite.",
    # 15 des 18 réponses justes seulement en EX relâchée étaient des SELECT *.
    "Sélectionne uniquement les colonnes qui répondent à la question. N'utilise jamais SELECT *.",
    # 7 échecs avec des valeurs mal écrites ('payée' au lieu de 'Payée').
    "Pour filtrer une colonne listée dans « Valeurs possibles », recopie l'une de ces valeurs "
    "exactement, majuscules et accents compris.",
    "Les booléens valent 0 ou 1 : un abonnement actif a statut_actif = 1.",
)

SYSTEM_TEMPLATE = (
    "Tu es un expert SQL (SQLite). Réponds uniquement avec une requête SQL valide, "
    "sans explication ni Markdown.\n\nRègles :\n{regles}\n\nSchéma :\n{schema}"
)


def _sql_literal(value) -> str:
    """Valeur écrite comme dans une requête SQLite, pour que le modèle la recopie telle quelle."""
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def build_schema_context(schema: str, values: dict[str, list]) -> str:
    """Schéma + valeurs possibles des colonnes catégorielles, en commentaires SQL sous les CREATE TABLE."""
    if not values:
        return schema
    lignes = [f"-- {col} : {', '.join(_sql_literal(v) for v in vals)}" for col, vals in values.items()]
    return schema + "\n\n-- Valeurs possibles :\n" + "\n".join(lignes)


def build_messages(question: str, schema: str, shots=()) -> list[dict]:
    """Messages au format chat. `shots` : exemples {question, sql} injectés en tours user/assistant."""
    regles = "\n".join(f"- {regle}" for regle in REGLES)
    messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(regles=regles, schema=schema)}]
    for ex in shots:
        messages.append({"role": "user", "content": f"Question : {ex['question']}"})
        messages.append({"role": "assistant", "content": ex["sql"]})
    messages.append({"role": "user", "content": f"Question : {question}"})
    return messages


def to_sft_record(example: dict, schema: str) -> dict:
    """Exemple de fine-tuning : le même prompt qu'à l'inférence, plus la réponse attendue."""
    messages = build_messages(example["question"], schema)
    messages.append({"role": "assistant", "content": example["sql"]})
    return {"messages": messages}


def pick_shots(train, k: int, seed: int = 42) -> list[dict]:
    """k exemples du train, de familles différentes (un par famille), tirage reproductible."""
    rng = random.Random(seed)
    by_family = defaultdict(list)
    for ex in train:
        by_family[ex["famille"]].append(ex)
    families = sorted(by_family)
    rng.shuffle(families)
    return [rng.choice(by_family[family]) for family in families[:k]]
