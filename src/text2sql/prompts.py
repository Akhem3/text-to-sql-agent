"""Prompts : une seule source de vérité pour l'évaluation, le few-shot, le fine-tuning et le serving.

Si le format change ici, il change partout : le modèle fine-tuné est entraîné
exactement sur ce qu'il verra à l'inférence.
"""

import random
from collections import defaultdict

PROMPT_VERSION = "v1"

SYSTEM_TEMPLATE = (
    "Tu es un expert SQL (SQLite). Réponds uniquement avec une requête SQL valide, "
    "sans explication ni Markdown.\n\nSchéma :\n{schema}"
)


def build_messages(question: str, schema: str, shots=()) -> list[dict]:
    """Messages au format chat. `shots` : exemples {question, sql} injectés en tours user/assistant."""
    messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(schema=schema)}]
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
