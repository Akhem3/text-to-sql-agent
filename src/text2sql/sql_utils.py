"""Nettoyage de la sortie brute du modèle."""

import re


def extract_sql(text: str) -> str:
    """Isole la requête : bloc ```sql``` si présent, puis de SELECT/WITH jusqu'au premier ';'."""
    block = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if block:
        text = block.group(1)
    start = re.search(r"\b(SELECT|WITH)\b.*", text, re.DOTALL | re.IGNORECASE)
    if start:
        text = start.group(0)
    return text.split(";")[0].strip() + ";"
