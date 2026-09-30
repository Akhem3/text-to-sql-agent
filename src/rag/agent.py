"""Agent Text-to-SQL avec auto-correction (Llama 3.1 8B + LangChain LCEL)."""

import re
from dataclasses import dataclass
from functools import lru_cache

import torch
from database_setup import db
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# ---------------------------------------------------------------------------
# 1. Modèle
# ---------------------------------------------------------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_ITERATIONS = 3

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, clean_up_tokenization_spaces=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)

# LangChain -> rôles attendus par le chat template de Llama
ROLES = {"system": "system", "human": "user", "ai": "assistant"}


def make_llm(max_new_tokens: int = 512, stream: bool = False) -> RunnableLambda:
    """Runnable LangChain : ChatPromptValue -> texte généré.

    Le chat template de Llama ajoute lui-même <|begin_of_text|> et les
    en-têtes : plus de tokens spéciaux écrits à la main dans les prompts,
    et un seul BOS (tokenisation faite une seule fois).
    """
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None

    def _generate(prompt_value) -> str:
        messages = [{"role": ROLES[m.type], "content": m.content} for m in prompt_value.to_messages()]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # SQL : sortie déterministe
                temperature=None,
                top_p=None,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return RunnableLambda(_generate)


llm_sql = make_llm(max_new_tokens=512)                   # silencieux
llm_synthese = make_llm(max_new_tokens=512, stream=True)  # streamé uniquement ici

# ---------------------------------------------------------------------------
# 2. Schéma, extraction et garde-fou
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_schema() -> str:
    return db.get_table_info()


FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|ATTACH|PRAGMA|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def extract_sql(text: str) -> str:
    """Isole la requête : bloc ```sql``` si présent, puis de SELECT/WITH à ';'."""
    block = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if block:
        text = block.group(1)
    start = re.search(r"\b(SELECT|WITH)\b.*", text, re.DOTALL | re.IGNORECASE)
    if start:
        text = start.group(0)
    return text.split(";")[0].strip() + ";"


def validate_readonly(sql: str) -> None:
    """Lève ValueError si la requête n'est pas un SELECT/WITH en lecture seule."""
    if not re.match(r"\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Seules les requêtes SELECT / WITH sont autorisées.")
    if FORBIDDEN.search(sql):
        raise ValueError("Mot-clé d'écriture ou d'administration interdit.")


# ---------------------------------------------------------------------------
# 3. Prompts (format chat, sans tokens spéciaux)
# ---------------------------------------------------------------------------
PROMPT_INITIAL = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un expert SQL de niveau Senior. Génère uniquement une requête SQL valide, "  # noqa: ISC004
     "sans explication ni Markdown.\n"
     "Schéma de la base de données :\n{table_info}\n\n"
     "RÈGLE D'OR : pour les questions de type « Top N par catégorie », utilise des "
     "fonctions de fenêtrage avec PARTITION BY.\n"
     "Exemple :\n"
     "Question : Quels sont les 3 employés les mieux payés par département ?\n"
     "Requête : WITH Ranked AS (SELECT nom, departement, salaire, "
     "ROW_NUMBER() OVER(PARTITION BY departement ORDER BY salaire DESC) AS rn "
     "FROM employes) SELECT nom, departement, salaire FROM Ranked WHERE rn <= 3;"),
    ("human", "Question : {question}"),
])

PROMPT_CORRECTION = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un expert SQL. Ta requête précédente a échoué. Analyse le problème et "  # noqa: ISC004
     "corrige la requête.\n"
     "RÈGLE CRITIQUE : si une colonne utilise la mauvaise table ou le mauvais alias "
     "(ex : T2 au lieu de T3), applique la correction dans TOUTES les clauses "
     "(SELECT, WHERE, GROUP BY, ORDER BY...).\n"
     "Ne génère que la requête SQL corrigée, sans explication.\n"
     "Schéma de la base de données :\n{table_info}"),
    ("human",
     "Question initiale : {question}\n"  # noqa: ISC004
     "Requête erronée : {bad_query}\n"
     "Problème rencontré : {error_message}\n\n"
     "Requête SQL corrigée :"),
])

PROMPT_SYNTHESE = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un analyste de données expert. Formule une réponse claire et professionnelle, "  # noqa: ISC004
     "en phrases naturelles. Ne montre JAMAIS de code Python ni de tuples. "
     "Formate les montants en euros.\n"
     "Données brutes : {resultats_sql}"),
    ("human", "Question : {question}"),
])

# ---------------------------------------------------------------------------
# 4. Chaînes LCEL (définies une seule fois)
# ---------------------------------------------------------------------------
inject_schema = RunnablePassthrough.assign(table_info=lambda _: get_schema())

chain_initiale = inject_schema | PROMPT_INITIAL | llm_sql | extract_sql
chain_correction = inject_schema | PROMPT_CORRECTION | llm_sql | extract_sql
chain_synthese = PROMPT_SYNTHESE | llm_synthese


# ---------------------------------------------------------------------------
# 5. Agent avec auto-correction
# ---------------------------------------------------------------------------
@dataclass
class AgentResult:
    sql: str
    rows: str
    attempts: int


def run_sql_agent(question: str, max_iterations: int = MAX_ITERATIONS) -> AgentResult | None:
    print(f"\n--- Agent SQL : {question!r} ---")
    sql = chain_initiale.invoke({"question": question})

    for attempt in range(1, max_iterations + 1):
        print(f"\n[Tentative {attempt}] {sql}")
        try:
            validate_readonly(sql)
            rows = db.run(sql)
            # Une requête valide mais vide est suspecte : on laisse une chance de correction
            if not rows or rows.strip() in ("", "[]"):  # noqa: SIM102
                if attempt < max_iterations:
                    raise ValueError("La requête s'exécute mais ne retourne aucune ligne.")
            print(f"✅ Succès :\n{rows}")
            return AgentResult(sql=sql, rows=str(rows), attempts=attempt)
        except Exception as e:  # erreur SQL, requête refusée ou résultat vide  # noqa: BLE001
            print(f"❌ {e}")
            if attempt == max_iterations:
                print("⚠️ Nombre maximum de tentatives atteint.")
                return None
            sql = chain_correction.invoke(
                {"question": question, "bad_query": sql, "error_message": str(e)}
            )
    return None


# ---------------------------------------------------------------------------
# 6. Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    question = "Quels sont les 3 clients avec les plus gros montants d'abonnements au sein de chaque région ?"

    result = run_sql_agent(question)
    if result:
        print("\n📝 Synthèse :\n")
        chain_synthese.invoke({"resultats_sql": result.rows, "question": question})
        print()
    else:
        print("\n❌ Impossible de répondre : la requête SQL a échoué.")