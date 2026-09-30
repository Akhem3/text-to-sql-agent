import torch
from database_setup import db
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# 1. Chargement du modèle Llama 3.1
model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    clean_up_tokenization_spaces=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16
).to(device)

streamer = TextStreamer(tokenizer, skip_prompt=True)

# 2. Création du pipeline compatible LangChain
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=512,
    max_length=None,
    truncation=True, 
    return_full_text=False,
    streamer=streamer
)
llm = HuggingFacePipeline(pipeline=pipe)

# 3. Fonction partagée pour injecter le schéma à la volée
def get_schema(_):
    return db.get_table_info()

# 4. Définition du Prompt Initial (avec la règle d'or pour le SQL complexe)
template_initial = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Tu es un expert SQL de niveau Senior. Ta mission est de générer une requête SQL valide. Ne donne aucune explication.
Schéma de la base de données :
{table_info}

RÈGLE D'OR : Pour les questions demandant un "Top N par catégorie" (ex: les meilleurs clients par région), tu DOIS utiliser des fonctions de fenêtrage avec PARTITION BY.
Exemple :
Question : Quels sont les 3 employés les mieux payés par département ?
Requête : WITH Ranked AS (SELECT nom, departement, salaire, ROW_NUMBER() OVER(PARTITION BY departement ORDER BY salaire DESC) as rn FROM employes) SELECT nom, departement, salaire FROM Ranked WHERE rn <= 3;
<|eot_id|><|start_header_id|>user<|end_header_id|>
Question : {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

prompt_initial = PromptTemplate.from_template(template_initial)

# 5. Prompt spécialisé pour la correction d'erreurs
template_correction = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Tu es un expert SQL. Ta requête précédente a échoué. Analyse l'erreur de la base de données et corrige rigoureusement la requête.
RÈGLE CRITIQUE : Si une colonne utilise la mauvaise table ou le mauvais alias (ex: T2 au lieu de T3), tu dois appliquer la correction dans TOUTES les clauses de la requête (SELECT, WHERE, GROUP BY, ORDER BY, etc...).
Ne génère que la requête SQL corrigée finale, sans explication ni texte supplémentaire.
Schéma de la base de données :
{table_info}<|eot_id|><|start_header_id|>user<|end_header_id|>
Question initiale : {question}
Requête erronée générée : {bad_query}
Erreur retournée : {error_message}

Génère la requête SQL corrigée :<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

prompt_correction = PromptTemplate.from_template(template_correction)

# 6. La boucle Agentic (Self-Correction)
def agent_sql_autonome(question, max_iterations=3):
    print(f"\n--- Lancement de l'Agent pour : '{question}' ---")
    
    # Étape A : Première tentative
    chain_initiale = RunnablePassthrough.assign(table_info=get_schema) | prompt_initial | llm | StrOutputParser()
    sql_query = chain_initiale.invoke({"question": question})
    
    # Nettoyage basique
    sql_query = sql_query.strip().replace("```sql", "").replace("```", "")
    
    iteration = 1
    
    while iteration <= max_iterations:
        print(f"\n[Tentative {iteration}] Requête testée : {sql_query}")
        
        try:
            result = db.run(sql_query)
            print(f"\n✅ SUCCÈS ! Résultat de la base de données :\n{result}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ ERREUR SQL interceptée : {error_msg}")
            
            if iteration == max_iterations:
                print("\n⚠️ L'agent a atteint le nombre maximum de tentatives d'autocorrection.")
                return None
                
            print("🔄 L'agent analyse l'erreur et génère une correction...")
            
            # Étape B : Appel de la chaîne de correction
            chain_correction = RunnablePassthrough.assign(table_info=get_schema) | prompt_correction | llm | StrOutputParser()
            
            sql_query = chain_correction.invoke({
                "question": question,
                "bad_query": sql_query,
                "error_message": error_msg
            })
            
            sql_query = sql_query.strip().replace("```sql", "").replace("```", "")
            iteration += 1

# 7. Le prompt de synthèse finale
template_reponse = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Tu es un analyste de données expert. Formule une réponse claire et professionnelle. 
Rédige des phrases naturelles et ne montre JAMAIS de code Python ou de parenthèses de tuples. Formate les montants en euros.
Données brutes : {resultats_sql}<|eot_id|><|start_header_id|>user<|end_header_id|>
Question : {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

prompt_reponse = PromptTemplate.from_template(template_reponse)

# 8. La chaîne LCEL de synthèse
chain_reponse = prompt_reponse | llm | StrOutputParser()

if __name__ == "__main__":
    question_utilisateur = "Quels sont les 3 clients avec les plus gros montants d'abonnements au sein de chaque région ?"
    
    resultats_bruts = agent_sql_autonome(question_utilisateur)
    
    if resultats_bruts:
        print("\n📝 Rédaction de la synthèse en cours...")
        
        synthese = chain_reponse.invoke({
            "resultats_sql": str(resultats_bruts),
            "question": question_utilisateur
        })

        synthese = synthese.replace("<|eot_id|>", "").strip()
        
        print(f"\n✨ RÉPONSE FINALE POUR L'UTILISATEUR :\n{synthese}")
    else:
        print("\n❌ Impossible de générer une réponse suite à l'échec de la requête SQL.")