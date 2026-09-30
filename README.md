# Agent Text-to-SQL & LLMOps (Projet R&D IA)

Pipeline end-to-end de conversion de texte en requêtes SQL complexes combinant agent autonome, injection dynamique de schémas et fine-tuning.

## 🚀 Fonctionnalités Clés
- **Agent Autonome (LangChain / LCEL) :** Utilisation de Llama 3.1 8B pour traduire le langage métier en requêtes SQL.
- **Auto-Correction :** Boucle de rétroaction capturant les erreurs de la base de données SQLite pour corriger itérativement la requête SQL.
- **Injection Dynamique de Schémas :** Alimentation du prompt en temps réel avec la structure relationnelle des tables.
- **Fine-Tuning LoRA/QLoRA (à venir) :** Préparation et entraînement (Hugging Face / PyTorch) adapté aux fonctions et dialectes spécifiques.
- **Serving & Ops (à venir) :** Optimisation de l'inférence via vLLM et traçabilité des performances.

## 🛠️ Stack Technique
- **Langage :** Python
- **LLM & GenAI :** Llama 3.1, Hugging Face (Transformers, PEFT), PyTorch
- **Orchestration :** LangChain (LCEL)
- **Base de données :** SQLite