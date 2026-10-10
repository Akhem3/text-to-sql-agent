import os

from langchain_community.utilities import SQLDatabase

current_dir = os.path.dirname(os.path.abspath(__file__))
db_file_path = os.path.abspath(os.path.join(current_dir, "../../data/raw/banque_entreprise.db"))

# URI de connexion 
db_uri = f"sqlite:///{db_file_path}"
db = SQLDatabase.from_uri(db_uri)

def get_schema(_):
    """Extrait le schéma DDL (Data Definition Language) de la base et 3 lignes d'exemples."""
    return db.get_table_info()

if __name__ == "__main__":
    print("--- Schéma extrait automatiquement par LangChain pour le prompt --- \n")
    print(get_schema(None))