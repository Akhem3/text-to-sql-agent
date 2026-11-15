# Image de l'agent servi : la CLI Text-to-SQL, sans le modèle.
# llama-server tourne sur l'hôte (Metal) ; le conteneur le joint par LLAMA_SERVER_URL (voir compose.yaml).
# Doc : https://docs.docker.com/reference/dockerfile/ et https://docs.astral.sh/uv/guides/integration/docker/

# Version exacte de .python-version : avec UV_PYTHON_DOWNLOADS=never, une autre 3.13.x ferait échouer le build.
FROM python:3.13.9-slim

# Binaire uv, copié depuis son image officielle (plus léger et plus rapide qu'un pip install uv).
# Version épinglée sur celle du poste de dev : même résolveur, même lecture de uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

# Réglages d'uv pour une image :
# - COMPILE_BYTECODE : compile les .pyc au build, pas au premier lancement du conteneur ;
# - PYTHON_DOWNLOADS=never : utiliser le Python de l'image, échouer plutôt qu'en télécharger un.
ENV UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dépendances avant le code : la couche uv sync reste en cache quand seul src/ change.
# --locked : échoue si uv.lock ne correspond plus à pyproject.toml ; --no-default-groups : sans le groupe ml (torch, mlx...).
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-default-groups

COPY src/ /app/src

# Compte de service sans privilèges : l'agent ne fait que lire /app et la base montée.
RUN useradd -r agent_text2sql
USER agent_text2sql

# Le venv d'uv passe devant dans le PATH : « python » a httpx et langgraph, sans uv run.
ENV PATH="/app/.venv/bin:$PATH"

# Forme exec : la question passée à `docker run` / `compose run` s'ajoute à la fin de la liste.
ENTRYPOINT ["python", "-m", "src.serving.agent"]