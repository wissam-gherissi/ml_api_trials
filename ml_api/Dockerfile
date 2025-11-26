# 1. Base Image : On prend une version "slim" pour réduire la taille (sécurité + vitesse)
FROM python:3.9-slim

# 2. Variables d'environnement pour Python en mode Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \ 
    TRANSFORMERS_CACHE=/tmp/huggingface

# 3. Dossier de travail
WORKDIR /app

# 4. OPTIMISATION DU CACHE : On copie SEULEMENT les requirements d'abord
COPY requirements.txt .

# 5. Installation des dépendances
# --no-cache-dir : Ne pas garder les fichiers d'installation (gain de place)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 6. Maintenant, on copie le reste du code
# Si vous modifiez app.py, Docker reprendra la construction À PARTIR D'ICI seulement.
COPY . .

# 7. SÉCURITÉ : Créer un utilisateur non-root
RUN useradd -m appuser
USER appuser

# 8. Documentation du port
EXPOSE 8080

# 9. Lancement avec Gunicorn (Process Manager) + Uvicorn (Workers)
# C'est la configuration standard pour la prod.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]