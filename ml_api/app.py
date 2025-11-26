import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


# 1. Définition du Schéma de Données (Le Contrat)
# C'est ici que Pydantic brille : on force les types.
class RecruitmentInput(BaseModel):
    message_text: str
    source: str = "linkedin" # Valeur par défaut

class PredictionOutput(BaseModel):
    category: str
    confidence: float
    latency_ms: float

# 2. Initialisation de l'App
app = FastAPI(
    title="Recruitment Classifier Microservice",
    description="API haute performance pour classifier les messages candidats.",
    version="1.1.0"
)

# 3. Chargement du Modèle (VERSION LÉGÈRE)
# cross-encoder/nli-distilroberta-base fait ~300MB vs 1.6GB pour Bart-Large
MODEL_NAME = "cross-encoder/nli-distilroberta-base"
classifier = None
model_loaded = False

print(f"Initialisation du modèle léger: {MODEL_NAME}...")

try:
    # On charge le tokenizer et le modèle séparément pour mieux gérer les erreurs
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # Création du pipeline
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
    model_loaded = True
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"FATAL ERROR loading model: {e}")
CANDIDATE_LABELS = ["job application", "spam", "inquiry", "networking"]

# 4. Endpoints
@app.get("/health")
async def health_check():
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model failed to load")
    return {"status": "alive", "model_loaded": model_loaded}

@app.post("/predict", response_model=PredictionOutput)
async def predict(payload: RecruitmentInput):
    if not classifier:
        raise HTTPException(status_code=503, detail="Service initializing.")
    """Effectue une classification sur le texte reçu."""
    start_time = time.time()
    
    if len(payload.message_text) < 10:
        raise HTTPException(status_code=400, detail="Texte trop court.")

    # Inférence
    try:
        # Le modèle DistilRoberta est beaucoup plus rapide
        result = classifier(payload.message_text, CANDIDATE_LABELS)
        
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        duration = (time.time() - start_time) * 1000
        
        return {
            "category": top_label,
            "confidence": top_score,
            "latency_ms": round(duration, 2)
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
