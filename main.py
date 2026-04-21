from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load the trained model
with open('models/detector_model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

class InputText(BaseModel):
    text: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/predict")
async def predict(data: InputText):
    vec_text = vectorizer.transform([data.text])
    prob = model.predict_proba(vec_text)[0]
    ai_score = prob[1] # Probability of being AI
    
    # Decision Layer
    if ai_score >= 0.80:
        verdict = "AI Generated"
    elif ai_score >= 0.70:
        verdict = "Likely AI / Uncertain"
    else:
        verdict = "Human Written"
        
    return {
        "verdict": verdict,
        "confidence": f"{ai_score * 100:.2f}%"
    }