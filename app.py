from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Carrega o modelo Falcon-7B-Instruct
model = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device_map="auto")

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"status": "API funcionando com Falcon-7B"}

@app.post("/chat")
def chat(req: ChatRequest):
    resposta = model(req.prompt, max_length=250, num_return_sequences=1)
    return {"response": resposta[0]["generated_text"]}
