from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = FastAPI(title="ArqGeo API", description="API de Inteligência Urbana e Sintaxe Espacial")

# --- Modelos de Dados ---
class AnalysisRequest(BaseModel):
    bairro: str
    nodos_criticos: List[dict] # Lista de {rua, lat, lon, b_val, c_val}

class QueryRequest(BaseModel):
    pergunta: str

# --- Carregamento de Recursos (Singleton/Lazy) ---
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ID = "arqgeo-llm-lora"

resources = {}

def get_ml_resources():
    if "tokenizer" not in resources:
        print("[API] Carregando Modelos...")
        resources["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, device_map="cpu"
        )
        if os.path.exists(LORA_ID):
            resources["model"] = PeftModel.from_pretrained(base_model, LORA_ID)
        else:
            resources["model"] = base_model
            
    return resources

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok", "hardware": "cpu", "model": MODEL_ID}

@app.post("/analyze")
async def analyze_nodes(req: AnalysisRequest):
    res = get_ml_resources()
    tokenizer = res["tokenizer"]
    model = res["model"]
    
    nodos_str = ""
    for i, n in enumerate(req.nodos_criticos):
        coords = f"{n.get('lat', 0):.5f}, {n.get('lon', 0):.5f}"
        nodos_str += f"- Ponto {i+1} ({coords}): Fluxo {n.get('b_val', 0):.4f}, Acesso {n.get('c_val', 0):.4f}\n"

    prompt = f"<|system|>\nVocê é o ArqGeo-API. Responda APENAS no formato:\n- PONTO: [Nome]\n- NEGÓCIO: [Ideia]\n- VALOR M2: [R$]\n- PECULIARIDADE: [Detalhe]\n\nUse [PADRÃO MERCADO UBERLÂNDIA]. Sem introduções.</s>\n<|user|>\nUberlândia - {req.bairro}:\n{nodos_str}</s>\n<|assistant|>\n"

    # 2. Inferência
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.6, repetition_penalty=1.3, do_sample=True)
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = full_text.split("<|assistant|>")[-1].strip()
    
    return {"bairro": req.bairro, "laudo_criativo": response_text}

@app.delete("/rag", include_in_schema=False)
def disabled_rag():
    return {"message": "RAG Desativado pelo usuário."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
