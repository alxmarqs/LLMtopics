import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Configurações
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro')

FILE_DATASET = "urbanismo_dataset.jsonl"

def gerar_exemplos_sinteticos(quantidade=10):
    print(f"=== GERANDO {quantidade} NOVOS EXEMPLOS PARA O TREINAMENTO LOCAL ===")
    
    prompt = """
    Aja como um Professor Doutor em Urbanismo e Planejamento Urbano especialista em Uberlândia-MG.
    Sua tarefa é criar um conjunto de dados para treinar uma IA local.
    
    Cada exemplo deve ser um JSON com 'instruction' (pergunta técnica) e 'output' (resposta completa e profissional).
    Foque em:
    1. Cruzamento de Sintaxe Espacial (Betweenness vs Closeness).
    2. Plano Diretor de Uberlândia (Zoneamento, Recuos, Uso do Solo).
    3. Casos práticos: Bairro Santa Mônica, Jaraguá, Aclimação.
    
    Formato:
    {"instruction": "...", "output": "..."}
    
    Gere exatamente {quantidade} exemplos únicos, variados e com respostas ricas.
    """.replace("{quantidade}", str(quantidade))

    try:
        response = model.generate_content(prompt)
        # Tenta extrair apenas os JSONs da resposta
        linhas = response.text.strip().split('\n')
        novos_exemplos = []
        for linha in linhas:
            if linha.startswith('{'):
                try:
                    # Valida se é um JSON vivo
                    json.loads(linha)
                    novos_exemplos.append(linha)
                except:
                    continue
        
        if novos_exemplos:
            with open(FILE_DATASET, "a", encoding="utf-8") as f:
                for ex in novos_exemplos:
                    f.write(ex + "\n")
            print(f"SUCESSO: {len(novos_exemplos)} novos exemplos adicionados ao {FILE_DATASET}!")
        else:
            print("ERRO: O modelo não gerou JSONs válidos.")
            print("Resposta bruta do IA:")
            print(response.text)
            
    except Exception as e:
        print(f"Falha ao gerar dados: {e}")

if __name__ == "__main__":
    # Vamos gerar em blocos de 10 para evitar timeouts
    gerar_exemplos_sinteticos(10)
