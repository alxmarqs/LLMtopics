# ArqGeo: Inteligência Urbana e Real Estate via LLMs Locais
**Disciplina:** Tópicos Especiais em LLM
**Objetivo:** Desenvolvimento de uma plataforma soberana para análise de vazios urbanos e viabilidade imobiliária EXCLUSIVA para a cidade de Uberlândia/MG.

---

## 1. Arquitetura do Modelo
Para garantir a privacidade dos dados (Soberania de Dados) e a execução em hardware convencional (CPU), optamos pelo **TinyLlama-1.1B-Chat-v1.0**. 

*   **Por que TinyLlama?** É um modelo compacto (1.1 bilhões de parâmetros) que mantém um equilíbrio excepcional entre desempenho e custo computacional, permitindo inferência em tempo real sem a necessidade obrigatória de GPUs de alta performance (A100/H100).

## 2. Técnicas de Especialização (Fine-Tuning)
Não utilizamos o modelo apenas "out-of-the-box". Implementamos **PEFT (Parameter-Efficient Fine-Tuning)** via **LoRA (Low-Rank Adaptation)**:

*   **LoRA:** Técnica que injeta camadas treináveis de baixo posto no modelo original. Isso permite que a IA aprenda a terminologia específica de Uberlândia (bairros, valores de m², fluxos de sintaxe espacial) sem a necessidade de re-treinar bilhões de parâmetros.
*   **Dataset Sintético Local:** Criamos um corpus de conhecimento focado em bairros como Santa Mônica, Granja Marileusa e Centro, permitindo que a IA entenda as nuances de cada micro-mercado de Uberlândia.

## 3. RAG vs. Fine-Tuning (Evolução do Projeto)
Durante o desenvolvimento, exploramos duas rotas principais:

1.  **RAG (Retrieval-Augmented Generation):** Utilizamos **ChromaDB** e **HuggingFace Embeddings** para indexar PDFs do Plano Diretor. O modelo consultava a lei em tempo real antes de responder.
2.  **Transição para Negócios:** Por solicitação do usuário, o foco migrou de "cumprimento legal" para "estratégia de negócio". Removemos a camada RAG para permitir que a IA fosse mais propositiva e criativa, focando em modelos de ROI e valorização imobiliária.

## 4. Otimizações de Inferência em CPU (High Performance)
Rodar LLMs em CPU é um desafio de latência. Para contornar isso, aplicamos uma stack de otimização de baixo nível:

*   **SDPA (Scaled Dot Product Attention):** Habilitamos a implementação nativa de atenção do PyTorch, que utiliza instruções vetoriais da CPU (como AVX-512) para acelerar o cálculo de pesos do Transformer.
*   **Multithreading Simétrico:** Uso de `torch.set_num_threads` sincronizado com o número de núcleos físicos para evitar gargalos de troca de contexto.
*   **Inference Streaming:** Implementação do `TextIteratorStreamer`. Isso permite a renderização progressiva de tokens no front-end, reduzindo o *Time to First Token* (TTFT) para milissegundos.
*   **Prompt Engineering de Alta Densidade:** Ajuste de `repetition_penalty=1.2` e limites de tokens (600 tokens) para garantir respostas de até 2200 caracteres com máxima eficiência energética e temporal.

## 5. Integração com Engenharia Urbana (Graph-to-Text)
O diferencial técnico deste projeto é o "Grounding" (ancoragem) em dados reais via **Graph Theory**:
*   **Pipeline OSMnx:** Extração de topologia urbana real do OpenStreetMap.
*   **Métricas de Centralidade:** A IA recebe via System Prompt os valores de **Betweenness** (Fluxo) e **Closeness** (Acesso) extraídos do grafo.
*   **Tradução Algorítmica:** A IA atua como uma interface de linguagem natural para dados complexos de engenharia, transformando métricas matemáticas em recomendações estratégicas de Real Estate.

## 6. Stack Tecnológica Detalhada
*   **Processamento Geográfico:** OSMnx, NetworkX, Folium.
*   **Deep Learning:** PyTorch, Transformers (Hugging Face), Accelerate.
*   **Especialização (PEFT):** LoRA (matriz de adaptação injetada nas camadas de atenção).
*   **Back-end & Front-end:** FastAPI (API REST), Streamlit (UI React-based).

---
**Conclusão:** O ArqGeo demonstra que a combinação de **IA Generativa local**, **Teoria dos Grafos** e **Otimização de Hardware** permite criar ferramentas de soberania tecnológica de baixo custo e alta precisão para o mercado imobiliário regional.
