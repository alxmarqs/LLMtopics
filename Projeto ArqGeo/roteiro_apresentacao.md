# 🎙️ Roteiro de Apresentação: ArqGeo Engine (Uberlândia Special Edition)

## Parte 1: O Pilar Técnico (Arquitetura e Engenharia de IA)
1. **Engine de Dados Geométricos (OSMnx & NetworkX):**
   - Extração automática de malhas urbanas via APIs do **OpenStreetMap**.
   - Modelagem da cidade como um **Grafo Direcionado**, onde ruas são arestas e cruzamentos são nós.
2. **Algoritmos de Sintaxe Espacial:**
   - Cálculo de **Betweenness Centrality**: Identifica os eixos de maior fluxo potencial (as "artérias" da cidade).
   - Cálculo de **Closeness Centrality**: Mede a eficiência de acesso a serviços e centralidades.
3. **Cérebro de IA Local (TinyLlama 1.1B):**
   - **PEFT / LoRA (Low-Rank Adaptation):** Técnica de ajuste fino que treina apenas uma fração dos parâmetros (matrizes de baixo posto), especializando o modelo no vocabulário imobiliário de Uberlândia sem exigir super-Gpus.
   - **Quantização & Precisão:** Otimização para rodar em FP32 (CPU) com performance de alto nível.
4. **Otimização de Inferência (Performance Real-Time):**
   - **SDPA (Scaled Dot Product Attention):** Implementação nativa do PyTorch que acelera o cálculo de atenção de transformadores em CPUs.
   - **Texto por Streaming:** Uso de `TextIteratorStreamer` e multithreading para renderização progressiva no Dashboard.
5. **Grounding Geométrico:** O modelo não apenas escreve; ele recebe as métricas matemáticas dos grafos como contexto (*System Prompting*), garantindo que o laudo seja baseado em lógica espacial real.

## 🧠 O Prompt de Engenharia (A "Receita" da IA)
Para garantir que a IA não "alucina", enviamos um comando estruturado (*Prompt*) que combina regras de negócio e dados geográficos brutos:

**System Prompt (As Regras):**
> "Você é o ArqGeo-GPT. RESPONDA APENAS NO FORMATO ABAIXO... Use [PADRÃO MERCADO UBERLÂNDIA]. ESTRUTURA OBRIGATÓRIA: PONTO, NEGÓCIO, VALOR M2 e PECULIARIDADE."

**User Prompt (Os Dados):**
> "Análise (Uberlândia) - Bairro {bairro_nome}:
> - Ponto 1 (-18.91, -48.27): Fluxo X, Acesso Y...
> Responda agora no formato solicitado."

## Parte 2: O Pilar de Negócio (Valor Imobiliário e Estratégia)
1. **Identificação de Vazios Urbanos Críticos:** Localização algorítmica de terrenos subutilizados em áreas de alta vitalidade.
2. **Estimativa de VGV e Viabilidade:** Projeção de potencial de venda e modelos de negócio (Residencial vertical, Hub Comercial, etc.).
3. **Soberania Regional:** IA calibrada para o mercado de Uberlândia/MG, superando modelos genéricos (como ChatGPT puro) em precisão local.
4. **Privacidade e Custo Zero:** Diferencial de rodar 100% offline, protegendo estratégias comerciais e eliminando custos de tokens externos.

## Script de Demo (Live)
- **Seleção:** Escolha de um território (ex: Granja Marileusa).
- **Mapa:** Visualização dos fluxos (Vermelho = Calor de Fluxo | Azul = Oportunidade).
- **IA em Tempo Real:** Clique no botão e mostre a IA gerando o laudo técnico por streaming, demonstrando a velocidade e a profundidade da análise.
