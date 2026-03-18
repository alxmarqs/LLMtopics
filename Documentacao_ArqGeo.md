# ArqGeo Engine: Inteligência Espacial e IA para o Mercado Imobiliário

O **ArqGeo Engine** é uma plataforma avançada de *Location Intelligence* (Inteligência de Localização) focada em análises estratégicas para expansão de negócios e mercado imobiliário. O sistema une a matemática rígida da **Sintaxe Espacial** com o poder de interpretação semântica de **Inteligência Artificial Generativa (LLMs)** rodando localmente (Edge AI).

O objetivo principal do projeto é transformar a malha viária bruta de uma cidade em um mapa de oportunidades de alto rendimento, detectando os chamados "Vazios Urbanos" — áreas com tráfego e potencial imensos, mas subutilizadas pelo mercado atual.

---

## 1. Fundamentos Teóricos e Conceitos Empregados

O coração analítico do ArqGeo não se baseia em "achismos", mas na Teoria da Sintaxe Espacial (Space Syntax), que quantifica o movimento humano e a conectividade através da estrutura da rua.

### Centralidade de Intermediação (Betweenness Centrality)
*   **O que é:** Mede quantas vezes uma rua atua como a rota "mais curta" entre todos os outros pares de ruas da cidade.
*   **Significado Comercial:** Ruas com alto *Betweenness* são os "rios de movimento" de uma cidade. Elas possuem o maior tráfego orgânico de passagem, tornando-as pontos premium para **comércio de visibilidade**, compras por impulso e conveniência (ex: postos de combustível, fast-foods com drive-thru, farmácias de grande porte).

### Centralidade de Proximidade (Closeness Centrality)
*   **O que é:** Mede o quão "perto" estruturalmente um nó (cruzamento) está de todos os outros pontos da rede em termos de alcance topológico.
*   **Significado Comercial:** Áreas com alta *Closeness* são de fácil acesso geral. São locais ideais para **equipamentos âncora e serviços de destino**, onde o cliente planeja ir e precisa chegar rápido de várias partes do bairro (ex: supermercados de bairro, escolas, clínicas médicas).

### Identificação de Vazios Urbanos
Com o casamento das métricas, o ArqGeo busca pontos focais onde os cruzamentos possuem picos dessas métricas, destacando-os no mapa. Quando um local de altíssimo fluxo não possui a infraestrutura correspondente instalada, ele é identificado como um **Vazio Crítico** (uma "mina de ouro" inexplorada para investimentos imobiliários).

---

## 2. O Papel da Inteligência Artificial (Edge AI)

Para traduzir os dados matemáticos complexos em insights acionáveis de negócios, o ArqGeo incorpora um **Motor de Viabilidade** empoderado por Inteligência Artificial.

*   **Processamento Local Contínuo:** Utilizando *Edge Computing*, o modelo roda **100% localmente na GPU/CPU** da máquina do analista. Isso garante sigilo total de dados comerciais sensíveis, zero custo com APIs externas (ex: OpenAI) e processamento resiliente offline.
*   **A Abordagem Small Language Model (SLM):** Em vez de gigantes monstruosos, o sistema escolheu ser cirúrgico. Utiliza o **TinyLlama-1.1B**, um LLM ultra-rápido projetado para rodar de forma leve, entregando análises de fluxo na tela em formato de streaming.
*   **Especialização (LoRA Finetuning):** O modelo sofreu um processo de "ensino contínuo" e ajuste fino através da tecnologia **PEFT/LoRA** (`arqgeo-llm-lora`). Ele não atua como um previsor de palavras generalista, mas simporta-se puramente como um **Consultor de Expansão de Negócios**. O bot injeta contexto geoespacial, foca restritamente no layout imobiliário fornecido, usa *Few-Shot Prompting* em engenharia de prompt, e restringe a resposta cirurgicamente ao tipo de negócio que dará lucro naquele cruzamento específico.

---

## 3. Stack Tecnológico (Core Technologies)

A arquitetura do ArqGeo Engine é baseada nas robustas ferramentas de engenharia de dados e machine learning do Python:

*   **Frontend Interativo (Streamlit):** Hospeda a interface da plataforma (Dashboard). O Streamlit foi desenhado usando um Glassmorphis customizado via CSS Injection, provendo uma estética "premium", fluida e responsiva voltada para executivos e investidores.
*   **Análise de Malha Urbana (OSMnx e NetworkX):** Estas bibliotecas processam em background os dados geoespaciais e geográficos provindos do projeto OpenStreetMap. Elas modelam as ruas como grafos complexos e calculam todas as centralidades topológicas.
*   **Visualização Georreferenciada (Folium):** Renderiza dinamicamente os resultados num mapa noturno (`CartoDB dark_matter`), destacando a sintaxe espacial via mapas de calor (heatmaps) onde os nós aquecidos (alta centralidade) ou vazios urbanos são renderizados interativamente.
*   **Machine Learning Frame (PyTorch e HuggingFace Transformers):** É a espinha dorsal de execução da IA. Carrega os tensores da IA em memória (seja VRAM de placa de vídeo CUDA via Float16, ou CPU) e coordena a geração textual token por token via as classes de `TextIteratorStreamer`.

---

## 4. Fluxo Prático do Usuário (Pipeline)

1.  **Seleção do Território:** O usuário seleciona no dashboard um bairro de Uberlândia que foi previamente mapeado nos `dados_processados` da ferramenta em formato `.graphml`.
2.  **Identificação Visual (Mapa):** O algoritmo de *backend* carrega o grafo, lê os valores pré-calculados de Centralidade, e plota imediatamente o local na interface de mapas, sinalizando em realce as esquinas de maior fluxo estrutural.
3.  **Filtragem Crítica:** O Python extrai apenas a "Nata" dos pontos, ou seja, onde a concentração de fluxo cruzado atinge a viabilidade crítica de investimento.
4.  **Injeção de IA & Streaming Textual:** O mapa de nós filtrado é traduzido em prompt e injetado na memória da rede neuro/linguística restrita do TinyLlama-1.1B. Em tempo real, a tela inicia a digitação do relatório focando na melhor estratégia de uso comercial do solo.
