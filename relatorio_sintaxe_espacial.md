# Análise de Sintaxe Espacial: Vazios Urbanos em Santa Mônica, Uberlândia

Este documento descreve detalhadamente o processo, a metodologia e os resultados da Análise de Sintaxe Espacial realizada no bairro de Santa Mônica, Uberlândia. O objetivo principal do estudo foi identificar áreas com alto potencial de fluxo de pedestres (vitalidade comercial latente) que atualmente sofrem com baixa acessibilidade ou integração à malha principal — os chamados **"Vazios Urbanos"**.

---

## 1. Escopo e Ferramentas

Para garantir a reprodutibilidade e escalabilidade do estudo, utilizamos uma stack baseada em Python voltada para grafos e análise geoespacial de redes rua.

**Bibliotecas Utilizadas:**
- **OSMnx:** Para download da malha urbana diretamente do OpenStreetMap.
- **NetworkX:** Para os cálculos matemáticos complexos de teoria dos grafos e centralidade.
- **Folium & Matplotlib:** Para visualização geo-referenciada interativa (HTML) e geração de gráficos estáticos (PNG).
- **GeoPandas & NumPy:** Manipulação dos limites percentis e manipulação vetorial dos dados do grafo geográfico.

---

## 2. Metodologia: Malha e Obtenção de Dados

1. **Obtenção do Grafo:** Utilizando o `osmnx.graph_from_place()`, foi feito o download da malha focada no modo `walk` (pedestre) mapeando assim todas as vias caminháveis do bairro Santa Mônica.
2. **Normalização do Grafo:** Os dados foram convertidos para considerar uma rede não-direcional, sendo que velocidades de caminhada e tempos de percurso foram pré-calculados (`add_edge_travel_times`). 
3. **Conversão de Pesos:** Para o cálculo dos caminhos mínimos (shortest-path), foi utilizado o comprimento das vias (`weight='length'`) nas métricas de fluxo.

---

## 3. Cálculos de Centralidade (Network Centrality)

O núcleo da análise baseia-se em combinar dois conceitos cruciais de *Sintaxe Espacial* (Space Syntax):

### 3.1 Closeness Centrality (Integração e Acessibilidade)
O **Closeness Centrality** mede quão "próximo" (na média) um nó (esquina/interseção) está de absolutamente todos os outros nós da mesma malha urbana. 
- **O que significa na prática:** Alto Closeness significa que aquele ponto está central e integrado; é muito fácil e rápido chegar nele partindo de qualquer lugar.
- **Como calculamos:** `nx.closeness_centrality(G, distance='length')`.

### 3.2 Betweenness Centrality (Fluxo Potencial e Passagem)
O **Betweenness Centrality** penaliza os caminhos e mede quantas vezes esse nó atua como uma "ponte" essencial nos caminhos mais curtos conectando pares de outros nós espalhados pela cidade.
- **O que significa na prática:** Alto Betweenness indica um corredor poderoso, um "gargalo vivo" onde muitas pessoas *têm* que passar para chegar a seus destinos. Essas vias são artérias comerciais potentes (fluxo de pessoas).
- **Como calculamos:** `nx.betweenness_centrality(G, weight='length', normalized=True)`.

---

## 4. Definição do Computacional de "Vazio Urbano"

O objetivo crítico do usuário era achar Vazios Urbanos que têm:
> *Potencial de alta vitalidade comercial, mas baixa conectividade atual.*

Pela lógica matemática em grafos, descrevemos essa condição como a interseção de:
1. **Baixa Integração:** O local hoje está "escondido", a acessibilidade global é ruim (Baixa *Closeness Centrality*).
2. **Alto Fluxo Potencial:** Apesar de não ser integrado com a cidade como um todo, estruturalmente esse trecho conecta partes locais importantes e serviria de rota "natural" de passagem (Alta *Betweenness Centrality*).

**Processamento Estatístico:**
Para automatizar isso sem viés empírico, separamos os dados de nosso grafo usando quartis `np.percentile`:
- Foram filtrados os nós classificados entre os **25% de PIOR Acessibilidade** (Percentil `<= 25` em Closeness).
- E simultaneamente classificados entre os **25% com MAIOR Fluxo** (Percentil `>= 75` em Betweenness).

Todos os pontos na malha que responderam a essa condição dupla receberam a tag interna `vazio_urbano = 1`.

---

## 5. Visualização e Gradientes

Para que os resultados fossem perfeitamente legíveis para o tomador de decisão, a paleta visual foi programada da seguinte maneira:

1. **Os Nós Padrão (Gradient Closeness):**
   - Utilizamos um sistema de colormap onde o vermelho (`YlOrRd / Matplotlib`) aponta o alto grau de integração. 
   - Logo, áreas mais vermelhas e escuras no mapa são extremamente acessíveis. Áreas mais amarelas/claras são isoladas.
   
2. **Destaque Crítico (O Vazio Urbano):**
   - Os pontos (nós) que atendem às duas regras de Vazio Urbano foram extraídos da paleta padrão.
   - Foram explicitados no mapa com um destaque **AZUL CIANO (Cyan)** em formato redondo maior (`radius=6`). Isso garante um contraste gritante contra a paleta quentede vermelho.

---

## 6. Resultados e Entregáveis

O script processou a área com sucesso e gerou três produtos principais:

### 6.1 Gráficos Geográficos Estáticos
- `closeness_centrality.png`: O mapa geral medindo onde estão os pólos de integração avermelhados no bairro.
- `betweenness_centrality.png`: O mapa evidenciando os grandes eixos de ruas primárias.

### 6.2 Mapa Interativo Completo
- `santa_monica_sintaxe_espacial.html`: Construído utilizando *Folium*, o mapa carrega a camada Dark baseada no `CartoDB`.
- Ele plota as vias da malha do OSM. Sobre as vias, sobrepõe as bolhas (nós). 
- Ao deslizar o cursor sobre uma interseção comum, um tooltip exibe as matemáticas exatas de Fluxo e Integração.
- Ao deslizar sobre uma bolha Azul Ciano, ele alerta o usuário como "VAZIO URBANO", explicitando os porquês do cálculo estatístico ter classificado aquele ponto assim.

## Considerações Finais
As áreas em destaque, cruzando as pontas das malhas periféricas com grandes eixos (verificáveis visualmente no entorno do mapa interativo), indicam regiões que, por melhorias urbanas singelas (ciclovias rápidas, iluminação pública, recuos comerciais), teriam forte conversão de fluxo de pedestre em capital comercial, provando a tese espacial do Vazio Urbano com grande potencial latente.
