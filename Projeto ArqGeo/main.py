import os
import osmnx as ox
import networkx as nx
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# Configuracao global para ignorar warnings visuais menores
import warnings
warnings.filterwarnings("ignore")

import json

# ==========================================
# LISTA DE BAIRROS PARA PROCESSAMENTO
# ==========================================
# Carrega do arquivo json baixado anteriormente a lista de todos os bairros de Uberlândia
ARQUIVO_BAIRROS = "lista_bairros_uberlandia.json"

if os.path.exists(ARQUIVO_BAIRROS):
    with open(ARQUIVO_BAIRROS, "r", encoding='utf-8') as f:
        BAIRROS_UBERLANDIA = json.load(f)
else:
    # Fallback apenas para não dar erro se o arquivo sumir
    BAIRROS_UBERLANDIA = [
        "Santa Mônica, Uberlândia, Minas Gerais, Brazil",
        "Tibery, Uberlândia, Minas Gerais, Brazil"
    ]

PASTA_OUTPUT = "dados_processados"


def criar_pastas_se_necessario(nome_bairro):
    """Cria a estrutura de pastas para armazenar os outputs do bairro."""
    nome_limpo = nome_bairro.split(",")[0].strip().replace(" ", "_").lower()
    pasta_bairro = os.path.join(PASTA_OUTPUT, nome_limpo)
    os.makedirs(pasta_bairro, exist_ok=True)
    return pasta_bairro, nome_limpo


def obter_grafo_bairro(place_name, pasta_bairro, nome_limpo):
    """Baixa o grafo via OSMnx ou carrega do cache se ja existir."""
    arquivo_grafo = os.path.join(pasta_bairro, f"{nome_limpo}.graphml")
    
    # 1. Tentar ler do disco (Cache)
    if os.path.exists(arquivo_grafo):
        print(f"[CACHE] Carregando grafo pré-calculado para {place_name}...")
        G = ox.load_graphml(arquivo_grafo)
        return G
        
    print(f"[REDE] Baixando ruas de {place_name} (modo pedestre)...")
    try:
        G = ox.graph_from_place(place_name, network_type='walk')
    except Exception as e:
        print(f"[ERRO] Falha ao baixar {place_name} diretamente: {e}")
        return None
        
    # Converter para não-direcional, se não for
    # Adicionar fallback=30 para vias locais sem dado de velocidade na rede OSM
    try:
        G = ox.add_edge_speeds(G, fallback=30)
        G = ox.add_edge_travel_times(G)
    except Exception as e:
        print(f"[ERRO] Falha ao processar edge speeds para {place_name}: {e}")
        return None
    
    # Processamento pesado de matriz de centralidade
    print(f"    - Calculando Integacao (Closeness)...")
    closeness = nx.closeness_centrality(G, distance='length')
    nx.set_node_attributes(G, values=closeness, name='closeness')
    
    print(f"    - Calculando Fluxo Potencial (Betweenness)...")
    betweenness = nx.betweenness_centrality(G, weight='length', normalized=True)
    nx.set_node_attributes(G, values=betweenness, name='betweenness')
    
    # Encontrar vazios urbanos: Alta Betweenness (Fluxo >= 75%) e Baixa Closeness (Integração <= 25%)
    closeness_vals = list(closeness.values())
    betweenness_vals = list(betweenness.values())
    
    if not closeness_vals or not betweenness_vals:
        return None
        
    cl_25 = np.percentile(closeness_vals, 25) 
    bw_75 = np.percentile(betweenness_vals, 75)
    
    vazios_urbanos = {}
    for node, data in G.nodes(data=True):
        is_vazio = (data.get('closeness', 0) <= cl_25) and (data.get('betweenness', 0) >= bw_75)
        vazios_urbanos[node] = 1 if is_vazio else 0
        
    nx.set_node_attributes(G, values=vazios_urbanos, name='vazio_urbano')
    
    # 2. Salvar no disco (Cache)
    print(f"[SALVANDO] Armazenando grafo na pasta local {arquivo_grafo}...")
    ox.save_graphml(G, arquivo_grafo)
    
    return G


def exportar_visualizacoes(G, pasta_bairro, nome_limpo):
    """Gera o mapa interativo HTML e os gráficos estáticos do bairro"""
    
    # Verifica se o HTML ja existe para pular (economizar processamento)
    output_html = os.path.join(pasta_bairro, f"{nome_limpo}_mapa.html")
    if os.path.exists(output_html):
        print(f"[CACHE] Visualizacoes de {nome_limpo} ja existem, pulando exportacao visual...")
        return
        
    print(f"[VISUALIZACAO] Gerando mapas HTML e PNG para {nome_limpo}...")
    
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    if nodes.empty or edges.empty:
        return
        
    center_lat = nodes.geometry.y.mean()
    center_lon = nodes.geometry.x.mean()
    
    mapa = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='CartoDB dark_matter')
    
    # Extrair valores pros gradientes
    cl_vals = [data.get('closeness', 0) for _, data in G.nodes(data=True)]
    bw_vals = [data.get('betweenness', 0) for _, data in G.nodes(data=True)]
    
    norm_closeness = mcolors.Normalize(vmin=min(cl_vals), vmax=max(cl_vals))
    cmap_closeness = cm.get_cmap('YlOrRd')
    norm_betweenness = mcolors.Normalize(vmin=min(bw_vals), vmax=max(bw_vals))
    cmap_betweenness = cm.get_cmap('plasma')
    
    # 1. Folium (Mapa Web)
    edges_geojson = folium.GeoJson(
        edges,
        style_function=lambda x: {'color': '#333333', 'weight': 1}
    ).add_to(mapa)

    for index, row in nodes.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        c_val = row.get('closeness', 0)
        b_val = row.get('betweenness', 0)
        is_vazio = row.get('vazio_urbano', 0)
        
        rgba = cmap_closeness(norm_closeness(c_val))
        hex_color_closeness = mcolors.to_hex(rgba)
        
        if is_vazio == 1:
            folium.CircleMarker(
                location=[lat, lon], radius=6, color='cyan', fill=True, fill_color='blue', fill_opacity=0.9,
                tooltip=f"VAZIO URBANO<br>Betweenness: {b_val:.5f} (Alta)<br>Closeness: {c_val:.5f} (Baixa)"
            ).add_to(mapa)
        else:
            folium.CircleMarker(
                location=[lat, lon], radius=3, color=hex_color_closeness, weight=1, fill=True, fill_color=hex_color_closeness, fill_opacity=0.6,
                tooltip=f"Integração: {c_val:.5f}<br>Fluxo: {b_val:.5f}"
            ).add_to(mapa)
            
    mapa.save(output_html)
    
    # 2. Mapas Estáticos (Matplotlib)
    try:
        # Closeness
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='k')
        nc = [cmap_closeness(norm_closeness(G.nodes[n].get('closeness', 0))) for n in G.nodes()]
        ox.plot_graph(G, ax=ax, node_color=nc, node_size=15, edge_linewidth=0.5, bgcolor='k', show=False)
        ax.set_title(f"{nome_limpo.upper()} - Closeness (Integração)", color='w')
        fig.savefig(os.path.join(pasta_bairro, f"{nome_limpo}_closeness.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Betweenness
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='k')
        nb = [cmap_betweenness(norm_betweenness(G.nodes[n].get('betweenness', 0))) for n in G.nodes()]
        ox.plot_graph(G, ax=ax, node_color=nb, node_size=15, edge_linewidth=0.5, bgcolor='k', show=False)
        ax.set_title(f"{nome_limpo.upper()} - Betweenness (Fluxo/Passagem)", color='w')
        fig.savefig(os.path.join(pasta_bairro, f"{nome_limpo}_betweenness.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[AVISO] Erro na geração PNG estática para {nome_limpo}: {e}")


def processar_cidade():
    print(f"=== INICIANDO MOTOR DE ANÁLISE DE SINTAXE ESPACIAL EM LOTE ===")
    os.makedirs(PASTA_OUTPUT, exist_ok=True)
    
    sucesso = 0
    falha = 0
    
    for nome_bairro in BAIRROS_UBERLANDIA:
        print(f"\n--- Processando: {nome_bairro} ---")
        pasta_bairro, nome_limpo = criar_pastas_se_necessario(nome_bairro)
        
        grafo = obter_grafo_bairro(nome_bairro, pasta_bairro, nome_limpo)
        
        if grafo is not None:
            exportar_visualizacoes(grafo, pasta_bairro, nome_limpo)
            sucesso += 1
            print(f"[OK] {nome_bairro} concluído.")
        else:
            falha += 1
            print(f"[FALHA] Não foi possível completar processo para {nome_bairro}.")
            
    print(f"\n=== RESUMO DO PROCESSAMENTO ===")
    print(f"Total Sucesso: {sucesso}")
    print(f"Total Falhas : {falha}")


if __name__ == "__main__":
    processar_cidade()
