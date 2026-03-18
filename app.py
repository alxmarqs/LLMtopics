import streamlit as st
import os
import networkx as nx
import osmnx as ox
import folium
from streamlit_folium import folium_static
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
from transformers import TextIteratorStreamer
import threading
import time

# Detecção de Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[HARDWARE] Dispositivo selecionado: {DEVICE}")

# Carrega variaveis de ambiente (API Keys)
load_dotenv()

# Configuracoes Iniciais da Pagina
st.set_page_config(page_title="ArqGeo | Inteligência Urbana", layout="wide", page_icon="🏙️")

# --- DESIGN SYSTEM & CSS INJECTION ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@100;300;400;600;800&family=Inter:wght@300;400;500;700&display=swap');

    :root {
        --subtle-bg: #0b0e14;
        --card-bg: rgba(23, 28, 40, 0.7);
        --accent: #00d2ff;
        --secondary: #3a7bd5;
        --text: #f0f2f6;
        --border: rgba(255, 255, 255, 0.08);
    }

    /* Animação Global */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

    .stApp {
        background: radial-gradient(circle at top right, #161d31 0%, #0b0e14 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Esconde barra lateral padrão e limpa o topo */
    header {visibility: hidden;}
    .block-container {padding-top: 2rem !important;}

    /* Typography */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        background: linear-gradient(135deg, #ffffff 0%, #a0a0a0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2px;
    }

    .premium-header {
        font-size: 4rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px !important;
    }

    /* Glass Cards Evolution */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        animation: fadeIn 0.8s ease-out;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .glass-card:hover {
        border-color: rgba(0, 210, 255, 0.3);
        box-shadow: 0 25px 50px rgba(0, 210, 255, 0.1);
        transform: translateY(-5px);
    }

    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2rem !important;
        border-radius: 100px !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        box-shadow: 0 10px 20px rgba(0, 210, 255, 0.2) !important;
        width: 100% !important;
    }

    .stButton>button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 15px 30px rgba(0, 210, 255, 0.4) !important;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--subtle-bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

    /* Report Section */
    .report-text {
        color: #d1d5db;
        font-size: 1.15rem;
        line-height: 1.8;
        background: rgba(0,0,0,0.2);
        padding: 20px;
        border-radius: 15px;
        border-left: 4px solid var(--accent);
    }

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {
        background-color: #080a0e !important;
        border-right: 1px solid var(--border) !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1 {
        background: white;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

PASTA_OUTPUT = "dados_processados"

@st.cache_data
def listar_bairros_processados():
    """Lê as pastas locais e retorna apenas os bairros que já tem o arquivo .graphml pronto."""
    if not os.path.exists(PASTA_OUTPUT):
        return []
        
    bairros = []
    for pasta in os.listdir(PASTA_OUTPUT):
        caminho_pasta = os.path.join(PASTA_OUTPUT, pasta)
        if os.path.isdir(caminho_pasta):
            arquivo_grafo = os.path.join(caminho_pasta, f"{pasta}.graphml")
            if os.path.exists(arquivo_grafo):
                bairros.append(pasta)
    return sorted(bairros)

@st.cache_resource
def carregar_llm_local():
    """Carrega o cérebro da IA na GPU (float16) ou CPU como fallback."""
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_id = "arqgeo-llm-lora"
    
    print(f"[IA LOCAL] Carregando Tokenizer e Modelo: {model_id} no dispositivo {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Carrega modelo em float16 na GPU (metade da VRAM) ou float32 na CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    )
    
    # Tenta carregar o treinamento personalizado (LoRA) se ele já existir
    if os.path.exists(lora_id):
        print(f"[IA LOCAL] Injetando especialização urbanística encontrada em '{lora_id}'...")
        model = PeftModel.from_pretrained(model, lora_id)
        
    model.eval()
    return tokenizer, model

@st.cache_resource
def carregar_grafo_bairro(nome_pasta_bairro):
    """Carrega o grafo na memória rapidamente."""
    caminho = os.path.join(PASTA_OUTPUT, nome_pasta_bairro, f"{nome_pasta_bairro}.graphml")
    return ox.load_graphml(caminho)

def solicitar_analise_llm_stream(lista_top_nodes, bairro_nome):
    """Gera o laudo com STREAMING — Ultra-Restrito e Direto."""
    tokenizer, model = carregar_llm_local()
    
    nodos_str = ""
    for i, nodo in enumerate(lista_top_nodes):
        rua = nodo.get('rua', 'N/I')
        nodos_str += f"- {rua} (Fluxo: {nodo['b_val']:.4f}, Acesso: {nodo['c_val']:.4f})\n"

    # Prompt com Exemplo (1-Shot) focado EXCLUSIVAMENTE em Vocações de Negócios
    prompt = (
        f"<|system|>\n"
        "Você é um Consultor de Expansão de Negócios em Uberlândia, Minas Gerais.\n"
        "Você DEVE responder EXCLUSIVAMENTE em Português do Brasil (pt-BR).\n"
        "Análise estritamente a vocação comercial de cada local com base no Fluxo e Acesso.\n"
        "Regra: Retorne APENAS estas 2 linhas para cada local e NADA MAIS:\n"
        "LOCAL: [Nome]\n"
        "NEGÓCIOS IDEAIS: [Sua análise curada de 2 a 4 empresas/tipos de comércio que fariam sucesso aí]\n\n"
        "Exemplo de Saída Esperada:\n"
        "LOCAL: Avenida João Naves de Ávila (Fluxo Alto)\n"
        "NEGÓCIOS IDEAIS: Franquias de Fast Food (Drive-thru), Farmácias 24h, Supermercados Express e Postos de Combustível devido ao altíssimo tráfego de passagem.\n"
        "</s>\n"
        f"<|user|>\n"
        f"Analise o bairro {bairro_nome} em Uberlândia-MG.\n"
        f"Gere o relatório EM PORTUGUÊS com foco estrito em NEGÓCIOS para estes pontos:\n"
        f"{nodos_str}"
        f"</s>\n"
        f"<|assistant|>\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    thread = threading.Thread(target=model.generate, kwargs={
        **inputs,
        "max_new_tokens": 2000, 
        "temperature": 0.2, 
        "top_k": 40,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
        "streamer": streamer,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    })
    thread.start()
    return streamer

def gerar_mapa_folium(G):
    """Gera o objeto mapa dinamicamente para exibição no Streamlit"""
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    center_lat = nodes.geometry.y.mean()
    center_lon = nodes.geometry.x.mean()
    
    mapa = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='CartoDB dark_matter')
    
    # 0. Sanitização Absoluta de Tipos
    cl_vals = []
    bw_vals = []
    
    for _, dado in G.nodes(data=True):
        try:
            cl = float(dado.get('closeness', 0.0))
        except (ValueError, TypeError):
            cl = 0.0
            
        try:
            bw = float(dado.get('betweenness', 0.0))
        except (ValueError, TypeError):
            bw = 0.0
            
        cl_vals.append(cl)
        bw_vals.append(bw)
    
    if not cl_vals: cl_vals = [0.0]
    if not bw_vals: bw_vals = [0.0]
    
    norm_closeness = mcolors.Normalize(vmin=min(cl_vals), vmax=max(cl_vals))
    cmap_closeness = cm.get_cmap('YlOrRd')
    
    # 1. Plotar Arestas
    folium.GeoJson(
        edges,
        style_function=lambda x: {'color': '#333333', 'weight': 1}
    ).add_to(mapa)

    # 2. Plotar Nós
    for index, row in nodes.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        
        try:
            c_val = float(row.get('closeness', 0.0))
        except (ValueError, TypeError):
            c_val = 0.0
            
        try:
            b_val = float(row.get('betweenness', 0.0))
        except (ValueError, TypeError):
            b_val = 0.0
            
        is_vazio = str(row.get('vazio_urbano', 0))
        
        rgba = cmap_closeness(norm_closeness(c_val))
        hex_color_closeness = mcolors.to_hex(rgba)
        
        if is_vazio in ["1", "1.0", "True"]:
            folium.CircleMarker(
                location=[lat, lon], radius=6, color='cyan', fill=True, fill_color='blue', fill_opacity=0.9,
                tooltip=f"VAZIO URBANO<br>Fluxo: {b_val:.5f}<br>Acessibilidade: {c_val:.5f}"
            ).add_to(mapa)
        else:
            folium.CircleMarker(
                location=[lat, lon], radius=3, color=hex_color_closeness, weight=0, fill=True, fill_color=hex_color_closeness, fill_opacity=0.6,
                tooltip=f"Acessibilidade: {c_val:.5f}</br>Fluxo: {b_val:.5f}"
            ).add_to(mapa)
            
    return mapa


# ==========================================
# INTERFACE STREAMLIT
# ==========================================

# Sidebar de Configurações
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2099/2099192.png", width=80)
    st.title("ArqGeo Engine")
    st.markdown("---")
    st.markdown("### Status do Sistema")
    gpu_status = "GPU ⚡" if torch.cuda.is_available() else "CPU 🐢"
    st.success(f"IA Local: Online ({gpu_status})")
    st.info("Dataset: v2.5 (Real Estate)")
    st.markdown("---")
    st.warning("Otimizado para Uberlândia")

# Main Layout
st.markdown('<h1 class="premium-header">ArqGeo<span style="color:white; font-weight:300">Engine</span></h1>', unsafe_allow_html=True)

# SELETOR DE BAIRRO PROEMINENTE
bairros_disponiveis = listar_bairros_processados()
if not bairros_disponiveis:
    st.error("Nenhum dado encontrado.")
    bairro_selecionado = None
else:
    st.markdown('<div class="glass-card" style="padding: 1.5rem; margin-top: -1rem; margin-bottom: 1rem;">', unsafe_allow_html=True)
    col_desc, col_sel = st.columns([2, 1])
    with col_desc:
        st.markdown("""
            <p style="font-size: 1.1rem; margin-bottom: 0; opacity: 0.8;">
                Plataforma Soberana de <b>Inteligência Imobiliária</b>. Selecione o território para análise dinâmica de valor e fluxo.
            </p>
        """, unsafe_allow_html=True)
    with col_sel:
        bairro_selecionado = st.selectbox("📍 TERRITÓRIO EM FOCO:", bairros_disponiveis, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)


if bairro_selecionado:
    G = carregar_grafo_bairro(bairro_selecionado)
    
    col_mapa, col_relatorio = st.columns([1.8, 1.2], gap="large")
    
    with col_mapa:
        st.markdown(f"### 📍 Análise de Fluxo: {bairro_selecionado.upper()}")
        
        mapa = gerar_mapa_folium(G)
        folium_static(mapa, width=800, height=580)
        
        st.markdown("""
        <div style="display: flex; gap: 20px; margin-top: 1rem; opacity: 0.7; font-size: 0.9rem;">
            <div>🔴 <span style="color:#ff4b4b">Alta Entrecentralidade</span></div>
            <div>🔵 <span style="color:#00d2ff">Vazio Crítico</span></div>
            <div>⚪ <span style="color:#333">Malha Base</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_relatorio:
        st.markdown('### 🧠 Insights Financeiros')
        
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("""
                <h4 style="margin-top:0">Motor de Viabilidade</h4>
                <p style="opacity:0.7; font-size: 0.9rem;">O modelo TinyLlama irá processar os fluxos de rede para estimar m² e modelos de negócio rentáveis.</p>
            """, unsafe_allow_html=True)
            
            if st.button("💎 CALCULAR VGV E VIABILIDADE"):
                with st.spinner("Analisando padrões de mercado..."):
                    # 1. Achar tds os vazios urbanos
                    vazios = [(n, d) for n, d in G.nodes(data=True) if str(d.get('vazio_urbano', 0)) in ["1", "1.0", "True"]]
                    vazios.sort(key=lambda x: x[1].get('betweenness', 0), reverse=True)
                    top5 = vazios[:3] # Focar no top 3 para laudos mais densos
                    
                    lista_nodos = []
                    for n, data in top5:
                        ruas_conectadas = []
                        for u, v, k, d_aresta in G.edges([n], keys=True, data=True):
                            if 'name' in d_aresta:
                                if isinstance(d_aresta['name'], list): ruas_conectadas.extend(d_aresta['name'])
                                else: ruas_conectadas.append(d_aresta['name'])
                        
                        nome_rua = ", ".join(set(ruas_conectadas)) if ruas_conectadas else "Eixo n/ ident."
                        lista_nodos.append({
                            'lat': float(data.get('y', 0.0)), 
                            'lon': float(data.get('x', 0.0)),
                            'b_val': float(data.get('betweenness', 0.0)),
                            'c_val': float(data.get('closeness', 0.0)),
                            'rua': nome_rua
                        })
                    
                    if not lista_nodos:
                        st.info("Nenhum dado financeiro crítico encontrado.")
                    else:
                        st.markdown("---")
                        # Geração Fluida em Tempo Real
                        stream = solicitar_analise_llm_stream(lista_nodos, bairro_selecionado)
                        st.write_stream(stream)
            st.markdown('</div>', unsafe_allow_html=True)
