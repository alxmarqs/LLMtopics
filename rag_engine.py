import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PASTA_DOCS = "docs/legislacao"
PASTA_CHROMA = "chroma_db"

def criar_banco_vetorial():
    """
    Lê uma pasta cheia de PDFs, extrai o texto, quebra em 'chunks' (pedaços) 
    e salva os chunks como Embeddings Matematicos no banco local ChromaDB.
    """
    print(f"=== INICIANDO CRIAÇÃO DO BANCO RAG 'ARQGEO' ===")
    
    # 1. Checa se o usuário alimentou a pasta com as leis de Uberlândia
    if not os.path.exists(PASTA_DOCS):
        os.makedirs(PASTA_DOCS)
        print(f"ATENCAO: Pasta '{PASTA_DOCS}' não existia e foi criada.")
        print(f"Por favor, solte seus PDFs do Plano Diretor lá dentro e rode este script novamente.")
        return
        
    arquivos = os.listdir(PASTA_DOCS)
    if not arquivos:
        print(f"ERRO: Nenhum arquivo encontrado em '{PASTA_DOCS}'.")
        print("Solte seus arquivos PDF lá dentro antes de treinar a IA.")
        return
        
    print(f"1. Carregando documentos da pasta {PASTA_DOCS}...")
    loader = PyPDFDirectoryLoader(PASTA_DOCS)
    docs = loader.load()
    print(f"-> {len(docs)} páginas/documentos lidos com sucesso.")
    
    # 2. Quebrar os textos em pedaços menores (Chunks). 
    # Leis tem artigos e paragrafos, precisamos manter um certo contexto visual (overlap).
    print("2. Quebrando as leis em chunks contextualizados...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", "Art.", "§", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"-> O material foi despedaçado em {len(chunks)} blocos de texto independentes.")

    # 3. Modelos de Embedding (Traduz texto em números). 
    # Estamos usando um open-source gratuito para funcionar offline na nossa máquina (HuggingFace)
    print("3. Inicializando modelo de 'compreensão matemática de texto' (Embeddings)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Criar ou Atualizar Banco
    print(f"4. Indexando TUDO no banco de dados ChromaDB na pasta local '{PASTA_CHROMA}'...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=PASTA_CHROMA
    )
    
    print("=== SUCESSO! RAG TREINADO! ===")
    print(f"Agora o banco está pronto para ser consultado pelo 'app.py' no Streamlit.")


if __name__ == "__main__":
    criar_banco_vetorial()
