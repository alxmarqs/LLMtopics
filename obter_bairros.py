import osmnx as ox
import json

def fetch_bairros_uberlandia():
    print("Iniciando query ao Overpass API (OSMnx) para limites de bairros...")
    place = "Uberlândia, Minas Gerais, Brazil"
    
    # Busca todas as feicoes do OSM que representam "suburbs" (bairros) de Uberlandia
    tags = {'place': ['suburb', 'neighbourhood']}
    
    try:
        bairros_gdf = ox.features_from_place(place, tags=tags)
        
        # Filtrar o nome
        nomes_bairros = bairros_gdf['name'].dropna().unique().tolist()
        
        # Formatar pro padrão do nosso motor de busca: "Nome_Bairro, Uberlândia, Minas Gerais, Brazil"
        lista_formatada = [f"{nome}, Uberlândia, Minas Gerais, Brazil" for nome in nomes_bairros]
        lista_formatada.sort()
        
        print(f"Foram encontrados {len(lista_formatada)} bairros!")
        
        with open("lista_bairros_uberlandia.json", "w", encoding='utf-8') as f:
            json.dump(lista_formatada, f, ensure_ascii=False, indent=4)
            
        print("Salvo em 'lista_bairros_uberlandia.json'.")
        
    except Exception as e:
        print(f"Erro ao buscar: {e}")

if __name__ == "__main__":
    fetch_bairros_uberlandia()
