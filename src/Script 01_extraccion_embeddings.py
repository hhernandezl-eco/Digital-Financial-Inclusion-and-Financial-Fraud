import os
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

def cargar_datos(ruta_pt: str, ruta_es: str) -> pd.DataFrame:
    """Carga y unifica los corpus textuales."""
    df_pt = pd.read_csv(ruta_pt)
    df_es = pd.read_csv(ruta_es)
    df_pt['lang'] = 'pt'
    df_es['lang'] = 'es'
    df_fraude = pd.concat([df_pt, df_es], ignore_index=True)
    df_fraude['fecha'] = pd.to_datetime(df_fraude['fecha'], utc=True)
    df_fraude = df_fraude.dropna(subset=['texto'])
    df_fraude['texto_limpio'] = df_fraude['texto'].astype(str).str.lower().str.strip()
    return df_fraude

def generar_embeddings(textos: pd.Series, directorio_salida: str, tamano_lote: int = 32):
    """Genera embeddings utilizando mDeBERTa-v3 y los almacena en disco."""
    modelo_nombre = 'microsoft/mdeberta-v3-base'
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizador = AutoTokenizer.from_pretrained(modelo_nombre)
    modelo = AutoModel.from_pretrained(modelo_nombre).to(dispositivo)
    
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        
    textos_lista = textos.tolist()
    total_textos = len(textos_lista)
    
    for i in tqdm(range(0, total_textos, tamano_lote), desc='Procesando embeddings'):
        lote = textos_lista[i:i+tamano_lote]
        entradas = tokenizador(lote, padding=True, truncation=True, max_length=128, return_tensors='pt').to(dispositivo)
        with torch.no_grad():
            salidas = modelo(**entradas)
        embeddings = salidas.last_hidden_state[:, 0, :].cpu().numpy()
        np.save(f'{directorio_salida}/emb_{i}.npy', embeddings)

if __name__ == '__main__':
    ruta_pt = '../data/raw/dataset_fraude_pt.csv'
    ruta_es = '../data/raw/dataset_fraude_es.csv'
    dir_embeddings = '../data/processed/embeddings'
    
    df = cargar_datos(ruta_pt, ruta_es)
    generar_embeddings(df['texto_limpio'], dir_embeddings)
