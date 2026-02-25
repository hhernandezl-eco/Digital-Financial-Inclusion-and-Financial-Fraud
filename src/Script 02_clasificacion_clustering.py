import os
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def reducir_dimensionalidad(dir_embeddings: str, componentes: int = 50) -> np.ndarray:
    """Aplica PCA incremental sobre los embeddings guardados."""
    ipca = IncrementalPCA(n_components=componentes)
    archivos = sorted([f for f in os.listdir(dir_embeddings) if f.endswith('.npy')])
    
    for archivo in archivos:
        lote = np.load(os.path.join(dir_embeddings, archivo))
        ipca.partial_fit(lote)
        
    lista_reducida = [ipca.transform(np.load(os.path.join(dir_embeddings, f))).astype(np.float32) for f in archivos]
    return np.vstack(lista_reducida)

def agrupar_dbscan(matriz_caracteristicas: np.ndarray, tamano_muestra: int = 50000) -> np.ndarray:
    """Aplica agrupamiento espacial basado en densidad con ruido."""
    escalador = StandardScaler()
    matriz_escalada = escalador.fit_transform(matriz_caracteristicas)
    
    indices_muestra = np.random.choice(len(matriz_escalada), min(tamano_muestra, len(matriz_escalada)), replace=False)
    muestra = matriz_escalada[indices_muestra]
    
    modelo_dbscan = DBSCAN(eps=0.5, min_samples=10, metric='euclidean', n_jobs=-1)
    etiquetas_muestra = modelo_dbscan.fit_predict(muestra)
    
    clasificador_knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    clasificador_knn.fit(muestra, etiquetas_muestra)
    
    etiquetas_finales = clasificador_knn.predict(matriz_escalada)
    return etiquetas_finales

if __name__ == '__main__':
    dir_embeddings = '../data/processed/embeddings'
    ruta_datos_originales = '../data/raw/dataset_fraude_es.csv' 
    ruta_salida = '../data/processed/dataset_fraude_clasificado.csv'
    
    matriz_reducida = reducir_dimensionalidad(dir_embeddings)
    etiquetas = agrupar_dbscan(matriz_reducida)
    
    df_resultados = pd.DataFrame({'cluster': etiquetas})
    df_resultados.to_csv(ruta_salida, index=False)
