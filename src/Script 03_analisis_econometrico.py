import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

def preparar_panel(ruta_frisk: str, ruta_ifi: str) -> pd.DataFrame:
    """Prepara el conjunto de datos de panel fusionando las variables de interés."""
    df_frisk = pd.read_csv(ruta_frisk)
    df_ifi = pd.read_csv(ruta_ifi)
    
    df_panel = pd.merge(df_frisk, df_ifi, on=['country', 'year', 'month'], how='inner')
    df_panel = df_panel.set_index(['country', 'time_index']) 
    return df_panel

def estimar_modelo_efectos_fijos(df_panel: pd.DataFrame, ruta_resultados: str):
    """Estima el modelo de regresión de panel con efectos fijos bidireccionales."""
    variable_dependiente = df_panel['IFI']
    variables_independientes = df_panel[['std_ln_FRISK', 'unemployment_rate', 'inflation_index', 'conexiones_moviles']]
    variables_independientes = sm.add_constant(variables_independientes)
    
    modelo = PanelOLS(variable_dependiente, variables_independientes, entity_effects=True, time_effects=True)
    resultados = modelo.fit(cov_type='clustered', cluster_entity=True)
    
    with open(ruta_resultados, 'w') as archivo:
        archivo.write(resultados.summary.as_text())

if __name__ == '__main__':
    ruta_frisk = '../data/processed/dataset_FRISK_panel.csv'
    ruta_ifi = '../data/raw/IFI_Final.csv'
    ruta_resultados = '../results/tablas_regresion.txt'
    
    panel_datos = preparar_panel(ruta_frisk, ruta_ifi)
    estimar_modelo_efectos_fijos(panel_datos, ruta_resultados)
