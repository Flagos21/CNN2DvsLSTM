import pandas as pd
import numpy as np

# Ruta del archivo
file_path = r'C:/Users/flago/Python/Modelos/minidatasetGrande.csv'

# Cargar el dataset
data = pd.read_csv(file_path)

# Eliminar columnas no necesarias para el preprocesamiento (en este caso, ID)
data = data.drop(columns=["Unnamed: 0"], errors='ignore')  # Ajusta según el nombre exacto de la columna ID

# Definir límites normales para cada variable (basado en conocimiento del dominio o estadísticas)
limits = {
    "Potencia": (-3000, 10),  # Rango esperado para Potencia
    "Radiacion": (-10, 1300), # Rango esperado para Radiación (incluye valores negativos)
    "Temperatura": (-10, 50), # Rango plausible para temperatura ambiente
    "Temperatura panel": (-10, 60)  # Temperatura de panel esperada
}

# Corregir valores fuera de los límites
for column, (lower, upper) in limits.items():
    data[column] = np.where(data[column] < lower, lower, data[column])
    data[column] = np.where(data[column] > upper, upper, data[column])

# Rellenar valores faltantes con la mediana (si existieran)
for column in data.columns:
    if data[column].isnull().any():
        data[column].fillna(data[column].median(), inplace=True)

# Mostrar el dataset procesado
print("Dataset procesado (primeras filas):")
print(data.head())

# Guardar el dataset procesado
output_path = r'C:/Users/flago/Python/Modelos/preprocessed_data.csv'
data.to_csv(output_path, index=False)
print(f"El dataset preprocesado se ha guardado en '{output_path}'.")
