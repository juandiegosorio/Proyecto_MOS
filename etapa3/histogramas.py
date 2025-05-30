import pandas as pd
import matplotlib.pyplot as plt
import os

def generar_histogramas_csv(file_path, caso_num):
    """
    Genera histogramas de carga total y distancia total por vehículo
    a partir de un archivo de verificación GA.
    
    Args:
        file_path (str): Ruta al archivo CSV de verificación
        caso_num (int): Número del caso (para nombrar el archivo)
    """
    df = pd.read_csv(file_path)

    # Carga total por vehículo
    plt.figure(figsize=(10, 5))
    plt.bar(df['VehicleId'], df['InitialLoad'])
    plt.xlabel('Vehículo')
    plt.ylabel('Carga Total')
    plt.title(f'Caso {caso_num} - Carga Total por Vehículo')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'histograma_carga_caso_{caso_num}.png')
    plt.close()

    # Distancia total por vehículo
    plt.figure(figsize=(10, 5))
    plt.bar(df['VehicleId'], df['TotalDistance'])
    plt.xlabel('Vehículo')
    plt.ylabel('Distancia Total (km)')
    plt.title(f'Caso {caso_num} - Distancia Total por Vehículo')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'histograma_distancia_caso_{caso_num}.png')
    plt.close()
    
    print(f"✓ Histogramas guardados para Caso {caso_num}")

# Ejecutar para los 3 casos
generar_histogramas_csv("datos/verificacion_metaheuristica_GA_1.csv", 1)
generar_histogramas_csv("datos/verificacion_metaheuristica_GA_2.csv", 2)
generar_histogramas_csv("datos/verificacion_metaheuristica_GA_3.csv", 3)
