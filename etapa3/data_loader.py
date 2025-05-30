# -*- coding: utf-8 -*-
"""
Data Loader para CVRP
Carga y preprocesa datos desde archivos CSV para el problema CVRP
"""

import pandas as pd
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt

class CVRPDataLoader:
    """
    Clase para cargar y preparar datos CVRP desde archivos CSV.
    """
    
    def __init__(self, data_path="datos/"):
        """
        Inicializar el cargador de datos.
        
        Args:
            data_path (str): Ruta base donde se encuentran los archivos CSV
        """
        self.data_path = data_path
        self.clients_df = None
        self.depots_df = None
        self.vehicles_df = None
        self.distance_matrix = None
        self.demands = None
        self.coordinates = None
        
    def load_case(self, case_number=1):
        """
        Cargar datos para un caso específico.
        
        Args:
            case_number (int): Número del caso (1, 2, o 3)
        """
        print(f"\nCargando datos para Caso {case_number}...")
        
        # Determinar nombres de archivos según el caso
        if case_number == 1:
            clients_file = "clients.csv"
            depots_file = "depots.csv"
            vehicles_file = "vehicles.csv"
        else:
            clients_file = f"clients_case{case_number}.csv"
            depots_file = "depots.csv"  # Siempre usar el mismo depósito
            vehicles_file = f"vehicles_case{case_number}.csv"
        
        # Cargar archivos CSV
        try:
            self.clients_df = pd.read_csv(os.path.join(self.data_path, clients_file))
            self.depots_df = pd.read_csv(os.path.join(self.data_path, depots_file))
            self.vehicles_df = pd.read_csv(os.path.join(self.data_path, vehicles_file))
            
            print(f"✓ Clientes cargados: {len(self.clients_df)} clientes")
            print(f"✓ Depósitos cargados: {len(self.depots_df)} depósitos")
            print(f"✓ Vehículos cargados: {len(self.vehicles_df)} vehículos")
            
        except FileNotFoundError as e:
            print(f"Error: No se encontró el archivo {e.filename}")
            raise
        
        # Preparar datos
        self._prepare_data()
        
        return self
    
    def _prepare_data(self):
        """
        Preparar los datos en el formato necesario para el GA.
        """
        # 1. Crear diccionario de coordenadas (depot primero, luego clientes)
        self.coordinates = {}
        
        # Agregar depósito (siempre usar el primero)
        depot_row = self.depots_df.iloc[0]
        self.coordinates[0] = (depot_row['Longitude'], depot_row['Latitude'])
        
        # Agregar clientes (con índices 1, 2, 3, ...)
        for idx, client_row in self.clients_df.iterrows():
            # El índice en la matriz será idx + 1 (porque 0 es el depósito)
            self.coordinates[idx + 1] = (client_row['Longitude'], client_row['Latitude'])
        
        # 2. Crear diccionario de demandas
        self.demands = {}
        self.demands[0] = 0  # El depósito no tiene demanda
        
        for idx, client_row in self.clients_df.iterrows():
            self.demands[idx + 1] = client_row['Demand']
        
        # 3. Calcular matriz de distancias
        n_nodes = len(self.coordinates)
        self.distance_matrix = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    self.distance_matrix[i][j] = self._calculate_distance(
                        self.coordinates[i], self.coordinates[j]
                    )
        
        print(f"✓ Matriz de distancias calculada: {n_nodes}x{n_nodes}")
        
    def _calculate_distance(self, coord1, coord2):
        """
        Calcular distancia haversine entre dos coordenadas.
        
        Args:
            coord1 (tuple): (longitud, latitud)
            coord2 (tuple): (longitud, latitud)
            
        Returns:
            float: Distancia en kilómetros
        """
        lon1, lat1 = coord1
        lon2, lat2 = coord2
        
        # Radio de la Tierra en kilómetros
        R = 6371
        
        # Convertir a radianes
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Fórmula haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return R * c
    
    def get_vehicle_capacity(self):
        """
        Obtener la capacidad del vehículo.
        Para flota homogénea, usar la capacidad promedio o la del primer vehículo.
        """
        if self.vehicles_df is not None and not self.vehicles_df.empty:
            # Para el caso base, todos los vehículos tienen la misma capacidad
            # Para otros casos, podríamos usar el promedio
            return self.vehicles_df['Capacity'].iloc[0]
        return 100  # Valor por defecto
    
    def get_num_vehicles(self):
        """
        Obtener el número de vehículos disponibles.
        """
        if self.vehicles_df is not None:
            return len(self.vehicles_df)
        return None  # Dejar que el GA calcule el número necesario
    
    def get_problem_info(self):
        """
        Obtener información resumida del problema.
        """
        info = {
            'num_customers': len(self.clients_df),
            'num_vehicles': len(self.vehicles_df),
            'total_demand': self.clients_df['Demand'].sum(),
            'vehicle_capacity': self.get_vehicle_capacity(),
            'avg_demand': self.clients_df['Demand'].mean(),
            'min_demand': self.clients_df['Demand'].min(),
            'max_demand': self.clients_df['Demand'].max()
        }
        
        # Calcular número mínimo de vehículos necesarios
        info['min_vehicles_needed'] = int(np.ceil(
            info['total_demand'] / info['vehicle_capacity']
        ))
        
        return info
    
    def print_problem_summary(self):
        """
        Imprimir resumen del problema cargado.
        """
        info = self.get_problem_info()
        
        print("\n" + "="*50)
        print("RESUMEN DEL PROBLEMA CVRP")
        print("="*50)
        print(f"Número de clientes: {info['num_customers']}")
        print(f"Número de vehículos disponibles: {info['num_vehicles']}")
        print(f"Capacidad del vehículo: {info['vehicle_capacity']}")
        print(f"\nDemanda total: {info['total_demand']}")
        print(f"Demanda promedio por cliente: {info['avg_demand']:.1f}")
        print(f"Demanda mínima: {info['min_demand']}")
        print(f"Demanda máxima: {info['max_demand']}")
        print(f"\nVehículos mínimos necesarios (teórico): {info['min_vehicles_needed']}")
        print("="*50)
    
    def get_ga_parameters(self):
        """
        Obtener parámetros preparados para el algoritmo genético.
        """
        return {
            'distance_matrix': self.distance_matrix,
            'demands': self.demands,
            'vehicle_capacity': self.get_vehicle_capacity(),
            'depot_id': 0,  # Siempre usamos índice 0 para el depósito
            'num_vehicles': self.get_num_vehicles()
        }
    
    def save_verification_file(self, solution_details, case_number, method_name="GA"):
        """
        Guardar archivo de verificación en el formato requerido.
        
        Args:
            solution_details (list): Lista de diccionarios con detalles de cada ruta
            case_number (int): Número del caso
            method_name (str): Nombre del método (para el nombre del archivo)
        """
        if not solution_details:
            print("No hay solución para guardar")
            return
        
        # Crear DataFrame
        df = pd.DataFrame(solution_details)
        
        # Asegurar el orden correcto de columnas
        columns_order = [
            'VehicleId', 'DepotId', 'InitialLoad', 'RouteSequence',
            'ClientsServed', 'DemandsSatisfied', 'TotalDistance',
            'TotalTime', 'FuelCost'
        ]
        
        # Reordenar columnas
        df = df[columns_order]
        
        # Nombre del archivo
        filename = f"verificacion_metaheuristica_{method_name}_{case_number}.csv"
        filepath = os.path.join(self.data_path, filename)
        
        # Guardar
        df.to_csv(filepath, index=False)
        print(f"\n✓ Archivo de verificación guardado: {filename}")
        
        # Mostrar preview
        print("\nPreview del archivo:")
        print(df.head())
        
        return filepath


# Función auxiliar para cargar y ejecutar un caso completo
def load_and_prepare_case(case_number=1, data_path="etapa2/scripts y archivos/"):
    """
    Función conveniente para cargar un caso completo.
    
    Args:
        case_number (int): Número del caso (1, 2, o 3)
        data_path (str): Ruta a los archivos de datos
        
    Returns:
        dict: Parámetros listos para el GA
    """
    loader = CVRPDataLoader(data_path)
    loader.load_case(case_number)
    loader.print_problem_summary()
    
    return loader.get_ga_parameters(), loader


# Código de prueba
if __name__ == "__main__":
    # Probar carga del caso 1
    print("Probando carga de datos del Caso 1...")
    
    loader = CVRPDataLoader()
    loader.load_case(1)
    loader.print_problem_summary()
    
    # Obtener parámetros para GA
    ga_params = loader.get_ga_parameters()
    print(f"\nParámetros listos para GA:")
    print(f"- Tamaño matriz de distancias: {ga_params['distance_matrix'].shape}")
    print(f"- Número de nodos con demanda: {len(ga_params['demands'])}")
    print(f"- Capacidad del vehículo: {ga_params['vehicle_capacity']}")