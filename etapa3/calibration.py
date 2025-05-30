# calibration.py
"""
Script de calibración de parámetros para el GA CVRP
"""

import numpy as np
from ga_cvrp import GeneticAlgorithmCVRP
from data_loader import CVRPDataLoader
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import time

class GACalibration:
    """
    Clase para calibrar parámetros del GA
    """
    
    def __init__(self, case_number=1):
        self.case_number = case_number
        self.loader = CVRPDataLoader()
        self.loader.load_case(case_number)
        self.ga_params = self.loader.get_ga_parameters()
        self.results = []
        
    def run_single_experiment(self, params, num_runs=3, verbose=False):
        """
        Ejecutar un experimento con parámetros específicos
        """
        results = []
        
        for run in range(num_runs):
            if verbose:
                print(f"  Run {run+1}/{num_runs}...", end="")
            
            # Crear GA con parámetros específicos
            ga = GeneticAlgorithmCVRP(
                distance_matrix=self.ga_params['distance_matrix'],
                demands=self.ga_params['demands'],
                vehicle_capacity=self.ga_params['vehicle_capacity'],
                depot_id=self.ga_params['depot_id'],
                num_vehicles=self.ga_params['num_vehicles'],
                population_size=params['population_size'],
                generations=params['generations'],
                mutation_rate=params['mutation_rate'],
                crossover_rate=params['crossover_rate'],
                elitism_rate=params['elitism_rate'],
                tournament_size=params['tournament_size']
            )
            
            # Resolver
            start_time = time.time()
            best_solution, best_fitness = ga.solve(verbose=False)
            end_time = time.time()
            
            # Calcular métricas
            num_vehicles_used = len([r for r in best_solution if r])
            total_distance = sum(ga._calculate_route_distance(r) for r in best_solution if r)
            
            results.append({
                'fitness': best_fitness,
                'num_vehicles': num_vehicles_used,
                'total_distance': total_distance,
                'execution_time': end_time - start_time,
                'generations_executed': len(ga.fitness_history)
            })
            
            if verbose:
                print(f" Fitness: {best_fitness:.2f}")
        
        # Calcular estadísticas
        fitness_values = [r['fitness'] for r in results]
        
        return {
            'params': params,
            'avg_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'min_fitness': np.min(fitness_values),
            'max_fitness': np.max(fitness_values),
            'avg_vehicles': np.mean([r['num_vehicles'] for r in results]),
            'avg_distance': np.mean([r['total_distance'] for r in results]),
            'avg_time': np.mean([r['execution_time'] for r in results]),
            'results': results
        }
    
    def grid_search(self, param_grid, num_runs=3):
        """
        Búsqueda en grilla de parámetros
        """
        # Generar todas las combinaciones
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        print(f"Total de combinaciones a probar: {len(param_combinations)}")
        print(f"Corridas por combinación: {num_runs}")
        print(f"Total de ejecuciones: {len(param_combinations) * num_runs}")
        print("-" * 50)
        
        results = []
        
        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))
            
            print(f"\nExperimento {i+1}/{len(param_combinations)}")
            print(f"Parámetros: {params}")
            
            result = self.run_single_experiment(params, num_runs, verbose=True)
            results.append(result)
            
            print(f"  Fitness promedio: {result['avg_fitness']:.2f} ± {result['std_fitness']:.2f}")
        
        self.results = results
        return results
    
    def analyze_results(self):
        """
        Analizar y visualizar resultados de calibración
        """
        if not self.results:
            print("No hay resultados para analizar")
            return
        
        # Convertir a DataFrame para análisis
        df_results = []
        for result in self.results:
            row = result['params'].copy()
            row.update({
                'avg_fitness': result['avg_fitness'],
                'std_fitness': result['std_fitness'],
                'min_fitness': result['min_fitness'],
                'avg_vehicles': result['avg_vehicles'],
                'avg_distance': result['avg_distance'],
                'avg_time': result['avg_time']
            })
            df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # Encontrar mejores parámetros
        best_idx = df['avg_fitness'].idxmin()
        best_params = df.iloc[best_idx]
        
        print("\n" + "="*60)
        print("MEJORES PARÁMETROS ENCONTRADOS")
        print("="*60)
        print(f"Fitness promedio: {best_params['avg_fitness']:.2f}")
        print(f"Desviación estándar: {best_params['std_fitness']:.2f}")
        print(f"Vehículos promedio: {best_params['avg_vehicles']:.1f}")
        print(f"Distancia promedio: {best_params['avg_distance']:.2f} km")
        print(f"Tiempo promedio: {best_params['avg_time']:.2f} s")
        print("\nParámetros:")
        for param in self.results[0]['params'].keys():
            print(f"  {param}: {best_params[param]}")
        
        # Visualizaciones
        self._plot_parameter_effects(df)
        
        return df
    
    def _plot_parameter_effects(self, df):
        """
        Graficar efectos de cada parámetro
        """
        params = list(self.results[0]['params'].keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, param in enumerate(params):
            if i < len(axes):
                ax = axes[i]
                
                # Agrupar por valor del parámetro
                grouped = df.groupby(param)['avg_fitness'].agg(['mean', 'std'])
                
                # Graficar
                x = grouped.index
                y = grouped['mean']
                yerr = grouped['std']
                
                ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5)
                ax.set_xlabel(param)
                ax.set_ylabel('Fitness Promedio')
                ax.set_title(f'Efecto de {param}')
                ax.grid(True, alpha=0.3)
        
        # Ocultar ejes no usados
        for i in range(len(params), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Análisis de Sensibilidad de Parámetros', y=1.02)
        plt.show()
        
        # Gráfico de convergencia para mejores parámetros
        self._plot_best_convergence()
    
    def _plot_best_convergence(self):
        """
        Mostrar convergencia con mejores parámetros
        """
        # Encontrar mejores parámetros
        best_result = min(self.results, key=lambda x: x['avg_fitness'])
        best_params = best_result['params']
        
        print("\nEjecutando GA con mejores parámetros para visualización...")
        
        # Ejecutar una vez más con verbose
        ga = GeneticAlgorithmCVRP(
            distance_matrix=self.ga_params['distance_matrix'],
            demands=self.ga_params['demands'],
            vehicle_capacity=self.ga_params['vehicle_capacity'],
            depot_id=self.ga_params['depot_id'],
            num_vehicles=self.ga_params['num_vehicles'],
            **best_params
        )
        
        ga.solve(verbose=False)
        ga.plot_convergence()
        

def run_calibration():
    """
    Ejecutar calibración completa
    """
    print("CALIBRACIÓN DE PARÁMETROS PARA GA CVRP")
    print("="*60)
    
    # Definir grilla de parámetros a probar
    param_grid = {
        'population_size': [50, 100, 150],
        'generations': [200, 300, 400],
        'mutation_rate': [0.1, 0.2, 0.3],
        'crossover_rate': [0.7, 0.8, 0.9],
        'elitism_rate': [0.05, 0.1, 0.15],
        'tournament_size': [3, 5, 7]
    }
    
    # Para prueba rápida, usar grilla más pequeña
    param_grid_small = {
        'population_size': [100, 150],
        'generations': [300],
        'mutation_rate': [0.15, 0.25],
        'crossover_rate': [0.8, 0.85],
        'elitism_rate': [0.1],
        'tournament_size': [5]
    }
    
    # Crear calibrador
    calibrator = GACalibration(case_number=1)
    
    # Ejecutar búsqueda
    print("\nUsando grilla reducida para calibración rápida...")
    results = calibrator.grid_search(param_grid_small, num_runs=3)
    
    # Analizar resultados
    df_results = calibrator.analyze_results()
    
    # Guardar resultados
    df_results.to_csv('calibration_results.csv', index=False)
    print("\nResultados guardados en 'calibration_results.csv'")
    
    return calibrator, df_results


if __name__ == "__main__":
    calibrator, results = run_calibration()