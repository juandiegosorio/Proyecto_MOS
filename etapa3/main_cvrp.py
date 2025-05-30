# -*- coding: utf-8 -*-
"""
Main script para ejecutar el GA en CVRP
"""

from ga_cvrp import GeneticAlgorithmCVRP
from data_loader import CVRPDataLoader
import matplotlib.pyplot as plt
import time

def run_ga_for_case(case_number, generations=300, population_size=150):
    """
    Ejecutar el algoritmo genético para un caso específico.
    
    Args:
        case_number (int): Número del caso (1, 2, o 3)
        generations (int): Número de generaciones
        population_size (int): Tamaño de la población
    """
    print(f"\n{'='*60}")
    print(f"EJECUTANDO ALGORITMO GENÉTICO PARA CASO {case_number}")
    print(f"{'='*60}")
    
    # Cargar datos
    loader = CVRPDataLoader()
    loader.load_case(case_number)
    loader.print_problem_summary()
    
    # Obtener parámetros para GA
    ga_params = loader.get_ga_parameters()
    
    # Crear y configurar GA
    ga = GeneticAlgorithmCVRP(
        distance_matrix=ga_params['distance_matrix'],
        demands=ga_params['demands'],
        vehicle_capacity=ga_params['vehicle_capacity'],
        depot_id=ga_params['depot_id'],
        num_vehicles=ga_params['num_vehicles'],
        population_size=population_size,
        generations=generations,
        mutation_rate=0.25,
        crossover_rate=0.8,
        elitism_rate=0.1,
        tournament_size=5
    )
    
    # Resolver
    print(f"\nIniciando optimización con {generations} generaciones...")
    start_time = time.time()
    
    best_solution, best_fitness = ga.solve(verbose=True)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nTiempo total de ejecución: {execution_time:.2f} segundos")
    
    # Obtener detalles de la solución
    solution_details = ga.get_solution_details()
    
    # Guardar archivo de verificación
    if solution_details:
        loader.save_verification_file(solution_details, case_number, "GA")
    
    # Graficar convergencia
    ga.plot_convergence()
    
    return ga, loader, execution_time


def main():
    """
    Función principal para ejecutar todos los casos.
    """
    # Configuración
    cases_to_run = [1] 
    
    results = {}
    
    for case in cases_to_run:
        print(f"\n{'#'*70}")
        print(f"CASO {case}")
        print(f"{'#'*70}")
        
        # Ejecutar GA
        ga, loader, exec_time = run_ga_for_case(
            case_number=case,
            generations=300,  # Ajustar según necesidad
            population_size=100
        )
        
        # Guardar resultados
        results[case] = {
            'ga': ga,
            'loader': loader,
            'execution_time': exec_time,
            'best_fitness': ga.best_fitness,
            'solution': ga.best_solution
        }
    
    # Resumen de resultados
    print(f"\n{'='*60}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'='*60}")
    
    for case, result in results.items():
        print(f"\nCaso {case}:")
        print(f"  - Mejor costo encontrado: ${result['best_fitness']:,.2f}")
        print(f"  - Tiempo de ejecución: {result['execution_time']:.2f} segundos")
        print(f"  - Número de vehículos usados: {len([r for r in result['solution'] if r])}")
    
    return results


if __name__ == "__main__":
    results = main()