# -*- coding: utf-8 -*-
"""
Genetic Algorithm for Capacitated Vehicle Routing Problem (CVRP)
Adaptado del tutorial MTSP para resolver CVRP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
import time
import csv
from collections import defaultdict
from copy import deepcopy
import folium
import requests

class GeneticAlgorithmCVRP:
    """
    Genetic Algorithm for Capacitated Vehicle Routing Problem (CVRP).
    
    A diferencia del MTSP, aquí los vehículos tienen capacidad limitada
    y deben satisfacer las demandas de los clientes.
    """
    
    def __init__(self, distance_matrix, demands, vehicle_capacity, depot_id=0,
                 num_vehicles=None, population_size=100, generations=500, 
                 mutation_rate=0.2, crossover_rate=0.8, elitism_rate=0.1, 
                 tournament_size=5):
        """
        Initialize the Genetic Algorithm solver for CVRP.
        
        Args:
            distance_matrix (np.array): Matriz de distancias entre todos los nodos
            demands (dict): Diccionario {cliente_id: demanda}
            vehicle_capacity (float): Capacidad máxima de cada vehículo
            depot_id (int): ID del depósito (default: 0)
            num_vehicles (int): Número de vehículos disponibles (None = ilimitado)
            population_size (int): Tamaño de la población
            generations (int): Número máximo de generaciones
            mutation_rate (float): Probabilidad de mutación
            crossover_rate (float): Probabilidad de cruce
            elitism_rate (float): Proporción de elite a mantener
            tournament_size (int): Tamaño del torneo para selección
        """
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.depot_id = depot_id
        self.num_vehicles = num_vehicles
        
        # Crear lista de clientes (excluyendo el depósito)
        self.customers = [i for i in range(len(distance_matrix)) if i != depot_id]
        
        # Si no se especifica número de vehículos, calcular el mínimo necesario
        if self.num_vehicles is None:
            total_demand = sum(demands.get(c, 0) for c in self.customers)
            self.num_vehicles = int(np.ceil(total_demand / vehicle_capacity))
            # Agregar algunos vehículos extra por flexibilidad
            self.num_vehicles += 2
        
        # Parámetros del GA
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        
        # Resultados
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.best_solution_history = []
        self.population = []
        
        # Costo por km (del proyecto)
        self.cost_per_km = 5548.1  # F + M + Pf = 5000 + 23 + 525.1
        
    def initialize_population(self):
        """
        Crear población inicial de soluciones factibles para CVRP.
        Cada solución es una lista de rutas, una por vehículo.
        """
        population = []
        
        for _ in range(self.population_size):
            solution = self._create_random_solution()
            population.append(solution)
            
        self.population = population
        return population
    
    def _create_random_solution(self):
        """
        Crear una solución aleatoria factible respetando capacidades.
        """
        # Copiar lista de clientes y mezclar aleatoriamente
        unassigned_customers = self.customers.copy()
        random.shuffle(unassigned_customers)
        
        # Crear rutas vacías para cada vehículo
        routes = [[] for _ in range(self.num_vehicles)]
        vehicle_loads = [0] * self.num_vehicles
        
        # Asignar clientes a vehículos respetando capacidad
        for customer in unassigned_customers:
            demand = self.demands.get(customer, 0)
            assigned = False
            
            # Intentar asignar al vehículo con menor carga que pueda acomodarlo
            vehicle_indices = list(range(self.num_vehicles))
            vehicle_indices.sort(key=lambda v: vehicle_loads[v])
            
            for v in vehicle_indices:
                if vehicle_loads[v] + demand <= self.vehicle_capacity:
                    routes[v].append(customer)
                    vehicle_loads[v] += demand
                    assigned = True
                    break
            
            # Si no se pudo asignar, forzar en el vehículo con menor carga
            if not assigned:
                v = vehicle_indices[0]
                routes[v].append(customer)
                vehicle_loads[v] += demand
        
        # Aplicar 2-opt a cada ruta para mejorarla
        for i in range(len(routes)):
            if len(routes[i]) > 3:
                routes[i] = self._two_opt(routes[i])
        
        return routes
    
    def _two_opt(self, route, max_iterations=10):
        """
        Aplicar mejora 2-opt a una ruta individual.
        """
        if len(route) < 4:
            return route
            
        best_route = route.copy()
        best_distance = self._calculate_route_distance(best_route)
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(1, len(route) - 2):
                for j in range(i + 2, len(route)):
                    # Crear nueva ruta con segmento invertido
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_distance = self._calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        route = new_route
                        improved = True
                        break
                        
                if improved:
                    break
                    
        return best_route
    
    def _calculate_route_distance(self, route):
        """
        Calcular distancia total de una ruta (incluyendo ida y vuelta al depósito).
        """
        if not route:
            return 0
            
        distance = 0
        # Desde depósito al primer cliente
        distance += self.distance_matrix[self.depot_id][route[0]]
        
        # Entre clientes
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i]][route[i+1]]
            
        # Desde último cliente al depósito
        distance += self.distance_matrix[route[-1]][self.depot_id]
        
        return distance
    
    def _calculate_route_load(self, route):
        """
        Calcular carga total de una ruta.
        """
        return sum(self.demands.get(customer, 0) for customer in route)
    
    def evaluate_fitness(self, solution):
        """
        Calcular fitness (costo total) de una solución CVRP.
        Incluye penalizaciones por violación de capacidad.
        """
        total_distance = 0
        total_penalty = 0
        
        for route in solution:
            if route:  # Si la ruta no está vacía
                # Calcular distancia
                route_distance = self._calculate_route_distance(route)
                total_distance += route_distance
                
                # Verificar capacidad
                route_load = self._calculate_route_load(route)
                if route_load > self.vehicle_capacity:
                    # Penalización proporcional al exceso
                    excess = route_load - self.vehicle_capacity
                    total_penalty += excess * self.cost_per_km * 10  # Penalización alta
        
        # Verificar que todos los clientes estén asignados
        assigned_customers = set()
        for route in solution:
            assigned_customers.update(route)
            
        missing_customers = set(self.customers) - assigned_customers
        if missing_customers:
            # Penalización muy alta por clientes no atendidos
            total_penalty += len(missing_customers) * self.cost_per_km * 100
        
        # Fitness total = costo de distancia + penalizaciones
        total_cost = total_distance * self.cost_per_km + total_penalty
        
        return total_cost
    
    def select_parents(self):
        """
        Selección de padres mediante torneo.
        """
        def tournament():
            participants = random.sample(range(len(self.population)), 
                                      min(self.tournament_size, len(self.population)))
            participants_fitness = [(p, self.evaluate_fitness(self.population[p])) 
                                  for p in participants]
            winner = min(participants_fitness, key=lambda x: x[1])[0]
            return self.population[winner]
        
        parent1 = tournament()
        parent2 = tournament()
        
        return parent1, parent2
    
    def crossover(self, parent1, parent2):
        """
        Cruce adaptado para CVRP manteniendo factibilidad.
        """
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        # Usar Best Route Crossover
        return self._best_route_crossover(parent1, parent2)
    
    def _best_route_crossover(self, parent1, parent2):
        """
        Best Route Crossover: selecciona las mejores rutas de cada padre.
        """
        # Evaluar cada ruta individualmente
        routes_p1 = [(i, route, self._calculate_route_distance(route) * self.cost_per_km) 
                     for i, route in enumerate(parent1) if route]
        routes_p2 = [(i, route, self._calculate_route_distance(route) * self.cost_per_km) 
                     for i, route in enumerate(parent2) if route]
        
        # Combinar y ordenar por costo
        all_routes = routes_p1 + routes_p2
        all_routes.sort(key=lambda x: x[2])
        
        # Crear offspring seleccionando las mejores rutas sin repetir clientes
        child1 = [[] for _ in range(self.num_vehicles)]
        child2 = [[] for _ in range(self.num_vehicles)]
        
        assigned_c1 = set()
        assigned_c2 = set()
        
        # Llenar child1
        route_idx = 0
        for _, route, _ in all_routes:
            if route_idx >= self.num_vehicles:
                break
            
            # Verificar que no haya clientes repetidos
            route_customers = set(route)
            if not route_customers.intersection(assigned_c1):
                # Verificar capacidad
                if self._calculate_route_load(route) <= self.vehicle_capacity:
                    child1[route_idx] = route.copy()
                    assigned_c1.update(route_customers)
                    route_idx += 1
        
        # Asignar clientes faltantes a child1
        missing_c1 = set(self.customers) - assigned_c1
        for customer in missing_c1:
            # Buscar vehículo con capacidad disponible
            for v in range(self.num_vehicles):
                current_load = self._calculate_route_load(child1[v])
                if current_load + self.demands.get(customer, 0) <= self.vehicle_capacity:
                    child1[v].append(customer)
                    break
        
        # Proceso similar para child2 (con rutas en orden inverso para diversidad)
        route_idx = 0
        for _, route, _ in reversed(all_routes):
            if route_idx >= self.num_vehicles:
                break
                
            route_customers = set(route)
            if not route_customers.intersection(assigned_c2):
                if self._calculate_route_load(route) <= self.vehicle_capacity:
                    child2[route_idx] = route.copy()
                    assigned_c2.update(route_customers)
                    route_idx += 1
        
        missing_c2 = set(self.customers) - assigned_c2
        for customer in missing_c2:
            for v in range(self.num_vehicles):
                current_load = self._calculate_route_load(child2[v])
                if current_load + self.demands.get(customer, 0) <= self.vehicle_capacity:
                    child2[v].append(customer)
                    break
        
        return child1, child2
    
    def mutate(self, solution):
        """
        Aplicar operadores de mutación específicos para CVRP.
        """
        if random.random() > self.mutation_rate:
            return solution
        
        mutated = deepcopy(solution)
        
        # Elegir tipo de mutación
        mutation_type = random.choice(['swap', 'relocate', 'exchange', '2-opt'])
        
        if mutation_type == 'swap':
            # Intercambiar dos clientes dentro de la misma ruta
            non_empty_routes = [i for i, route in enumerate(mutated) if len(route) >= 2]
            if non_empty_routes:
                route_idx = random.choice(non_empty_routes)
                route = mutated[route_idx]
                if len(route) >= 2:
                    i, j = random.sample(range(len(route)), 2)
                    route[i], route[j] = route[j], route[i]
                    
        elif mutation_type == 'relocate':
            # Mover un cliente de una ruta a otra
            non_empty_routes = [i for i, route in enumerate(mutated) if route]
            if len(non_empty_routes) >= 2:
                from_idx = random.choice(non_empty_routes)
                to_idx = random.choice([i for i in range(self.num_vehicles) if i != from_idx])
                
                if mutated[from_idx]:
                    customer = random.choice(mutated[from_idx])
                    
                    # Verificar si cabe en la ruta destino
                    new_load = self._calculate_route_load(mutated[to_idx]) + self.demands.get(customer, 0)
                    if new_load <= self.vehicle_capacity:
                        mutated[from_idx].remove(customer)
                        insert_pos = random.randint(0, len(mutated[to_idx]))
                        mutated[to_idx].insert(insert_pos, customer)
                        
        elif mutation_type == 'exchange':
            # Intercambiar clientes entre dos rutas
            non_empty_routes = [i for i, route in enumerate(mutated) if route]
            if len(non_empty_routes) >= 2:
                idx1, idx2 = random.sample(non_empty_routes, 2)
                if mutated[idx1] and mutated[idx2]:
                    pos1 = random.randint(0, len(mutated[idx1]) - 1)
                    pos2 = random.randint(0, len(mutated[idx2]) - 1)
                    
                    customer1 = mutated[idx1][pos1]
                    customer2 = mutated[idx2][pos2]
                    
                    # Verificar factibilidad del intercambio
                    load1_new = (self._calculate_route_load(mutated[idx1]) - 
                               self.demands.get(customer1, 0) + 
                               self.demands.get(customer2, 0))
                    load2_new = (self._calculate_route_load(mutated[idx2]) - 
                               self.demands.get(customer2, 0) + 
                               self.demands.get(customer1, 0))
                    
                    if (load1_new <= self.vehicle_capacity and 
                        load2_new <= self.vehicle_capacity):
                        mutated[idx1][pos1] = customer2
                        mutated[idx2][pos2] = customer1
                        
        elif mutation_type == '2-opt':
            # Aplicar 2-opt a una ruta aleatoria
            non_empty_routes = [i for i, route in enumerate(mutated) if len(route) > 3]
            if non_empty_routes:
                route_idx = random.choice(non_empty_routes)
                mutated[route_idx] = self._two_opt(mutated[route_idx], max_iterations=5)
        
        return mutated
    
    def evolve_population(self):
        """
        Evolucionar la población a la siguiente generación.
        """
        # Evaluar población actual
        population_fitness = [(i, self.evaluate_fitness(solution)) 
                            for i, solution in enumerate(self.population)]
        
        # Ordenar por fitness (menor es mejor)
        population_fitness.sort(key=lambda x: x[1])
        
        # Mantener élite
        num_elite = max(1, int(self.elitism_rate * self.population_size))
        elite_indices = [idx for idx, _ in population_fitness[:num_elite]]
        new_population = [deepcopy(self.population[idx]) for idx in elite_indices]
        
        # Generar resto de la población
        while len(new_population) < self.population_size:
            # Seleccionar padres
            parent1, parent2 = self.select_parents()
            
            # Cruce
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutación
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Añadir a nueva población
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
    
    def solve(self, verbose=True, early_stopping_generations=50):
        """
        Ejecutar el algoritmo genético para resolver CVRP.
        """
        # Inicializar población
        self.initialize_population()
        
        # Variables para tracking
        best_solution = None
        best_fitness = float('inf')
        generations_without_improvement = 0
        start_time = time.time()
        
        # Bucle principal
        for generation in range(self.generations):
            # Evolucionar población
            self.evolve_population()
            
            # Encontrar mejor solución en la generación actual
            current_best = None
            current_best_fitness = float('inf')
            
            for solution in self.population:
                fitness = self.evaluate_fitness(solution)
                if fitness < current_best_fitness:
                    current_best = solution
                    current_best_fitness = fitness
            
            # Actualizar mejor global
            if current_best_fitness < best_fitness:
                best_solution = deepcopy(current_best)
                best_fitness = current_best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Guardar historial
            self.fitness_history.append(current_best_fitness)
            self.best_solution_history.append(best_fitness)
            
            # Imprimir progreso
            if verbose and generation % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Generación {generation}: Mejor Fitness = {best_fitness:.2f}, "
                      f"Fitness Actual = {current_best_fitness:.2f}, "
                      f"Tiempo = {elapsed_time:.2f}s")
            
            # Early stopping
            if generations_without_improvement >= early_stopping_generations:
                if verbose:
                    print(f"Early stopping en generación {generation} - "
                          f"Sin mejora por {early_stopping_generations} generaciones.")
                break
        
        # Resultados finales
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        
        if verbose:
            total_time = time.time() - start_time
            print(f"\nOptimización completa.")
            print(f"Mejor fitness: {best_fitness:.2f}")
            print(f"Tiempo total: {total_time:.2f}s")
            
            # Verificar factibilidad
            self._print_solution_details()
        
        return best_solution, best_fitness
    
    def _print_solution_details(self):
        """
        Imprimir detalles de la mejor solución encontrada.
        """
        print("\nDetalles de la mejor solución:")
        total_distance = 0
        total_customers = 0
        
        for i, route in enumerate(self.best_solution):
            if route:
                distance = self._calculate_route_distance(route)
                load = self._calculate_route_load(route)
                total_distance += distance
                total_customers += len(route)
                
                print(f"Vehículo {i+1}: {len(route)} clientes, "
                      f"Carga: {load}/{self.vehicle_capacity}, "
                      f"Distancia: {distance:.2f} km")
        
        print(f"\nTotal clientes atendidos: {total_customers}/{len(self.customers)}")
        print(f"Distancia total: {total_distance:.2f} km")
        print(f"Costo total: {self.best_fitness:.2f}")
    
    def plot_convergence(self):
        """
        Graficar la convergencia del algoritmo.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, label='Mejor Fitness Generación Actual', alpha=0.6)
        plt.plot(self.best_solution_history, label='Mejor Fitness Global', linewidth=2)
        plt.xlabel('Generación')
        plt.ylabel('Fitness (Costo Total)')
        plt.title('Convergencia del Algoritmo Genético para CVRP')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_solution_details(self):
        """
        Obtener detalles de la solución en formato para verificación.
        """
        if not self.best_solution:
            return None
        
        details = []
        
        for vehicle_idx, route in enumerate(self.best_solution):
            if not route:
                continue
                
            # Calcular métricas
            distance = self._calculate_route_distance(route)
            load = self._calculate_route_load(route)
            fuel_cost = distance * self.cost_per_km
            
            # Crear secuencia de ruta con depósito
            route_sequence = [self.depot_id] + route + [self.depot_id]
            route_str = ' -> '.join(f'C{c}' if c != self.depot_id else 'D1' 
                                  for c in route_sequence)
            
            # Demandas satisfechas en orden
            demands_str = '-'.join(str(self.demands.get(c, 0)) for c in route)
            
            details.append({
                'VehicleId': f'V{vehicle_idx + 1}',
                'DepotId': 'D1',
                'InitialLoad': int(load),
                'RouteSequence': route_str,
                'ClientsServed': len(route),
                'DemandsSatisfied': demands_str,
                'TotalDistance': round(distance, 2),
                'TotalTime': round(distance * 2, 1),  # Estimación simple
                'FuelCost': round(fuel_cost, 2)
            })
        
        return details