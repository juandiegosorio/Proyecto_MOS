# Proyecto de Optimización Logística: Ruteo de Vehículos en Bogotá para LogistiCo

## 1. Descripción del Proyecto

Este proyecto aborda el desafío de optimizar las operaciones logísticas de LogistiCo en Bogotá mediante la implementación de un modelo de optimización para el problema de Ruteo de Vehículos con capacidades y restricciones de rango (Capacitated Vehicle Routing Problem - CVRP). Nuestro enfoque integra la modelación matemática con datos geográficos reales de la ciudad, obtenidos a través de la API de Openrouteservice, para generar planes de ruteo eficientes que minimicen costos operativos y de transporte.

## 2. Contexto y Objetivos

El contexto operativo de LogistiCo en Bogotá presenta complejidades significativas, incluyendo la gestión de entregas desde un centro de distribución principal, la atención a múltiples clientes con demandas variadas, y la necesidad de operar una flota de vehículos con capacidades y rangos definidos, todo esto dentro de un entorno urbano conocido por su alta congestión y dinámicas de movilidad particulares.

El **objetivo principal** de esta fase fue implementar en Pyomo el modelo de optimización previamente formulado. Esta implementación busca ser una herramienta práctica y ejecutable que capture fielmente estas complejidades, permitiendo la minimización efectiva de los costos totales de operación al optimizar la asignación de inventario desde el depot y la planificación detallada de las rutas vehiculares, siempre respetando las limitaciones de LogistiCo.

## 3. Metodología y Herramientas

Para cumplir con los objetivos, seguimos la siguiente metodología e implementamos la solución utilizando las siguientes herramientas:

* **Modelación Matemática:** Utilizamos una formulación estándar de Programación Lineal Entera Mixta (MILP) para el CVRP, adaptada para incluir restricciones de capacidad de depot, capacidad vehicular, rango de vehículos y eliminación de subtours (mediante la formulación MTZ).
* **Framework de Optimización:** Implementamos el modelo matemático en **Pyomo**, un framework de modelado algebraico en Python, que nos permite definir el problema de manera abstracta y conectarlo con diversos solvers.
* **Datos Geográficos Reales:** Obtuvimos matrices de distancia y duración entre todas las ubicaciones (depots y clientes) utilizando la **API de Openrouteservice (driving-car profile)**. Esto asegura que las rutas y costos calculados se basen en estimaciones realistas del tiempo y la distancia de viaje en Bogotá.
* **Solver:** Utilizamos el solver de programación lineal entera **HiGHS** (integrado con Pyomo) para encontrar soluciones al modelo. Dada la complejidad del problema, configuramos un límite de tiempo para obtener soluciones factibles dentro de un marco temporal razonable.
* **Visualización:** Desarrollamos visualizaciones interactivas utilizando **Folium** para representar las ubicaciones de depots y clientes, así como las rutas optimizadas en un mapa real de Bogotá. Opcionalmente, también incluimos una visualización estática con **NetworkX** durante el desarrollo.
* **Procesamiento de Datos:** Empleamos **Pandas** para cargar, manipular y procesar los datos de entrada (depots, clientes, vehículos) desde archivos CSV.
* **Archivo de Verificación:** Implementamos la generación de un archivo CSV estandarizado que resume los resultados clave por vehículo para facilitar la verificación y auditoría de las soluciones.
