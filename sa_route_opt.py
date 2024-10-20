import numpy as np
import matplotlib.pyplot as plt

import random
random.seed(42)

import warnings 
warnings.filterwarnings('ignore')

# Parameters
n_locations = 10  # 10 delivery locations
drones = 2  # Number of drones
warehouse = (0, 0)  # Starting point for both drones

# Generate random coordinates for the 10 delivery locations
np.random.seed(42)  # Set seed for reproducibility
locations = [(random.uniform(5, 30), random.uniform(5, 30)) for _ in range(n_locations)]
locations.insert(0, warehouse)  # Add the warehouse as the starting point (index 0)

print(locations)

# Calculate distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Create a distance matrix for all locations
distance_matrix = np.zeros((n_locations + 1, n_locations + 1))
for i in range(n_locations + 1):
    for j in range(n_locations + 1):
        distance_matrix[i, j] = distance(locations[i], locations[j])

# Simulated Annealing Parameters
initial_temp = 100  # Starting temperature
final_temp = 1  # Stopping temperature
alpha = 0.995  # Cooling rate
max_iterations = 1000  # Number of iterations at each temperature

# Initialize solution (random division of locations between two drones)
def initial_solution():
    drone1 = random.sample(range(1, n_locations + 1), n_locations // drones)
    drone2 = [i for i in range(1, n_locations + 1) if i not in drone1]
    return drone1, drone2

# Calculate total distance for a drone route (including return to warehouse)
def route_distance(route):
    total_distance = distance_matrix[0, route[0]]  # From warehouse to first location
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i+1]]
    total_distance += distance_matrix[route[-1], 0]  # Return to warehouse
    return total_distance

# Calculate total cost (distance) of a solution
def total_distance(drone1, drone2):
    return route_distance(drone1) + route_distance(drone2)

# Generate a neighbor solution by swapping a location between the two drones
def swap_locations(drone1, drone2):
    if len(drone1) > 0 and len(drone2) > 0:
        loc1 = random.choice(drone1)
        loc2 = random.choice(drone2)
        drone1.remove(loc1)
        drone2.remove(loc2)
        drone1.append(loc2)
        drone2.append(loc1)
    return drone1, drone2

# Simulated Annealing
def simulated_annealing():
    current_solution = initial_solution()
    current_cost = total_distance(*current_solution)
    best_solution = current_solution
    best_cost = current_cost
    
    temp = initial_temp
    while temp > final_temp:
        for _ in range(max_iterations):
            # Generate neighbor solution
            new_solution = swap_locations(current_solution[0][:], current_solution[1][:])
            new_cost = total_distance(*new_solution)

            # Accept new solution with probability based on temperature
            if new_cost < current_cost or random.uniform(0, 1) < np.exp((current_cost - new_cost) / temp):
                current_solution = new_solution
                current_cost = new_cost

                # Update the best solution found so far
                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost

        temp *= alpha  # Cool down

    return best_solution, best_cost

# Run Simulated Annealing
best_solution, best_cost = simulated_annealing()

best_solution, best_cost, locations

print(best_solution)

print(best_cost)