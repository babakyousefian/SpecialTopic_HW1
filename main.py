import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import combinations

def load_tsp_data(file_path):
    cities = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        start_reading = False
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                start_reading = True
                continue
            if line.startswith("EOF"):
                break
            if start_reading:
                parts = line.strip().split()
                cities.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return cities

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def total_distance(tour, cities):
    return sum(euclidean_distance(cities[tour[i]-1][1:], cities[tour[i+1]-1][1:]) for i in range(len(tour)-1)) + \
           euclidean_distance(cities[tour[-1]-1][1:], cities[tour[0]-1][1:])


def two_opt_swap(tour, i, k):
    return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]


def local_search_2opt(tour, cities):
    best_distance = total_distance(tour, cities)
    improved = True
    while improved:
        improved = False
        for i, k in combinations(range(1, len(tour)-1), 2):
            new_tour = two_opt_swap(tour, i, k)
            new_distance = total_distance(new_tour, cities)
            if new_distance < best_distance:
                tour = new_tour
                best_distance = new_distance
                improved = True
    return tour, best_distance


def plot_tour(tour, cities, title):
    x = [cities[i-1][1] for i in tour] + [cities[tour[0]-1][1]]
    y = [cities[i-1][2] for i in tour] + [cities[tour[0]-1][2]]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-', markersize=4)
    plt.title(title)
    plt.show()


tsp_cities = load_tsp_data("kroB200.txt")


initial_tour = list(range(1, len(tsp_cities) + 1))
random.shuffle(initial_tour)


best_tour_2opt, best_distance_2opt = local_search_2opt(initial_tour, tsp_cities)
print("Best Distance (2-opt):", best_distance_2opt)
plot_tour(best_tour_2opt, tsp_cities, "Best Tour (2-opt)")
