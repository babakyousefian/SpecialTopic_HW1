# SpecialTopic_HW1

---

Code Breakdown and Explanation:

## 1. Import Necessary Libraries
```bash
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import combinations
```

- numpy: Used for numerical operations.

- matplotlib.pyplot: Used for plotting the TSP route.

- random: Used for shuffling the initial TSP tour.

- math: Provides mathematical functions like square root.

- itertools.combinations: Used to generate all possible pairs of indices for the 2-opt swap.

---

## 2. Load TSP Data from File

```bash
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
```

- Reads the file containing the TSP problem.

- Extracts city coordinates starting from NODE_COORD_SECTION.

- Each city is stored as a tuple: (city_number, x, y).

---

## 3. Compute Euclidean Distance Between Two Cities

```bash
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
```

- Computes the straight-line (Euclidean) distance between two cities using the formula:

- - 𝑑=(𝑥2−𝑥1)2+(𝑦2−𝑦1)2
 
---

## 4. Compute Total Distance of a Tour

```bash
def total_distance(tour, cities):
    return sum(euclidean_distance(cities[tour[i]-1][1:], cities[tour[i+1]-1][1:]) for i in range(len(tour)-1)) + \
           euclidean_distance(cities[tour[-1]-1][1:], cities[tour[0]-1][1:])
```

- Iterates over the tour, summing up the Euclidean distances between consecutive cities.

- Closes the tour by adding the distance from the last city back to the first.

---

## 5. 2-opt Swap Function

```bash
def two_opt_swap(tour, i, k):
    return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
```

- Swaps a segment of the tour between i and k, reversing the order of cities in that segment.

- This helps to find a shorter route.

---

## 6. 2-opt Local Search Algorithm
```bash
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
```

- Iteratively improves the tour by performing 2-opt swaps.

- If a swap reduces the total distance, it updates the tour.

- The process continues until no further improvements can be made.

---

## 7. Plot the Best Tour
```bash
def plot_tour(tour, cities, title):
    x = [cities[i-1][1] for i in tour] + [cities[tour[0]-1][1]]
    y = [cities[i-1][2] for i in tour] + [cities[tour[0]-1][2]]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-', markersize=4)
    plt.title(title)
    plt.show()
```

- Extracts the x, y coordinates of cities in the tour.

- Plots the tour path with connected lines.

- Closes the tour by connecting the last city back to the first.

---

## 8. Load Cities from File
```bash
tsp_cities = load_tsp_data("/mnt/data/kroB200.txt")
```
- Reads 200-city TSP data from the provided file.

---

## 9. Generate an Initial Random Tour
```bash
initial_tour = list(range(1, len(tsp_cities) + 1))
random.shuffle(initial_tour)
```

- Creates a random permutation of city indices.

- This serves as the initial solution for the optimization.

---

## 10. Run 2-opt Local Search
```bash
best_tour_2opt, best_distance_2opt = local_search_2opt(initial_tour, tsp_cities)
print("Best Distance (2-opt):", best_distance_2opt)
plot_tour(best_tour_2opt, tsp_cities, "Best Tour (2-opt)")
```

- Runs 2-opt local search on the random tour.

- Prints the best tour distance found.

- Visualizes the best tour.

---
---
---

# TSP Optimization Using 2-opt Local Search

## Overview
This project implements the Traveling Salesman Problem (TSP) using the **2-opt local search algorithm** in Python. The goal is to find an optimized route that minimizes the total distance traveled.

## Dataset
The input dataset `kroB200.txt` contains **200 cities**, each represented by:
- A city **ID**
- **x, y coordinates** on a Euclidean plane

## Algorithm Implementation
### 1. **Loading the TSP Data**
The `load_tsp_data()` function reads the input file and extracts city coordinates.

### 2. **Computing Distance**
The `euclidean_distance()` function calculates the straight-line distance between two cities.

### 3. **Calculating Total Tour Distance**
The `total_distance()` function sums up the Euclidean distances of the given tour.

### 4. **2-opt Swap**
The `two_opt_swap()` function reverses a segment of the tour between two indices to generate a new route.

### 5. **2-opt Local Search**
The `local_search_2opt()` function iteratively improves the tour using the 2-opt swap technique until no further improvements can be made.

### 6. **Tour Visualization**
The `plot_tour()` function plots the best-found tour on a 2D plane.

## Steps to Run the Code
1. Load the dataset.
2. Generate an **initial random tour**.
3. Apply **2-opt local search** to optimize the tour.
4. Print the best tour distance.
5. **Plot the optimized tour.**

## Output
- **Best tour distance** found by the algorithm.
- **Graphical representation** of the best tour.

## Future Enhancements
This project is a foundation for other TSP optimization techniques, such as:
- **3-opt local search**
- **Simulated Annealing (SA)**
- **Variable Neighborhood Search (VNS)**
- **Greedy Randomized Adaptive Search Procedure (GRASP)**

---

---

---

### Author : babak yousefian

---

---

---

