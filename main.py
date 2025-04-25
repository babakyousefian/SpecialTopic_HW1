import pygame
import operator
import os
import csv
from fpdf import FPDF
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import combinations
from tqdm import tqdm

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


def calculate_variance(values):
    backup_values = values
    i = int(backup_values[0])
    if (int(i)) <= 1:
        return 0
    mean = sum(values) / 200
    variance = sum((x - mean) ** 2 for x in values) / 200
    return variance


def calculate_std(values):
    variance = calculate_variance(values)
    return math.sqrt(variance)


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

def three_opt_swap(tour, i, j, k):
    segments = [
        tour[:i] + tour[i:j] + tour[j:k] + tour[k:],
        tour[:i] + tour[i:j][::-1] + tour[j:k] + tour[k:],
        tour[:i] + tour[i:j] + tour[j:k][::-1] + tour[k:],
        tour[:i] + tour[j:k] + tour[i:j] + tour[k:],
        tour[:i] + tour[j:k][::-1] + tour[i:j][::-1] + tour[k:],
        tour[:i] + tour[j:k] + tour[i:j][::-1] + tour[k:],
        tour[:i] + tour[i:j][::-1] + tour[j:k][::-1] + tour[k:],
        tour[:i] + tour[j:k][::-1] + tour[i:j] + tour[k:]
    ]
    return segments

def local_search_3opt_limited(tour, cities, max_time=1200):
    best_distance = total_distance(tour, cities)
    start_time = time.time()
    improved = True
    iterations = 0
    n = len(tour)

    while improved and (time.time() - start_time < max_time):
        improved = False
        iterations += 1
        for i in tqdm(range(1, n - 2), desc=f"3-opt Iter {iterations}", leave=False):
            for j in range(i+1, n - 1):
                for k in range(j+1, n):
                    if time.time() - start_time > max_time:
                        print("\nReached time limit for 3-opt.")
                        return tour, best_distance
                    for new_tour in three_opt_swap(tour, i, j, k):
                        new_distance = total_distance(new_tour, cities)
                        if new_distance < best_distance:
                            tour = new_tour
                            best_distance = new_distance
                            improved = True
        if not improved:
            break
    return tour, best_distance

def simulated_annealing(tour, cities, initial_temp=10000, cooling_rate=0.995, stop_temp=1e-3, max_iter=1000):
    current_tour = tour[:]
    current_distance = total_distance(current_tour, cities)
    best_tour = current_tour[:]
    best_distance = current_distance
    temp = initial_temp

    while temp > stop_temp:
        for _ in tqdm(range(max_iter), desc="Simulated Annealing 2-opt", leave=False):
            i, k = sorted(random.sample(range(1, len(tour)-1), 2))
            new_tour = two_opt_swap(current_tour, i, k)
            new_distance = total_distance(new_tour, cities)

            delta = new_distance - current_distance
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_tour = new_tour
                current_distance = new_distance
                if current_distance < best_distance:
                    best_tour = current_tour[:]
                    best_distance = current_distance
        temp *= cooling_rate

    return best_tour, best_distance

def simulated_annealing_3opt(tour, cities, initial_temp=10000, cooling_rate=0.995, stop_temp=1e-3, max_iter=500):
    current_tour = tour[:]
    current_distance = total_distance(current_tour, cities)
    best_tour = current_tour[:]
    best_distance = current_distance
    temp = initial_temp
    n = len(tour)

    while temp > stop_temp:
        for _ in tqdm(range(max_iter), desc="Simulated Annealing 3-opt", leave=False):
            i, j, k = sorted(random.sample(range(1, n - 1), 3))
            for new_tour in random.sample(three_opt_swap(current_tour, i, j, k), 1):
                new_distance = total_distance(new_tour, cities)
                delta = new_distance - current_distance
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_tour = new_tour
                    current_distance = new_distance
                    if current_distance < best_distance:
                        best_tour = current_tour[:]
                        best_distance = current_distance
        temp *= cooling_rate

    return best_tour, best_distance

def vns(tour, cities, max_iter=10, max_time=1200):
    global new_distance, new_tour
    best_tour = tour[:]
    best_distance = total_distance(tour, cities)
    neighborhoods = ['2opt', '3opt']
    start_time = time.time()

    for iter in tqdm(range(max_iter), desc="VNS Iterations", leave=False):
        if time.time() - start_time > max_time:
            print("Time limit reached.")
            break

        for neighborhood in neighborhoods:
            if neighborhood == '2opt':
                new_tour, new_distance = local_search_2opt(best_tour[:], cities)
            elif neighborhood == '3opt':
                new_tour, new_distance = local_search_3opt_limited(best_tour[:], cities, max_time=10)

            if new_distance < best_distance:
                best_tour = new_tour
                best_distance = new_distance
                break
        else:
            break

    return best_tour, best_distance

def construct_greedy_randomized_solution(cities, alpha=0.2):
    unvisited = set(range(1, len(cities) + 1))
    current = random.choice(list(unvisited))
    tour = [current]
    unvisited.remove(current)

    while unvisited:
        distances = [(city, euclidean_distance(cities[current - 1][1:], cities[city - 1][1:])) for city in unvisited]
        distances.sort(key=lambda x: x[1])
        rcl_size = max(1, int(alpha * len(distances)))
        rcl = distances[:rcl_size]
        next_city = random.choice(rcl)[0]
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return tour


def grasp(cities, max_iter=10, alpha=0.2):
    best_tour = None
    best_distance = float('inf')

    for i in tqdm(range(max_iter), desc="GRASP Iterations", leave=False):
        greedy_tour = construct_greedy_randomized_solution(cities, alpha)
        local_tour, local_distance = local_search_2opt(greedy_tour, cities)

        if local_distance < best_distance:
            best_distance = local_distance
            best_tour = local_tour

    return best_tour, best_distance

def print_std_and_variance(tour, cities, algorithm_name):
    tour_distance = total_distance(tour, cities)
    variance = calculate_variance([tour_distance])
    std = calculate_std([tour_distance])
    print(f"{algorithm_name} - Distance: {tour_distance}, Variance: {variance}, Standard Deviation: {std}")


def plot_tour(tour, cities, title):
    x = [cities[i-1][1] for i in tour] + [cities[tour[0]-1][1]]
    y = [cities[i-1][2] for i in tour] + [cities[tour[0]-1][2]]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-', markersize=4)
    plt.title(title)
    plt.show()

def play_my_own_sounds(file_path:str=""):
    if not isinstance(file_path , str):
        raise TypeError("\n\n an Error occur during the play songs...!!!")
    else:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)



def run_all_algorithm_here(tsp_cities , initial_tour , among_patcher , starting_patcher , ending_patcher):

    pygame.init()

    result = {"2-opt":[] , "3-opt":[] , "SA_2-opt":[] , "SA_3-opt":[] , "VNS":[] , "GRASP":[]}

    myCheckerOutput = {"2-opt":0 , "3-opt":0 , "SA_2-opt":0 , "SA_3-opt":0 , "VNS":0 , "GRASP":0}

    play_my_own_sounds(starting_patcher)
    time.sleep(10)

    print("\nRunning 2-opt...")
    start_2opt = time.time()
    best_tour_2opt, best_distance_2opt = local_search_2opt(initial_tour[:], tsp_cities)
    print("Best Distance (2-opt):", best_distance_2opt)
    print("2-opt Time:", time.time() - start_2opt, "seconds")
    print_std_and_variance(best_tour_2opt, tsp_cities, "2-opt")
    plot_tour(best_tour_2opt, tsp_cities, "Best Tour (2-opt)")
    result["2-opt"] = best_tour_2opt
    myCheckerOutput["2-opt"] = best_distance_2opt
    play_my_own_sounds(among_patcher)
    time.sleep(4)

    print("\nRunning 3-opt (with 1200s limit)...")
    start_3opt = time.time()
    best_tour_3opt, best_distance_3opt = local_search_3opt_limited(initial_tour[:], tsp_cities, max_time=1200)
    print("Best Distance (3-opt, limited):", best_distance_3opt)
    print("3-opt Time:", time.time() - start_3opt, "seconds")
    print_std_and_variance(best_tour_3opt, tsp_cities, "3-opt")
    plot_tour(best_tour_3opt, tsp_cities, "Best Tour (3-opt, Limited)")
    result["3-opt"] = best_tour_3opt
    myCheckerOutput["3-opt"] = best_distance_3opt
    play_my_own_sounds(among_patcher)
    time.sleep(4)


    print("\nRunning Simulated Annealing...")
    start_sa = time.time()
    best_tour_sa, best_distance_sa = simulated_annealing(initial_tour[:], tsp_cities)
    print("Best Distance (Simulated Annealing with 2-opt):", best_distance_sa)
    print("SA Time:", time.time() - start_sa, "seconds")
    print_std_and_variance(best_tour_sa, tsp_cities, "Simulated Annealing")
    plot_tour(best_tour_sa, tsp_cities, "Best Tour (SA with 2-opt)")
    result["SA_2-opt"] = best_tour_sa
    myCheckerOutput["SA_2-opt"] = best_distance_sa
    play_my_own_sounds(among_patcher)
    time.sleep(4)


    print("\nRunning Simulated Annealing (3-opt)...")
    start_sa3 = time.time()
    best_tour_sa3opt, best_distance_sa3opt = simulated_annealing_3opt(initial_tour[:], tsp_cities)
    print("Best Distance (SA with 3-opt):", best_distance_sa3opt)
    print("SA 3-opt Time:", time.time() - start_sa3, "seconds")
    print_std_and_variance(best_tour_sa3opt, tsp_cities, "Simulated Annealing with 3-opt")
    plot_tour(best_tour_sa3opt, tsp_cities, "Best Tour (SA with 3-opt)")
    result["SA_3-opt"] = best_tour_sa3opt
    myCheckerOutput["SA_3-opt"] = best_distance_sa3opt
    play_my_own_sounds(among_patcher)
    time.sleep(4)


    print("\nRunning VNS (2-opt + 3-opt)...")
    start_vns = time.time()
    best_tour_vns, best_distance_vns = vns(initial_tour[:], tsp_cities)
    print("Best Distance (VNS):", best_distance_vns)
    print("VNS Time:", time.time() - start_vns, "seconds")
    print_std_and_variance(best_tour_vns, tsp_cities, "VNS")
    plot_tour(best_tour_vns, tsp_cities, "Best Tour (VNS)")
    result["VNS"] = best_tour_vns
    myCheckerOutput["VNS"] = best_distance_vns
    play_my_own_sounds(among_patcher)
    time.sleep(4)


    print("\nRunning GRASP with 2-opt...")
    start_grasp = time.time()
    best_tour_grasp, best_distance_grasp = grasp(tsp_cities, max_iter=10, alpha=0.3)
    print("Best Distance (GRASP with 2-opt):", best_distance_grasp)
    print("GRASP Time:", time.time() - start_grasp, "seconds")
    print_std_and_variance(best_tour_grasp, tsp_cities, "GRASP with 2-opt")
    plot_tour(best_tour_grasp, tsp_cities, "Best Tour (GRASP with 2-opt)")
    result["GRASP"] = best_tour_grasp
    myCheckerOutput["GRASP"] = best_distance_grasp
    play_my_own_sounds(among_patcher)
    time.sleep(4)

    stats = {}

    for name, tour in result.items():
        dist = total_distance(tour, tsp_cities)
        plot_tour(tour, tsp_cities, f"{name}_best_tour")
        stats[name] = dist
        print(f"{name} Distance: {dist:.2f}")

    play_my_own_sounds(ending_patcher)
    time.sleep(4)

    return stats, result , myCheckerOutput


def save_report(stats):
    with open("tsp_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Total Distance"])
        for k, v in stats.items():
            writer.writerow([k, v])

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "TSP Results Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for k, v in stats.items():
        pdf.cell(0, 10, f"{k}: Distance = {v:.2f}", ln=True)
    for name in stats.keys():
        img = f"{name}_best_tour.png"
        if os.path.exists(img):
            pdf.add_page()
            pdf.image(img, w=180)
    pdf.output("TSP_Report.pdf")


def checker(myCheckerOutput:dict={"":""})->dict:

    sorted_dictionary = dict(sorted(myCheckerOutput.items() , key=operator.itemgetter(1)))
    return sorted_dictionary



start_timer = time.time()

among_patcher = r"D:\lesson\highest bachelor\Term 2\special topics in software engineering1\homework\part1\project1\mySounds\among\Project_3.wav"
starting_patcher = r"D:\lesson\highest bachelor\Term 2\special topics in software engineering1\homework\part1\project1\mySounds\becoming\Project_1.wav"
ending_patcher = r"D:\lesson\highest bachelor\Term 2\special topics in software engineering1\homework\part1\project1\mySounds\ending\Project_2.wav"


print("\n\n the Project start...!!!\n\n")

tsp_cities = load_tsp_data("kroB200.txt")

initial_tour = list(range(1, len(tsp_cities) + 1))
random.shuffle(initial_tour)

stats , results , myCheckerOutput = run_all_algorithm_here(tsp_cities, initial_tour , among_patcher , starting_patcher , ending_patcher)

save_report(stats)

print("\n\n i can create a PDF reporter for this project successfully...!!!")

print(f"\n\n my results are : {results}.\n\n ")

bestDistanceChecker = checker(myCheckerOutput)

print(f"\n\n my best distances checker are : {bestDistanceChecker}. \n\n")

print(f"\n\nTotal Execution Time: {time.time() - start_timer:.2f} seconds")

print("\n\n the process is finished successfully...!!!")

