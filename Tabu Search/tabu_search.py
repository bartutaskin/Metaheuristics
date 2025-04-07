import random
import matplotlib.pyplot as plt
from typing import List

# Knapsack problem
SEED = 43
NUMBER_OF_ITEMS = 20

# random.seed(SEED)

weights = [random.randint(1, 20) for i in range(NUMBER_OF_ITEMS)]
values = [random.randint(10, 100) for i in range(NUMBER_OF_ITEMS)]
capacity = int(sum(weights) * 0.4)

print(f"Weights: {weights}")
print(f"Values: {values}")
print(f"Capacity: {capacity}")

def objective_function(solution: List[int]) -> int:
    total_weight = 0
    total_value = 0
    for idx in range(NUMBER_OF_ITEMS):
        if solution[idx] == 1:
            if total_weight + weights[idx] < capacity:
                total_weight += weights[idx]
                total_value += values[idx]   
            else:
                return -1

    return total_value

def get_neighborhood_solution(solution: List[int]) -> List[int]:
    random_idx = random.randint(0, NUMBER_OF_ITEMS-1)
    neighbor_sol = solution.copy()
    if neighbor_sol[random_idx] == 1:
        neighbor_sol[random_idx] = 0
    else:
        neighbor_sol[random_idx] = 1
    
    return neighbor_sol

def tabu_search(initial_solution, iterations, tabu_size=5):
    best_f = objective_function(initial_solution)
    best_s = initial_solution
    tabu_list = [best_s]
    f_list = [best_f]

    for i in range(iterations):
        neighbor_sol = get_neighborhood_solution(best_s)
        new_f = objective_function(neighbor_sol)

        if neighbor_sol in tabu_list:
            # If it is in the tabu list, apply aspiration criterion
            if new_f > best_f:
                best_f = new_f
                best_s = neighbor_sol
                tabu_list.append(best_s)
        else:
            # If it is not in the tabu list, accept it if it is better
            if new_f > best_f:
                best_f = new_f
                best_s = neighbor_sol
                tabu_list.append(best_s)
        f_list.append(best_f)
        
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return best_f, best_s, f_list

def calculate_weight(solution):
    total = 0
    for idx in range(NUMBER_OF_ITEMS):
        total += solution[idx] * weights[idx]
    
    return total

def generate_feasible_inital_solution():
    while True:
        initial_solution = [random.randint(0, 1) for i in range(NUMBER_OF_ITEMS)]
        if objective_function(initial_solution) != -1:
            return initial_solution

initial_solution = generate_feasible_inital_solution()
print(f"Initial Objective Function: {objective_function(initial_solution)}")
best_f, best_s, f_list = tabu_search(initial_solution=initial_solution, iterations=100)
print(f"Best Objective Function: {best_f}")
print(f"Best Solution: {best_s}")
print(f"Total Weight: {calculate_weight(best_s)}")

plt.plot(f_list)
plt.xticks(ticks=[i for i in range(0, len(f_list), max(1, len(f_list)//10))])
plt.title("Tabu Search Progress")
plt.xlabel("Iteration")
plt.ylabel("Best Total Value")
plt.grid(True)
plt.show()
