import numpy as np
import matplotlib.pyplot as plt


## Minimize the objective
def objective_function(x):
    y = np.where((x < -1) | (x > 1), 0, np.cos(50 * x) + np.sin(20 * x))
    return y


def simulated_annealing(objective, X, T0, alpha, max_iter):
    x = np.random.choice(X)
    T = T0
    history = []
    best_x = x
    best_f = objective(x)

    for i in range(max_iter):
        new_x = np.clip(x + np.random.normal(scale=0.1), -1, 1)
        new_f = objective(new_x)

        delta_E = new_f - objective(x)
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            x = new_x

        if objective(x) < best_f:
            best_x, best_f = x, objective(x)

        history.append(x)
        T *= alpha  # Reduce temperature

        if T < 1e-3:
            break  # Stop if temperature is too low

    return best_x, best_f, history


X = np.linspace(-1, 1, num=1000)
T0 = 100
alpha = 0.95
max_iter = 100
best_x, best_f, history = simulated_annealing(
    objective_function, X, T0, alpha, max_iter
)
history = np.asarray(history, dtype=np.float64)


plt.plot(X, objective_function(X))
plt.scatter(best_x, objective_function(best_x), marker="x", color="orange")
plt.plot(history[::5], objective_function(history[::5]))
plt.show()
