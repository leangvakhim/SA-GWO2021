import matplotlib.pyplot as plt
import numpy as np
from wsn import wsn
from sa_gwo import sa_gwo


width = 100.0
height = 100.0
num_sensors = 50
radius = 10.0

pop_size = 30
max_iter = 1000
t_start = 1000.0
cooling_rate = 0.98

wsn_model = wsn(width, height, num_sensors, radius, grid_resolution=1.5)

optimizer = sa_gwo(wsn_model, pop_size, max_iter, t_start, cooling_rate)

best_solution, history = optimizer.optimize()

# plot 1: convergence curve
plt.figure(figsize=(10, 5))
plt.plot(history, 'r-', linewidth=2)
plt.title('Convergence Curve (SA-GWO)')
plt.xlabel('Iteration')
plt.ylabel('Coverage Rate')
plt.grid(True)
plt.show()

# plot 2: sensor distribution
best_coords = best_solution.reshape((num_sensors, 2))

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_aspect('equal')

for i in best_coords:
    circle = plt.Circle((i[0], i[1]), radius, color='r', alpha=0.2)
    ax.add_artist(circle)
    ax.plot(i[0], i[1], 'k.', markersize=4)

plt.title(f"Deployment (coverage: {history[-1]*100 :.2f})")
plt.xlabel('Width (m)')
plt.ylabel('Height (m)')
plt.grid(True, linestyle='--')
plt.show()