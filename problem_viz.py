import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the functions
def ackley(x, y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    # Find the absolute minimum of each function

# Ackley function minimum
ackley_min = minimize(lambda v: ackley(v[0], v[1]), [0, 0], bounds=[(-5, 5), (-5, 5)])
ackley_min_x, ackley_min_y = ackley_min.x

# Rosenbrock function minimum
rosenbrock_min = minimize(lambda v: rosenbrock(v[0], v[1]), [0, 0], bounds=[(-5, 5), (-5, 5)])
rosenbrock_min_x, rosenbrock_min_y = rosenbrock_min.x

# Himmelblau function minimum
himmelblau_min_points = []
for start_point in [(-4, 4), (4, 4), (-4, -4), (4, -4)]:
    himmelblau_min = minimize(lambda v: himmelblau(v[0], v[1]), start_point, bounds=[(-5, 5), (-5, 5)])
    himmelblau_min_points.append(himmelblau_min.x)

# Extract unique Himmelblau minima
himmelblau_min_points = np.unique(np.round(himmelblau_min_points, decimals=5), axis=0)
himmelblau_min_x, himmelblau_min_y = himmelblau_min_points[:, 0], himmelblau_min_points[:, 1]

# Create mesh grid
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

# Compute function values
Z_ackley = ackley(X, Y)
Z_rosenbrock = rosenbrock(X, Y)
Z_himmelblau = himmelblau(X, Y)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Ackley function
contour1 = axes[0].contourf(X, Y, Z_ackley, levels=50, cmap='viridis')
fig.colorbar(contour1, ax=axes[0])
axes[0].set_title("Ackley Function")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")

# Rosenbrock function
contour2 = axes[1].contourf(X, Y, Z_rosenbrock, levels=50, cmap='plasma')
fig.colorbar(contour2, ax=axes[1])
axes[1].set_title("Rosenbrock Function")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")

# Himmelblau function
contour3 = axes[2].contourf(X, Y, Z_himmelblau, levels=50, cmap='coolwarm')
fig.colorbar(contour3, ax=axes[2])
axes[2].set_title("Himmelblau Function")
axes[2].set_xlabel("X")
axes[2].set_ylabel("Y")

# Add minimum points to the plot
# Ackley function minimum
axes[0].plot(ackley_min_x, ackley_min_y, 'r*', markersize=10, label='Ackley Min')
axes[0].legend()

# Rosenbrock function minimum
axes[1].plot(rosenbrock_min_x, rosenbrock_min_y, 'r*', markersize=10, label='Rosenbrock Min')
axes[1].legend()

# Himmelblau function minima
for x_min, y_min in zip(himmelblau_min_x, himmelblau_min_y):
    axes[2].plot(x_min, y_min, 'r*', markersize=10, label='Himmelblau Min')
axes[2].legend()

plt.tight_layout()
plt.savefig('/home/yaadreb/GitHub/CPL/function_visualizations.png', dpi=300)
plt.show()

# 3D Plotting
fig_3d = plt.figure(figsize=(18, 5))

# Ackley function 3D plot
ax1 = fig_3d.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z_ackley, cmap='viridis', edgecolor='none', alpha=0.8)
ax1.set_title("Ackley Function")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.scatter(ackley_min_x, ackley_min_y, ackley(ackley_min_x, ackley_min_y), color='r', s=50, label='Ackley Min')
ax1.legend()

# Rosenbrock function 3D plot
ax2 = fig_3d.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_rosenbrock, cmap='plasma', edgecolor='none', alpha=0.8)
ax2.set_title("Rosenbrock Function")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.scatter(rosenbrock_min_x, rosenbrock_min_y, rosenbrock(rosenbrock_min_x, rosenbrock_min_y), color='r', s=50, label='Rosenbrock Min')
ax2.legend()

# Himmelblau function 3D plot
ax3 = fig_3d.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z_himmelblau, cmap='coolwarm', edgecolor='none', alpha=0.8)
ax3.set_title("Himmelblau Function")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")
for x_min, y_min in zip(himmelblau_min_x, himmelblau_min_y):
    ax3.scatter(x_min, y_min, himmelblau(x_min, y_min), color='r', s=50, label='Himmelblau Min')
ax3.legend()

plt.tight_layout()
plt.show()