import matplotlib.pyplot as plt
import numpy as np

# Read data from file
data = np.loadtxt('iterations.txt')
n_values = data[:, 0]
iterations = data[:, 1]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(n_values, iterations, 'b-o', linewidth=2, markersize=4)
plt.xlabel('Problem Size (n)', fontsize=12)
plt.ylabel('Number of Iterations', fontsize=12)
plt.title('Iterative Improvement: Iterations vs Problem Size', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('iterations_vs_n.png', dpi=300)

# Print some statistics
print(f"Min iterations: {int(np.min(iterations))} (at n={int(n_values[np.argmin(iterations)])})")
print(f"Max iterations: {int(np.max(iterations))} (at n={int(n_values[np.argmax(iterations)])})")
print(f"Mean iterations: {np.mean(iterations):.2f}")
print(f"Median iterations: {np.median(iterations):.2f}")

plt.show()

