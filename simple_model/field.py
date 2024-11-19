import numpy as np
import matplotlib.pyplot as plt

# Create a field over movement directions (0° to 360°)
directions = np.linspace(0, 360, 180)  # movement parameter space
field = np.zeros_like(directions)      # activation at each direction

# Example: Input at 90° creates a "bump" of activation
def gaussian_input(x, center=90, width=20):
    return 5 * np.exp(-(x - center)**2 / (2*width**2))

field = gaussian_input(directions)

# Visualize the field
plt.figure(figsize=(10, 6))
plt.plot(directions, field)
plt.xlabel('Movement Direction (degrees)')
plt.ylabel('Activation u(x)')
plt.title('Activation Field: "Bump" of Activity around 90°')
plt.grid(True)
plt.show()

# Now simulate how the field evolves over time
def simulate_field_evolution(directions, time_points):
    # Initialize field over space and time
    field = np.zeros((len(time_points), len(directions)))
    
    # Add input starting at t=1.0s
    for t_idx, t in enumerate(time_points):
        if t >= 1.0:
            field[t_idx] = gaussian_input(directions)
            
    # Simple dynamics (just for visualization)
    tau = 0.3
    dt = time_points[1] - time_points[0]
    for t in range(1, len(time_points)):
        field[t] = field[t-1] + dt/tau * (-field[t-1] + field[t])
    
    return field

# Create space and time points
time_points = np.linspace(0, 2, 100)
field_evolution = simulate_field_evolution(directions, time_points)

# Visualize the evolution
plt.figure(figsize=(10, 6))
plt.imshow(field_evolution, aspect='auto', origin='lower')
plt.colorbar(label='Activation')
plt.xlabel('Movement Direction (degrees)')
plt.ylabel('Time')
plt.title('Evolution of Activation Field Over Time')
plt.show()