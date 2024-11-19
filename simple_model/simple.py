import numpy as np
import matplotlib.pyplot as plt

# Simple case: Just one direction (x fixed)
def simulate_single_direction(t_points, tau=1.0, h=-5.0):
    # Initial activation
    u = np.zeros_like(t_points)
    
    # Input: Target appears at t=1.0
    S = np.zeros_like(t_points)
    S[t_points >= 1.0] = 10.0  # Strong input after t=1
    
    # Simulate system over time
    dt = t_points[1] - t_points[0]
    for i in range(1, len(t_points)):
        # Change in u = (-current_u + resting_level + input) / tau
        du = (-u[i-1] + h + S[i]) / tau
        u[i] = u[i-1] + du * dt
    
    return u, S

# Create time points
t = np.linspace(0, 3, 300)
u, S = simulate_single_direction(t)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, u, label='Activation (u)')
plt.plot(t, S, '--', label='Input (S)')
plt.axhline(y=0, color='k', linestyle=':')
plt.grid(True)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Activation')
plt.title('Simple Movement Preparation')
plt.show()