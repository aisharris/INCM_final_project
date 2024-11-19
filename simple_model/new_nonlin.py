import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class NonlinearDynamicField:
    def __init__(self, n_points=180, tau=7.0, h=-5.0, beta=20.0, gamma=0.01):
        # Field parameters
        self.directions = np.linspace(0, 360, n_points)
        self.tau = tau          # Time constant
        self.h = h              # Resting level
        self.beta = beta        # Steepness of sigmoid
        self.gamma = gamma      # Sharpness of bistable transition
        
        # Initialize fields
        self.task_field = np.zeros_like(self.directions)
        self.movement_field = np.zeros_like(self.directions)
    
    def sigmoid(self, u):
        """Nonlinear activation function f[u]"""
        return 1.0 / (1.0 + np.exp(-self.beta * u))
    
    def sharpened_sigmoid(self, u):
        """Modified sigmoid with sharper bistable transition"""
        return 1.0 / (1.0 + np.exp(-self.beta * (u - self.gamma)))
    
    def gaussian_input(self, x, center, width=20, amplitude=10):
        return amplitude * np.exp(-(x - center)**2 / (2*width**2))
    
    def field_dynamics(self, u, t, S):
        """Implements the full dynamic field equation with nonlinearity"""
        # du/dt = (-u + h + S + f[u])/tau
        return (-u + self.h + S + self.sharpened_sigmoid(u)) / self.tau
    
    def simulate_decision(self, target_direction, t_points):
        # Create input
        S = self.gaussian_input(self.directions, target_direction)
        
        # Initialize field
        u0 = np.zeros_like(self.directions)
        
        # Simulate each point in space separately
        field_evolution = np.zeros((len(t_points), len(self.directions)))
        
        for i in range(len(self.directions)):
            # Solve ODE for this spatial point
            solution = odeint(self.field_dynamics, u0[i], t_points, args=(S[i],))
            field_evolution[:, i] = solution.flatten()
            
        return field_evolution

# Create instance and simulate
field = NonlinearDynamicField(gamma=2.0)
t_points = np.linspace(0, 20, 200)

# Simulate with different input strengths
target_dir = 90
field_evolution_strong = field.simulate_decision(target_dir, t_points)

# Visualize the nonlinear dynamics
fig, axes = plt.subplots(2, 2, figsize=(12, 7))

# Plot 1: Show sharpened sigmoid function
u_range = np.linspace(-10, 10, 100)
axes[0,0].plot(u_range, field.sharpened_sigmoid(u_range))
axes[0,0].set_title('Sharpened Sigmoid Function f[u]')
axes[0,0].set_xlabel('Activation u')
axes[0,0].set_ylabel('f[u]')
axes[0,0].grid(True)

# Plot 2: Show rate of change (du/dt) vs u for different inputs
u_test = np.linspace(-10, 10, 100)
for S in [0, 6.5, 12]:
    du_dt = field.field_dynamics(u_test, 0, S)
    axes[0,1].plot(u_test, du_dt, label=f'Input = {S}')
axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0,1].set_title('Rate of Change vs. Activation\nfor Different Inputs')
axes[0,1].set_xlabel('Activation u')
axes[0,1].set_ylabel('du/dt')
axes[0,1].set_xlim(1, 3)  # Zoom in on x-axis
axes[0,1].set_ylim(-2, 2)  # Zoom in on y-axis
axes[0,1].legend()
axes[0,1].grid(True)
axes[0,1].grid(True, which='minor', linestyle=':', alpha=0.4)
axes[0,1].minorticks_on()

# Plot 3: Evolution of field over time
im = axes[1,0].imshow(field_evolution_strong, aspect='auto', origin='lower',
                      extent=[0, 360, 0, t_points[-1]])
axes[1,0].set_title('Field Evolution Over Time')
axes[1,0].set_xlabel('Direction (degrees)')
axes[1,0].set_ylabel('Time')
plt.colorbar(im, ax=axes[1,0], label='Activation')

# Plot 4: Activation profiles at different times
time_indices = [0, 50, 100, -1]
for idx in time_indices:
    axes[1,1].plot(field.directions, field_evolution_strong[idx], 
                   label=f't = {t_points[idx]:.1f}')
axes[1,1].set_title('Activation Profiles at Different Times')
axes[1,1].set_xlabel('Direction (degrees)')
axes[1,1].set_ylabel('Activation')
axes[1,1].legend()
axes[1,1].grid(True)

plt.tight_layout()
plt.show()

# Simulate from different initial conditions
u_init_values = [-3, 0, 3]  # Different initial activations
t_short = np.linspace(0, 50, 100)
S_weak = 2  # Weak input
S_mid = 6.5
S_strong = 9

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Weak input
for u_init in u_init_values:
    solution = odeint(field.field_dynamics, u_init, t_short, args=(S_weak,))
    ax[0].plot(t_short, solution, label=f'Initial u = {u_init}')
ax[0].set_title(f'Bistable Behavior: Weak Input (S = {S_weak})')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Activation')
ax[0].set_xlim(0, 30)  # Zoom in on x-axis
ax[0].set_ylim(-3.5, 3.5)  # Zoom in on y-axis
ax[0].legend()
ax[0].grid(True)

# Mid input  
for u_init in u_init_values:
    solution = odeint(field.field_dynamics, u_init, t_short, args=(S_mid,))
    ax[1].plot(t_short, solution, label=f'Initial u = {u_init}')
ax[1].set_title(f'Bistable Behavior: Medium Input (S = {S_mid})')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Activation')
ax[1].set_xlim(0, 30)  # Zoom in on x-axis
ax[1].set_ylim(-4, 4)  # Zoom in on y-axis
ax[1].legend()
ax[1].grid(True)

# Strong input
for u_init in u_init_values:
    solution = odeint(field.field_dynamics, u_init, t_short, args=(S_strong,))
    ax[2].plot(t_short, solution, label=f'Initial u = {u_init}')
ax[2].set_title(f'Bistable Behavior: Strong Input (S = {S_strong})')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Activation')
ax[2].set_xlim(0, 30)  # Zoom in on x-axis
ax[2].set_ylim(-5.5, 5.5)  # Zoom in on y-axis
ax[2].legend()
ax[2].grid(True)

plt.tight_layout()
plt.show()