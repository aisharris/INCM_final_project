import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DynamicField:
    def __init__(self, field_size=100, tau=10.0, h=-5.0, beta=4.0):
        self.size = field_size
        self.tau = tau
        self.h = h
        self.beta = beta
        
        # Initialize activation field
        self.u = self.h * np.ones(field_size)
        
        # Interaction kernel with local excitation and global inhibition
        x = np.arange(field_size)
        self.kernel = (10.0 * np.exp(-(x - field_size/2)**2 / (2 * (8.0)**2)) - 
                      20.0 * np.ones_like(x))
        self.kernel = np.roll(self.kernel, field_size//2)

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-self.beta * x))

    def interaction(self, u):
        """Compute interaction within the field"""
        return np.convolve(self.sigmoid(u), self.kernel, mode='same')

    def create_task_input(self, positions, strength=0.8):
        """Create broad task input representing task environment"""
        stimulus = np.zeros(self.size)
        for pos in positions:
            x = np.arange(self.size)
            # Broader Gaussian for task input
            stimulus += strength * np.exp(-(x - pos)**2 / (2 * (10.0)**2))
        return stimulus

    def create_specific_input(self, position, strength=1.4, width=3.0):
        """Create specific input representing sensory information"""
        x = np.arange(self.size)
        return strength * np.exp(-(x - position)**2 / (2 * width**2))

    def step(self, task_input, specific_input, dt=0.1):
        """Evolve field for one timestep with both inputs"""
        total_input = task_input + specific_input
        du = (-self.u + self.h + self.interaction(self.u) + total_input) / self.tau
        self.u += dt * du
        return self.u

def run_simulation(field_size=100, n_steps=500):
    # Initialize field
    field = DynamicField(field_size)
    
    # Create task input (three overlapping positions)
    task_positions = [35, 45, 55]  # Three nearby positions
    task_input = field.create_task_input(task_positions)
    
    # Create specific input (activated at t=0)
    specific_position = 90
    specific_input = field.create_specific_input(specific_position)
    
    # Storage for results
    results = np.zeros((n_steps, field_size))
    
    # Run simulation
    for t in range(n_steps):
        # Only add specific input after t=0 (halfway through simulation)
        current_specific = specific_input if t > n_steps//2 else np.zeros_like(specific_input)
        results[t] = field.step(task_input, current_specific)
    
    return results, task_input, specific_input

def plot_simulation(results, task_input, specific_input):
    """Plot the complete simulation results"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot final field state
    ax1.plot(results[-1], 'b-', label='Final activation')
    ax1.set_title('Final Field State')
    ax1.set_xlabel('Field position')
    ax1.set_ylabel('Activation')
    ax1.legend()
    
    # Plot inputs
    ax2.plot(task_input, 'g-', label='Task input')
    ax2.plot(specific_input, 'r-', label='Specific input')
    ax2.set_title('Inputs')
    ax2.set_xlabel('Field position')
    ax2.set_ylabel('Input strength')
    ax2.legend()
    
    # Plot spatiotemporal evolution
    im = ax3.imshow(results.T, aspect='auto', origin='lower',
                    extent=[0, results.shape[0], 0, results.shape[1]])
    ax3.set_title('Spatiotemporal Evolution')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Field position')
    plt.colorbar(im, ax=ax3, label='Activation')
    
    plt.tight_layout()
    return fig

# Run simulation
if __name__ == "__main__":
    results, task_input, specific_input = run_simulation()
    fig = plot_simulation(results, task_input, specific_input)
    plt.show()