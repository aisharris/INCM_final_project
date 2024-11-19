import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.animation as animation

class DynamicField:
    def __init__(self, field_size=100, tau=8.0, h=-5.0, beta=4.0):
        """
        Initialize Dynamic Field
        field_size: number of units in the field
        tau: time constant
        h: resting level
        beta: slope of sigmoid function
        """
        self.size = field_size
        self.tau = tau
        self.h = h
        self.beta = beta
        
        # Initialize activation field
        self.u = self.h * np.ones(field_size)
        
        # Interaction kernel (Mexican hat)
        x = np.arange(field_size)
        self.kernel = (6.0 * np.exp(-(x - field_size/2)**2 / (2 * (5.0)**2)) - 
                      12.0 * np.ones_like(x))
        self.kernel = np.roll(self.kernel, field_size//2)

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-self.beta * x))

    def interaction(self, u):
        """Compute interaction within the field"""
        return np.convolve(self.sigmoid(u), self.kernel, mode='same')

    def step(self, stimulus, dt=0.1):
        """
        Evolve field for one timestep
        stimulus: external input to the field
        dt: time step size
        """
        # Dynamic field equation
        du = (-self.u + self.h + self.interaction(self.u) + stimulus) / self.tau
        self.u += dt * du
        return self.u

def create_stimulus(field_size, positions, strengths):
    """Create Gaussian stimuli at specified positions with given strengths"""
    stimulus = np.zeros(field_size)
    for pos, strength in zip(positions, strengths):
        x = np.arange(field_size)
        stimulus += strength * np.exp(-(x - pos)**2 / (2 * (3.0)**2))
    return stimulus

def run_simulation(field_size=100, n_steps=1000):
    """Run full simulation and return results"""
    # Initialize field
    field = DynamicField(field_size)
    
    # Create stimulus
    stimulus_positions = [25, 45]  # Two competing stimuli
    stimulus_strengths = [5.0, 6.0]
    stimulus = create_stimulus(field_size, stimulus_positions, stimulus_strengths)
    
    # Storage for results
    results = np.zeros((n_steps, field_size))
    
    # Run simulation
    for t in range(n_steps):
        results[t] = field.step(stimulus)
        
    return results, stimulus

def plot_results(results, stimulus):
    """Plot simulation results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot final field state
    ax1.plot(results[-1], 'b-', label='Final activation')
    ax1.plot(stimulus, 'r--', label='Stimulus')
    ax1.set_title('Final Field State')
    ax1.set_xlabel('Field position')
    ax1.set_ylabel('Activation')
    ax1.legend()
    
    # Plot spatiotemporal evolution
    im = ax2.imshow(results.T, aspect='auto', origin='lower', 
                    extent=[0, results.shape[0], 0, results.shape[1]])
    ax2.set_title('Spatiotemporal Evolution')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Field position')
    plt.colorbar(im, ax=ax2, label='Activation')
    
    plt.tight_layout()
    return fig

def create_animation(results):
    """Create animation of field evolution"""
    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot([], [], 'b-')
    
    ax.set_xlim(0, results.shape[1])
    ax.set_ylim(np.min(results), np.max(results))
    ax.set_xlabel('Field position')
    ax.set_ylabel('Activation')
    
    def init():
        line.set_data([], [])
        return line,
    
    def animate(i):
        line.set_data(np.arange(results.shape[1]), results[i])
        return line,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=results.shape[0], interval=20,
                                 blit=True)
    return anim

# Run example simulation
if __name__ == "__main__":
    # Run simulation
    results, stimulus = run_simulation()
    
    # Create and save plots
    fig = plot_results(results, stimulus)
    plt.show()
    
    # Create and save animation
    anim = create_animation(results)
    plt.show()