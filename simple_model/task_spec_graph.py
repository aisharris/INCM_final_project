import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

class DynamicField:
    def __init__(self, size=100, tau=8.0, h=-11.0, beta=8.0):
        """
        Initialize Dynamic Neural Field with stronger competition parameters
        """
        self.size = size
        self.tau = tau  # Increased time constant
        self.h = h      # Lower resting level for stronger competition
        self.beta = beta  # Steeper sigmoid for sharper competition
        
        # Create spatial dimension
        self.x = np.linspace(-10, 10, size)
        self.dx = self.x[1] - self.x[0]
        
        # Initialize activation
        self.u = np.zeros(size)
        
        # Create interaction kernel with stronger lateral inhibition
        self.w = self._create_interaction_kernel(
            exc_width=1.5,      # Narrower excitation
            exc_strength=15.0,  # Stronger local excitation
            inh_width=100.0,      # Wider inhibition
            inh_strength=0.0   # Stronger inhibition
        )
    
    def sigmoid(self, u):
        """Steeper sigmoid threshold function"""
        return 1 / (1 + np.exp(-self.beta * u))
    
    def _create_interaction_kernel(self, exc_width, exc_strength, inh_width, inh_strength):
        """Create interaction kernel with stronger competition"""
        x = np.arange(-self.size//2, self.size//2)
        
        # Local excitation (narrower gaussian)
        excitation = exc_strength * np.exp(-0.5 * (x/exc_width)**2)
        
        # Global inhibition (broader gaussian)
        inhibition = -inh_strength * np.exp(-0.5 * (x/inh_width)**2)
        
        # Combine and normalize
        w = excitation + inhibition
        return w - np.mean(w)
    
    def compute_interaction(self, u):
        """Compute field interaction"""
        f_u = self.sigmoid(u)
        return convolve1d(f_u, self.w, mode='wrap') * self.dx
    
    def simulate(self, T, dt, S_task, S_spec=None, t_spec=None, noise_std=0.2):
        t = np.arange(0, T, dt)
        n_steps = len(t)
        
        # Storage for simulation results
        u_history = np.zeros((n_steps, self.size))
        u_history[0] = self.u
        
        # Create total input with stronger specific input
        S = np.tile(S_task, (n_steps, 1))
        if S_spec is not None and t_spec is not None:
            idx_spec = int(t_spec / dt)
            S[idx_spec:] += S_spec
        
        # Simulate
        for i in range(1, n_steps):
            interaction = self.compute_interaction(u_history[i-1])
            du_dt = (-u_history[i-1] + S[i-1] + self.h + interaction) / self.tau
            # Add noise with standard deviation noise_std
            du_dt += np.random.normal(scale=noise_std, size=self.size)
            u_history[i] = u_history[i-1] + dt * du_dt
        
        return t, u_history
    
    def plot_simulation(self, t, u_history, S_task, S_spec):
        """Enhanced visualization"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Field evolution
        im = ax1.imshow(u_history.T, aspect='auto', origin='lower',
                       extent=[t[0], t[-1], self.x[0], self.x[-1]],
                       cmap='RdBu_r')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Space')
        ax1.set_title('Field Evolution')
        plt.colorbar(im, ax=ax1, label='Activation')
        
        # Inputs
        ax2.plot(self.x, S_task, 'g--', label='Task Input')
        ax2.plot(self.x, S_spec, 'r--', label='Specific Input')
        ax2.set_xlabel('Space')
        ax2.set_ylabel('Input Strength')
        ax2.set_title('Input Distribution')
        ax2.legend()
        
        # Final field state
        ax3.plot(self.x, u_history[-1], 'b-', label='Final Field')
        ax3.plot(self.x, S_task, 'g--', label='Task Input')
        ax3.plot(self.x, S_spec, 'r--', label='Specific Input')
        ax3.set_xlabel('Space')
        ax3.set_ylabel('Activation')
        ax3.set_title('Final Field State')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage with distant specific input
def run_simulation():
    # Create field
    field = DynamicField()

    
    # Create task input (three gaussians centered around x=0)
    x = field.x
    S_task = (1.5 * np.exp(-0.5 * ((x - (-2))/3.5)**2) +
              1.5 * np.exp(-0.5 * ((x - 0)/3.5)**2) +
              1.5 * np.exp(-0.5 * ((x - 2)/3.5)**2))

    
    # Create specific input (far from task inputs at x=6)
    S_spec = 4.0 * np.exp(-0.5 * ((x - 6)/0.8)**2)
    
    # Simulation parameters
    T = 30.0  # Longer simulation time
    dt = 0.1
    t_spec = 5.0
    
    # Run simulation
    t, u_history = field.simulate(T, dt, S_task, S_spec, t_spec)
    
    # Plot results
    field.plot_simulation(t, u_history, S_task, S_spec)
    
    return field, t, u_history

# Run simulation
field, t, u_history = run_simulation()