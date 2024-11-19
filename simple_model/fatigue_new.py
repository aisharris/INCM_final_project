import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.animation as animation

class BiologicalDynamicField:
    def __init__(self, field_size=100):
        """
        Initialize Dynamic Field with biological parameters
        
        Parameters based on typical neural time constants and firing rates:
        - tau_u: membrane time constant (~10-20ms)
        - tau_f: fatigue time constant (~100ms)
        - h: resting potential (-65mV normalized)
        - beta: firing rate gain
        """
        # Field parameters
        self.size = field_size
        self.tau_u = 0.01  # 10ms membrane time constant
        self.tau_f = 0.1   # 100ms fatigue time constant
        self.h = -65.0     # resting potential (mV)
        self.beta = 0.1    # firing rate gain
        
        # Synaptic parameters
        self.w_exc = 15.0  # excitatory synaptic strength
        self.w_inh = 10.0  # inhibitory synaptic strength
        self.sigma_exc = 5.0  # excitatory spread
        self.sigma_inh = 10.0 # inhibitory spread
        
        # Initialize fields
        self.u = self.h * np.ones(field_size)  # membrane potential
        self.f = np.zeros(field_size)          # fatigue variable
        
        # Create interaction kernel
        self.kernel = self._create_biological_kernel()
        
        # Storage for analysis
        self.history = {
            'u': [],
            'f': [],
            'firing_rate': []
        }
    
    def _create_biological_kernel(self):
        """Create interaction kernel based on synaptic connections"""
        x = np.arange(self.size)
        x = np.minimum(x, self.size - x)  # Consider periodic boundary
        
        # Mexican hat based on excitatory and inhibitory synaptic spread
        kernel = (self.w_exc * np.exp(-x**2 / (2 * self.sigma_exc**2)) -
                 self.w_inh * np.exp(-x**2 / (2 * self.sigma_inh**2)))
        return kernel - np.mean(kernel)  # normalize
    
    def firing_rate(self, v):
        """Convert membrane potential to firing rate using gain function"""
        return 1.0 / (1.0 + np.exp(-self.beta * (v - self.h)))
    
    def step(self, stimulus, dt=0.001):
        """
        Evolve field for one timestep
        
        Parameters:
        stimulus: external input (current)
        dt: timestep in seconds
        """
        # Current firing rate
        r = self.firing_rate(self.u)
        
        # Synaptic input
        syn_input = np.convolve(r, self.kernel, mode='same')
        
        # Membrane potential dynamics (including fatigue)
        du = (-self.u + self.h + syn_input + stimulus - self.f) / self.tau_u
        
        # Fatigue dynamics
        df = (-self.f + r) / self.tau_f
        
        # Update states
        self.u += dt * du
        self.f += dt * df
        
        # Store history
        self.history['u'].append(self.u.copy())
        self.history['f'].append(self.f.copy())
        self.history['firing_rate'].append(r.copy())
        
        return self.u, self.f, r

def run_motor_preparation_experiment(duration=1.0, dt=0.001):
    """
    Run motor preparation experiment
    
    Parameters:
    duration: simulation duration in seconds
    dt: timestep in seconds
    """
    # Initialize field
    field = BiologicalDynamicField()
    n_steps = int(duration / dt)
    
    # Create time-varying stimulus (e.g., two competing motor plans)
    t = np.arange(n_steps) * dt
    stimulus = np.zeros((n_steps, field.size))
    
    # First motor plan (appears at t=0.1s)
    stim1_onset = int(0.1 / dt)
    stimulus[stim1_onset:, 30] = 20.0  # Strong stimulus at position 30
    
    # Second motor plan (appears at t=0.3s)
    stim2_onset = int(0.3 / dt)
    stimulus[stim2_onset:, 70] = 25.0  # Stronger stimulus at position 70
    
    # Run simulation
    for i in range(n_steps):
        field.step(stimulus[i], dt)
    
    return field, stimulus, t

def analyze_results(field, stimulus, t):
    """Analyze and visualize results"""
    # Convert histories to arrays
    u_history = np.array(field.history['u'])
    f_history = np.array(field.history['f'])
    r_history = np.array(field.history['firing_rate'])
    
    # Create figures
    fig, axes = plt.subplots(4, 1, figsize=(12, 15))
    
    # Plot membrane potential evolution
    im1 = axes[0].imshow(u_history.T, aspect='auto', origin='lower',
                        extent=[0, t[-1], 0, field.size])
    axes[0].set_title('Membrane Potential Evolution')
    axes[0].set_ylabel('Field position')
    plt.colorbar(im1, ax=axes[0], label='mV')
    
    # Plot firing rate evolution
    im2 = axes[1].imshow(r_history.T, aspect='auto', origin='lower',
                        extent=[0, t[-1], 0, field.size])
    axes[1].set_title('Firing Rate Evolution')
    axes[1].set_ylabel('Field position')
    plt.colorbar(im2, ax=axes[1], label='Hz')
    
    # Plot fatigue evolution
    im3 = axes[2].imshow(f_history.T, aspect='auto', origin='lower',
                        extent=[0, t[-1], 0, field.size])
    axes[2].set_title('Fatigue Evolution')
    axes[2].set_ylabel('Field position')
    plt.colorbar(im3, ax=axes[2], label='a.u.')
    
    # Plot time courses at stimulus positions
    axes[3].plot(t, r_history[:, 30], 'b-', label='Motor plan 1')
    axes[3].plot(t, r_history[:, 70], 'r-', label='Motor plan 2')
    axes[3].set_title('Firing Rate Time Courses')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Firing rate (Hz)')
    axes[3].legend()
    
    plt.tight_layout()
    return fig

# Run experiment
if __name__ == "__main__":
    # Run simulation
    field, stimulus, t = run_motor_preparation_experiment()
    
    # Analyze and plot results
    fig = analyze_results(field, stimulus, t)
    plt.show()