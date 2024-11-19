import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

class DynamicFieldWithFatigue:
    def __init__(self, size=100, tau=8.0, h=-5.0, beta=8.0, 
                 tau_fatigue=20.0, alpha_fatigue=10.0):
        """
        Initialize Dynamic Neural Field with stronger fatigue dynamics
        """
        self.size = size
        self.tau = tau
        self.h = h  # Raised resting level to allow for more visible fatigue effects
        self.beta = beta
        
        # Strengthened fatigue parameters
        self.tau_fatigue = tau_fatigue
        self.alpha_fatigue = alpha_fatigue
        
        self.x = np.linspace(-10, 10, size)
        self.dx = self.x[1] - self.x[0]
        
        self.u = np.zeros(size)
        self.fatigue = np.zeros(size)
        
        # Adjusted interaction kernel for stronger local excitation
        self.w = self._create_interaction_kernel(
            exc_width=1.5,
            exc_strength=15.0,  # Increased excitation
            inh_width=8.0,
            inh_strength=8.0    # Reduced inhibition to allow multiple peaks
        )
    
    def sigmoid(self, u):
        return 1 / (1 + np.exp(-self.beta * u))
    
    def _create_interaction_kernel(self, exc_width, exc_strength, inh_width, inh_strength):
        x = np.arange(-self.size//2, self.size//2)
        excitation = exc_strength * np.exp(-0.5 * (x/exc_width)**2)
        inhibition = -inh_strength * np.exp(-0.5 * (x/inh_width)**2)
        w = excitation + inhibition
        return w - np.mean(w)
    
    def compute_interaction(self, u):
        f_u = self.sigmoid(u)
        return convolve1d(f_u, self.w, mode='wrap') * self.dx

def run_fatigue_comparison():
    # Create figure
    fig = plt.figure(figsize=(20, 25))
    
    # Define more extreme parameter combinations
    fatigue_params = [
        {'tau_f': 10.0, 'alpha_f': 8.0, 'title': 'Fast, Moderate Fatigue'},
        {'tau_f': 10.0, 'alpha_f': 20.0, 'title': 'Fast, Strong Fatigue'},
        {'tau_f': 50.0, 'alpha_f': 8.0, 'title': 'Slow, Moderate Fatigue'},
        {'tau_f': 50.0, 'alpha_f': 20.0, 'title': 'Slow, Strong Fatigue'}
    ]
    
    for idx, params in enumerate(fatigue_params):
        # Create subplot grid
        ax1 = plt.subplot(4, 2, 2*idx + 1)
        ax2 = plt.subplot(4, 2, 2*idx + 2)
        
        # Create field with current parameters
        field = DynamicFieldWithFatigue(
            tau_fatigue=params['tau_f'],
            alpha_fatigue=params['alpha_f']
        )
        
        # Create stronger inputs
        x = field.x
        
        # Stronger task input (three gaussians)
        S_task = (3.0 * np.exp(-0.5 * ((x - (-2))/1.0)**2) +  # Narrower, stronger peaks
                 3.0 * np.exp(-0.5 * ((x - 0)/1.0)**2) +
                 3.0 * np.exp(-0.5 * ((x - 2)/1.0)**2))
        
        # Stronger specific input
        S_spec = 6.0 * np.exp(-0.5 * ((x - 6)/0.8)**2)
        
        # Simulation parameters
        T = 100.0
        dt = 0.1
        t_spec = 5.0
        
        # Run simulation
        t = np.arange(0, T, dt)
        n_steps = len(t)
        
        # Storage for simulation results
        u_history = np.zeros((n_steps, field.size))
        fatigue_history = np.zeros((n_steps, field.size))
        u_history[0] = field.u
        fatigue_history[0] = field.fatigue
        
        # Create total input with periodic pulses
        S = np.tile(S_task, (n_steps, 1))
        
        # Add periodic pulses of specific input
        pulse_period = 20  # Time between pulses
        pulse_duration = 5  # Duration of each pulse
        for start_time in np.arange(t_spec, T, pulse_period):
            idx_start = int(start_time / dt)
            idx_end = min(idx_start + int(pulse_duration / dt), n_steps)
            S[idx_start:idx_end] += S_spec
        
        # Simulate with fatigue dynamics
        for i in range(1, n_steps):
            u_curr = u_history[i-1]
            fatigue_curr = fatigue_history[i-1]
            
            interaction = field.compute_interaction(u_curr)
            
            # Update activation (with stronger fatigue effect)
            du_dt = (-u_curr + S[i-1] + field.h + interaction - 
                    field.alpha_fatigue * fatigue_curr) / field.tau
            
            # Update fatigue (faster buildup)
            dfatigue_dt = (-fatigue_curr + field.sigmoid(u_curr)) / field.tau_fatigue
            
            u_history[i] = u_curr + dt * du_dt
            fatigue_history[i] = fatigue_curr + dt * dfatigue_dt
        
        # Plot field evolution
        im1 = ax1.imshow(u_history.T, aspect='auto', origin='lower',
                        extent=[t[0], t[-1], x[0], x[-1]],
                        cmap='RdBu_r', vmin=-10, vmax=10)
        ax1.set_title(f"{params['title']}\nField Evolution\n(τ_f={params['tau_f']}, α_f={params['alpha_f']})")
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Space')
        plt.colorbar(im1, ax=ax1, label='Activation')
        
        # Plot fatigue evolution
        im2 = ax2.imshow(fatigue_history.T, aspect='auto', origin='lower',
                        extent=[t[0], t[-1], x[0], x[-1]],
                        cmap='Oranges', vmin=0, vmax=1)
        ax2.set_title(f"{params['title']}\nFatigue Evolution")
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Space')
        plt.colorbar(im2, ax=ax2, label='Fatigue')
    
    plt.tight_layout()
    plt.show()
    
    # Create time series plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Use strong fatigue parameters for time series
    field = DynamicFieldWithFatigue(tau_fatigue=10.0, alpha_fatigue=20.0)
    
    # Track points near input peaks
    points = [int(field.size * 0.4), int(field.size * 0.7)]  # Near -2 and 6
    colors = ['b', 'r']
    labels = ['Near Task Input', 'Near Specific Input']
    
    for point, color, label in zip(points, colors, labels):
        ax1.plot(t, u_history[:, point], color=color, label=f'Activation {label}')
        ax2.plot(t, fatigue_history[:, point], color=color, label=f'Fatigue {label}')
    
    ax1.set_title('Activation Time Series at Key Points')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Activation')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Fatigue Time Series at Key Points')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Fatigue')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run the comparison
run_fatigue_comparison()