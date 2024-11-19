import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class EnhancedDFTModel:
    def __init__(self, field_size=100, dt=0.01, tau=10):
        self.field_size = field_size
        self.dt = dt
        self.tau = tau
        
        # Original parameters from the paper
        self.h = -5  # Resting level
        self.beta = 1.5  # Steepness of sigmoid
        
        # New parameters for enhancements
        self.noise_strength = 0.1
        self.adaptation_rate = 0.05
        self.learning_rate = 0.01
        
        # Initialize fields
        self.x = np.linspace(-50, 50, field_size)
        self.u = np.zeros(field_size)  # Activation field
        self.w = self.create_interaction_kernel()
        self.adaptation = np.zeros(field_size)  # Adaptation field
        self.memory_trace = np.zeros(field_size)  # Memory trace field
        
    def create_interaction_kernel(self):
        """Create interaction kernel with local excitation and lateral inhibition"""
        x_diff = np.subtract.outer(self.x, self.x)
        w_exc = 15 * np.exp(-0.5 * (x_diff/5)**2)
        w_inh = 5 * np.exp(-0.5 * (x_diff/10)**2)
        return w_exc - w_inh
    
    def sigmoid(self, u):
        """Nonlinear activation function"""
        return 1 / (1 + np.exp(-self.beta * u))
    
    def compute_field_dynamics(self, u, t, S):
        """Enhanced field dynamics with adaptation and memory"""
        # Original interaction integral
        interaction = np.dot(self.w, self.sigmoid(u))
        
        # New features
        noise = self.noise_strength * np.random.randn(self.field_size)
        adaptation_term = -self.adaptation
        memory_input = self.learning_rate * self.memory_trace
        
        # Combined dynamics
        du = (-u + self.h + S + interaction + adaptation_term + 
              memory_input + noise) / self.tau
        
        return du
    
    def update_adaptation(self):
        """Update adaptation field"""
        self.adaptation += self.dt * self.adaptation_rate * (
            self.sigmoid(self.u) - self.adaptation)
    
    def update_memory_trace(self):
        """Update memory trace"""
        self.memory_trace += self.dt * (
            self.sigmoid(self.u) - self.memory_trace)
    
    def simulate_step(self, S):
        """Simulate one time step with all enhancements"""
        # Update main field
        self.u += self.dt * self.compute_field_dynamics(self.u, 0, S)
        
        # Update additional fields
        self.update_adaptation()
        self.update_memory_trace()
        
    def prepare_movement(self, target_position, simulation_time):
        """Simulate movement preparation to target"""
        time_points = np.arange(0, simulation_time, self.dt)
        results = []
        
        # Create target input
        S = 10 * np.exp(-0.5 * ((self.x - target_position)/5)**2)
        
        # Run simulation
        for t in time_points:
            self.simulate_step(S)
            results.append(self.u.copy())
            
        return np.array(results), time_points
    
    def analyze_preparation_dynamics(self, results, time_points):
        """Analyze the preparation process"""
        # Peak detection
        max_activations = np.max(results, axis=1)
        peak_positions = np.argmax(results, axis=1)
        
        # Stability analysis
        stability_measure = np.std(peak_positions[-int(len(time_points)/5):])
        
        return {
            'max_activations': max_activations,
            'peak_positions': peak_positions,
            'stability': stability_measure
        }