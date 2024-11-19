import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.animation as animation

class DynamicFieldAnalyzer:
    def __init__(self, field_size=100):
        self.field_size = field_size
        self.kernels = {
            'Default': self.create_kernel(6.0, 5.0, 4.0, 10.0),
            'Strong Competition': self.create_kernel(6.0, 5.0, 5.5, 10.0),
            'Weak Competition': self.create_kernel(6.0, 5.0, 2.5, 10.0),
            'Wide Interaction': self.create_kernel(6.0, 8.0, 4.0, 16.0)
        }
        
    def create_kernel(self, exc_amp, exc_width, inh_amp, inh_width):
        """Create Mexican hat kernel with given parameters"""
        x = np.arange(self.field_size)
        kernel = (exc_amp * np.exp(-(x - self.field_size/2)**2 / (2 * exc_width**2)) -
                 inh_amp * np.exp(-(x - self.field_size/2)**2 / (2 * inh_width**2)))
        return np.roll(kernel, self.field_size//2)
    
    def simulate_pattern_formation(self, kernel_type, n_steps=1000, dt=0.1,
                                 tau=10.0, h=-5.0, beta=4.0):
        """Simulate field evolution with given kernel"""
        # Initialize field with random noise
        u = h * np.ones(self.field_size) + np.random.normal(0, 0.1, self.field_size)
        
        # Create stimulus (two Gaussian inputs)
        x = np.arange(self.field_size)
        stimulus = (10.0 * np.exp(-(x - 30)**2 / (2 * 3.0**2)) +
                   8.0 * np.exp(-(x - 70)**2 / (2 * 3.0**2)))
        
        # Storage for results
        results = np.zeros((n_steps, self.field_size))
        
        # Run simulation
        kernel = self.kernels[kernel_type]
        for t in range(n_steps):
            # Sigmoid activation
            activation = 1.0 / (1.0 + np.exp(-beta * u))
            
            # Field dynamics
            interaction = np.convolve(activation, kernel, mode='same')
            du = (-u + h + interaction + stimulus) / tau
            u += dt * du
            
            results[t] = u
            
        return results
    
    def analyze_stability(self, results):
        """Analyze stability of pattern formation"""
        # Calculate change between successive timesteps
        differences = np.diff(results, axis=0)
        stability_measure = np.mean(np.abs(differences), axis=1)
        
        # Calculate peak properties
        peak_positions = []
        peak_amplitudes = []
        
        for t in range(results.shape[0]):
            peaks = self.find_peaks(results[t])
            peak_positions.append(len(peaks))
            peak_amplitudes.append(np.mean([results[t][p] for p in peaks]) if peaks else 0)
            
        return stability_measure, peak_positions, peak_amplitudes
    
    def find_peaks(self, u, threshold=0):
        """Find peaks in activation field"""
        peaks = []
        for i in range(1, len(u)-1):
            if u[i] > threshold and u[i] > u[i-1] and u[i] > u[i+1]:
                peaks.append(i)
        return peaks

    def visualize_pattern_formation(self, results, kernel_type):
        """Create visualization of pattern formation"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot initial state
        axes[0].plot(results[0], 'b-', label='Initial')
        axes[0].set_title(f'Initial State - {kernel_type}')
        axes[0].grid(True)
        
        # Plot final state
        axes[1].plot(results[-1], 'r-', label='Final')
        axes[1].set_title('Final State')
        axes[1].grid(True)
        
        # Plot spatiotemporal evolution
        im = axes[2].imshow(results.T, aspect='auto', origin='lower',
                           extent=[0, results.shape[0], 0, results.shape[1]])
        axes[2].set_title('Spatiotemporal Evolution')
        axes[2].set_xlabel('Time step')
        axes[2].set_ylabel('Field position')
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        return fig
    
    def compare_stability(self):
        """Compare stability across different kernels"""
        stability_data = {}
        
        for kernel_type in self.kernels.keys():
            # Run simulation
            results = self.simulate_pattern_formation(kernel_type)
            
            # Analyze stability
            stability_measure, peak_positions, peak_amplitudes = self.analyze_stability(results)
            
            stability_data[kernel_type] = {
                'stability': stability_measure,
                'peaks': peak_positions,
                'amplitudes': peak_amplitudes
            }
            
        # Plot stability comparison
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot stability measures
        for kernel_type, data in stability_data.items():
            axes[0].plot(data['stability'], label=kernel_type)
        axes[0].set_title('Stability Measure Over Time')
        axes[0].set_xlabel('Time step')
        axes[0].set_ylabel('Average change')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot number of peaks
        for kernel_type, data in stability_data.items():
            axes[1].plot(data['peaks'], label=kernel_type)
        axes[1].set_title('Number of Peaks Over Time')
        axes[1].set_xlabel('Time step')
        axes[1].set_ylabel('Number of peaks')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig, stability_data

# Run analysis
if __name__ == "__main__":
    analyzer = DynamicFieldAnalyzer()
    
    # Compare pattern formation across different kernels
    for kernel_type in analyzer.kernels.keys():
        results = analyzer.simulate_pattern_formation(kernel_type)
        fig = analyzer.visualize_pattern_formation(results, kernel_type)
        plt.show()
    
    # Compare stability
    fig, stability_data = analyzer.compare_stability()
    plt.show()