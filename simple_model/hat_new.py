import numpy as np
import matplotlib.pyplot as plt

def create_interaction_kernel(field_size, exc_amplitude=6.0, exc_width=5.0, 
                            inh_amplitude=4.0, inh_width=10.0):
    """
    Create Mexican hat interaction kernel
    
    Parameters:
    field_size: size of the field
    exc_amplitude: amplitude of excitatory Gaussian
    exc_width: width of excitatory Gaussian
    inh_amplitude: amplitude of inhibitory Gaussian
    inh_width: width of inhibitory Gaussian
    """
    x = np.arange(field_size)
    kernel = (exc_amplitude * np.exp(-(x - field_size/2)**2 / (2 * exc_width**2)) - 
             inh_amplitude * np.exp(-(x - field_size/2)**2 / (2 * inh_width**2)))
    return np.roll(kernel, field_size//2)

def plot_kernel_components(field_size=100):
    """Plot the excitatory and inhibitory components separately and combined"""
    x = np.arange(field_size)
    
    # Create components
    excitatory = 6.0 * np.exp(-(x - field_size/2)**2 / (2 * 5.0**2))
    inhibitory = 4.0 * np.exp(-(x - field_size/2)**2 / (2 * 10.0**2))
    combined = excitatory - inhibitory
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Components
    ax1.plot(x, excitatory, 'g-', label='Excitatory')
    ax1.plot(x, -inhibitory, 'r-', label='Inhibitory')
    ax1.set_title('Kernel Components')
    ax1.legend()
    ax1.grid(True)
    
    # Combined Mexican hat
    ax2.plot(x, combined, 'b-', label='Mexican Hat')
    ax2.set_title('Combined Interaction Kernel')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def demonstrate_kernel_effect():
    """Demonstrate how the kernel affects activation patterns"""
    field_size = 100
    
    # Create initial activation pattern (two Gaussian bumps)
    x = np.arange(field_size)
    activation = (np.exp(-(x - 30)**2 / (2 * 3.0**2)) + 
                 np.exp(-(x - 70)**2 / (2 * 3.0**2)))
    
    # Apply kernel
    kernel = create_interaction_kernel(field_size)
    interaction = np.convolve(activation, kernel, mode='same')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(x, activation, 'b-', label='Initial Activation')
    ax1.set_title('Initial Activation Pattern')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(x, interaction, 'r-', label='After Interaction')
    ax2.set_title('Effect of Interaction Kernel')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def experiment_with_kernels():
    """Compare different kernel parameters"""
    field_size = 100
    
    kernels = {
        'Default': create_interaction_kernel(field_size, 6.0, 5.0, 4.0, 10.0),
        'Strong Excitation': create_interaction_kernel(field_size, 8.0, 5.0, 4.0, 10.0),
        'Wide Inhibition': create_interaction_kernel(field_size, 6.0, 5.0, 4.0, 15.0),
        'Narrow Interaction': create_interaction_kernel(field_size, 6.0, 3.0, 4.0, 6.0)
    }
    
    fig, axes = plt.subplots(len(kernels), 1, figsize=(10, 12))
    
    for (name, kernel), ax in zip(kernels.items(), axes):
        ax.plot(kernel, 'b-')
        ax.set_title(f'Kernel: {name}')
        ax.grid(True)
    
    plt.tight_layout()
    return fig

# Run demonstrations
if __name__ == "__main__":
    # Plot kernel components
    fig1 = plot_kernel_components()
    plt.show()
    
    # Demonstrate kernel effect
    fig2 = demonstrate_kernel_effect()
    plt.show()
    
    # Compare different kernels
    fig3 = experiment_with_kernels()
    plt.show()