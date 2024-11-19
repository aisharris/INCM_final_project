import numpy as np
import matplotlib.pyplot as plt

def simulate_HH_like(t, I_stim=0):
    # Highly simplified HH-like spike
    V = np.zeros_like(t)
    V[:] = -70  # resting potential
    # Add a spike when stimulated
    spike_idx = (t >= 5) & (t <= 5.5)
    V[spike_idx] = 30
    return V

def simulate_DFT_like(t, x_preferred=90):
    # Simplified DFT population response
    u = np.zeros_like(t)
    tau = 1.0
    h = -5.0
    
    # Input when stimulus matches preferred direction
    S = np.zeros_like(t)
    S[t >= 5] = 10.0
    
    # Simulate population activity
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        du = (-u[i-1] + h + S[i])/tau
        u[i] = u[i-1] + du * dt
    return u

# Create time points
t_HH = np.linspace(0, 10, 1000)  # milliseconds for HH
t_DFT = np.linspace(0, 10, 1000) # seconds for DFT

# Simulate both
V = simulate_HH_like(t_HH)
u = simulate_DFT_like(t_DFT)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(t_HH, V)
ax1.set_title('HH-like Single Neuron Response')
ax1.set_ylabel('Membrane Potential (mV)')
ax1.grid(True)

ax2.plot(t_DFT, u)
ax2.set_title('DFT Population Response')
ax2.set_ylabel('Population Activation')
ax2.set_xlabel('Time')
ax2.grid(True)

plt.tight_layout()
plt.show()