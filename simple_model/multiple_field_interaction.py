import numpy as np
import matplotlib.pyplot as plt

class InteractingFields:
    def __init__(self, n_points=180):
        # Movement parameter field (e.g., directions 0° to 360°)
        self.directions = np.linspace(0, 360, n_points)
        
        # Task parameter field (e.g., target locations or valid movement regions)
        self.task_field = np.zeros_like(self.directions)
        
        # Movement preparation field
        self.movement_field = np.zeros_like(self.directions)
        
    def gaussian_bump(self, x, center, width=20, amplitude=5):
        """Create a Gaussian bump of activation"""
        return amplitude * np.exp(-(x - center)**2 / (2*width**2))
    
    def update_task_field(self, valid_directions):
        """Update task field to represent valid movement directions"""
        self.task_field = np.zeros_like(self.directions)
        for direction in valid_directions:
            self.task_field += self.gaussian_bump(self.directions, direction, width=30)
    
    def compute_movement_field(self, target_direction):
        
        """Compute movement field influenced by task constraints"""
        # Direct input from target
        target_input = self.gaussian_bump(self.directions, target_direction)
        
        # Movement field is shaped by both target input and task constraints
        self.movement_field = target_input * (1 + 0.5 * self.task_field)
        return self.movement_field

fields = InteractingFields()
# Example simulation

# Set up plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# 1. Task field: Define valid movement regions (e.g., 90° and 270°)
fields.update_task_field([90, 270])
axes[0].plot(fields.directions, fields.task_field)
axes[0].set_title('Task Parameter Field\n(Valid Movement Regions)')
axes[0].set_ylabel('Activation')

# 2. Input for target at 75°
target_input_1 = fields.gaussian_bump(fields.directions, 75)
target_input_2 = fields.gaussian_bump(fields.directions, 175)
axes[1].plot(fields.directions, target_input_1)
axes[1].plot(fields.directions, target_input_2)
axes[1].set_title('Target Input\n(Target at 75° and 175°)')
axes[1].set_ylabel('Activation')

# 3. Resulting movement field
movement_field_1 = fields.compute_movement_field(75)
movement_field_2 = fields.compute_movement_field(175)
axes[2].plot(fields.directions, movement_field_1)
axes[2].plot(fields.directions, movement_field_2)

axes[2].set_title('Movement Parameter Field\n(Influenced by Task Constraints)')
axes[2].set_ylabel('Activation')
axes[2].set_xlabel('Movement Direction (degrees)')

for ax in axes:
    ax.grid(True)

plt.tight_layout()
plt.show()

# Now show how it changes with a different target
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Compare responses to targets at different locations
target_directions = [75, 260]  # One near 90°, one near 270°
for target in target_directions:
    movement_field = fields.compute_movement_field(target)
    axes[0].plot(fields.directions, movement_field, 
                 label=f'Target at {target}°')

axes[0].set_title('Movement Fields for Different Targets')
axes[0].set_xlabel('Movement Direction (degrees)')
axes[0].set_ylabel('Activation')
axes[0].grid(True)
axes[0].legend()

# Now show time evolution for a target
time_points = np.linspace(0, 2, 100)
field_evolution = np.zeros((len(time_points), len(fields.directions)))

for t_idx, t in enumerate(time_points):
    if t >= 0.5:  # Start input at t=0.5
        field_evolution[t_idx] = fields.compute_movement_field(75)
    # Add simple dynamics
    if t_idx > 0:
        dt = time_points[1] - time_points[0]
        field_evolution[t_idx] = field_evolution[t_idx-1] + \
            dt/0.3 * (-field_evolution[t_idx-1] + field_evolution[t_idx])

axes[1].imshow(field_evolution, aspect='auto', origin='lower', 
               extent=[0, 360, 0, 2])
axes[1].set_title('Field Evolution Over Time\n(Target at 75°)')
axes[1].set_xlabel('Movement Direction (degrees)')
axes[1].set_ylabel('Time (s)')
plt.colorbar(axes[1].images[0], ax=axes[1], label='Activation')

plt.tight_layout()
plt.show()