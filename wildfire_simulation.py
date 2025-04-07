import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import random
import time

class WildfireSimulation:
    def __init__(self, grid_size=6, max_burn_time=3):
        self.grid_size = grid_size
        self.max_burn_time = max_burn_time
        
        # Initialize grid states: 0 = unburned, 1 = burning, 2 = burned out
        self.grid = np.zeros((grid_size, grid_size))
        
        # Initialize environmental factors
        self.soil_dryness = np.random.uniform(0, 1, (grid_size, grid_size))
        self.vegetation_density = np.random.uniform(0, 1, (grid_size, grid_size))
        self.wind_direction = np.random.uniform(0, 2*np.pi, (grid_size, grid_size))
        self.wind_speed = np.random.uniform(0, 1, (grid_size, grid_size))
        
        # Initialize burn time tracking
        self.burn_time = np.zeros((grid_size, grid_size))
        
        # Set initial fire
        self.grid[0, 0] = 1
        
    def calculate_fire_probability(self, i, j):
        """Calculate the probability of a cell catching fire based on environmental factors"""
        base_prob = 0.2  # Reduced base probability
        
        # Environmental factors influence
        soil_factor = self.soil_dryness[i, j]
        vegetation_factor = self.vegetation_density[i, j]
        
        # Check adjacent burning cells
        adjacent_burning = 0
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                if self.grid[ni, nj] == 1:  # If adjacent cell is burning
                    adjacent_burning += 1
        
        # Calculate final probability
        probability = base_prob * (1 + soil_factor) * (1 + vegetation_factor) * (1 + 0.1 * adjacent_burning)
        return min(probability, 1.0)
    
    def update(self):
        """Update the grid state for one time step"""
        # Create a copy of the grid to store new states
        new_grid = self.grid.copy()
        
        # First, update burn times for currently burning cells
        self.burn_time[self.grid == 1] += 1
        
        # Cells that have burned for max_burn_time become burned out
        new_grid[self.burn_time >= self.max_burn_time] = 2
        
        # Create a list of cells to potentially catch fire
        potential_fire_cells = []
        
        # First pass: identify cells that might catch fire
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 0:  # If cell is unburned
                    if random.random() < self.calculate_fire_probability(i, j):
                        potential_fire_cells.append((i, j))
        
        # Second pass: update only the cells that were identified
        for i, j in potential_fire_cells:
            new_grid[i, j] = 1
        
        self.grid = new_grid
    
    def show_environmental_factors(self):
        """Display the environmental factors before starting the simulation"""
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        
        # Soil Dryness
        im1 = axs[0, 0].imshow(self.soil_dryness, cmap='YlOrRd', vmin=0, vmax=1)
        axs[0, 0].set_title('Soil Dryness')
        plt.colorbar(im1, ax=axs[0, 0])
        
        # Vegetation Density
        im2 = axs[0, 1].imshow(self.vegetation_density, cmap='Greens', vmin=0, vmax=1)
        axs[0, 1].set_title('Vegetation Density')
        plt.colorbar(im2, ax=axs[0, 1])
        
        # Wind Speed
        im3 = axs[1, 0].imshow(self.wind_speed, cmap='Blues', vmin=0, vmax=1)
        axs[1, 0].set_title('Wind Speed')
        plt.colorbar(im3, ax=axs[1, 0])
        
        # Wind Direction (using HSV colormap for direction)
        hsv = np.zeros((self.grid_size, self.grid_size, 3))
        hsv[..., 0] = self.wind_direction / (2 * np.pi)  # Hue
        hsv[..., 1] = 1  # Saturation
        hsv[..., 2] = 1  # Value
        rgb = mcolors.hsv_to_rgb(hsv)
        axs[1, 1].imshow(rgb)
        axs[1, 1].set_title('Wind Direction')
        
        plt.tight_layout()
        plt.show()
        time.sleep(3)  # Pause for 3 seconds to view the environmental factors

def run_simulation(steps=10):
    """Run the simulation for a specified number of steps"""
    sim = WildfireSimulation()
    
    # Show environmental factors first
    sim.show_environmental_factors()
    
    # Create figure for animation
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = mcolors.ListedColormap(['green', 'red', 'black'])
    bounds = [0, 1, 2, 3]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    def update(frame):
        ax.clear()
        sim.update()
        img = ax.imshow(sim.grid, cmap=cmap, norm=norm)
        ax.set_title(f'Wildfire Simulation - Step {frame + 1}')
        plt.pause(1)  # Pause for 1 second between frames
        return [img]
    
    anim = FuncAnimation(fig, update, frames=steps, interval=1000, blit=True)
    plt.show()

if __name__ == "__main__":
    run_simulation(steps=10) 