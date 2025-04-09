import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import random
from scipy.stats import linregress

class WildfireSimulation:
    def __init__(self, grid_size=20, max_burn_time=3, ignite_random=True):
        self.grid_size = grid_size
        self.max_burn_time = max_burn_time
        
        # Grid states: 0 = unburned, 1 = burning, 2 = burned out
        self.grid = np.zeros((grid_size, grid_size))
        self.burn_time = np.zeros((grid_size, grid_size))

        # Environmental factors
        self.soil_dryness = np.random.uniform(0, 1, (grid_size, grid_size))
        self.vegetation_density = np.random.uniform(0, 1, (grid_size, grid_size))
        self.air_humidity = np.random.uniform(0, 1, (grid_size, grid_size))
        self.shade_coverage = np.random.uniform(0, 1, (grid_size, grid_size))
        
        # Wind parameters (in km/h)
        self.wind_speed = random.uniform(0, 30)  # Random wind speed between 0-30 km/h
        self.wind_direction = random.uniform(0, 2 * np.pi)  # Random wind direction in radians
        
        if ignite_random:
            self.ignite()
        else:
            self.grid[0, 0] = 1  # Start fire at (0, 0)

    def ignite(self):
        x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
        self.grid[x, y] = 1

    def calculate_fire_probability(self, i, j):
        """Calculate probability of a cell catching fire based on environmental factors."""
        base_prob = 0.2
        
        # Environmental influences
        soil_factor = self.soil_dryness[i, j]
        vegetation_factor = self.vegetation_density[i, j]
        humidity_factor = 1 - self.air_humidity[i, j]
        shade_factor = 1 - 0.5 * self.shade_coverage[i, j]
        
        # Check adjacent cells and calculate wind influence
        adjacent_burning = 0
        wind_influence = 1.0  # Default no wind influence
        
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check 4 directions
            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                if self.grid[ni, nj] == 1:  # If neighbor is burning
                    adjacent_burning += 1
                    
                    # Calculate angle between wind and fire spread direction
                    # Note: wind_direction is the direction wind is coming FROM
                    # So we need to invert the fire direction to match
                    fire_direction = np.arctan2(-dj, -di)  # Invert the direction
                    angle_diff = abs(fire_direction - self.wind_direction)
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    
                    # Wind influence:
                    # - Increases with speed and alignment with wind direction
                    # - Decreases when spreading against the wind
                    # - Maximum influence when spreading with wind (angle_diff = 0)
                    # - Minimum influence when spreading against wind (angle_diff = pi)
                    wind_factor = np.cos(angle_diff)  # 1.0 for with wind, -1.0 for against wind
                    wind_influence = max(wind_influence, 1 + (self.wind_speed / 30) * wind_factor)
        
        # Return 0 probability if no burning neighbors
        if adjacent_burning == 0:
            return 0.0
            
        probability = base_prob * (1 + soil_factor) * (1 + vegetation_factor) * \
                     (humidity_factor) * (shade_factor) * (1 + 0.2 * adjacent_burning) * \
                     wind_influence
        return min(probability, 1.0)

    def update(self):
        """Update the grid state for one timestep."""
        new_grid = np.copy(self.grid)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 0:  # unburned
                    prob = self.calculate_fire_probability(i, j)
                    if random.random() < prob:
                        new_grid[i, j] = 1  # ignite
                elif self.grid[i, j] == 1:  # burning
                    self.burn_time[i, j] += 1
                    # Calculate dynamic burn out time based on vegetation and humidity
                    vegetation_factor = self.vegetation_density[i, j]
                    humidity_factor = 1 - self.air_humidity[i, j]
                    # More vegetation = longer burn time, more humidity = shorter burn time
                    dynamic_burn_time = self.max_burn_time * (1 + vegetation_factor) * (1 - 0.5 * humidity_factor)
                    if self.burn_time[i, j] >= dynamic_burn_time:
                        new_grid[i, j] = 2  # burned out
        
        self.grid = new_grid

    def run_simulation(self, steps=20):
        """Run simulation without visualization (for Monte Carlo analysis)."""
        for _ in range(steps):
            self.update()
        burned_cells = np.sum(self.grid == 2)
        total_cells = self.grid_size * self.grid_size
        burn_ratio = burned_cells / total_cells
        return burn_ratio

    def visualize(self):
        """Visualize the simulation using Matplotlib."""
        cmap = mcolors.ListedColormap(['green', 'red', 'black'])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        img = ax.imshow(self.grid, cmap=cmap, norm=norm)
        
        # Convert wind direction from radians to degrees
        wind_deg = np.degrees(self.wind_direction)
        
        # Convert to cardinal direction
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        index = int((wind_deg + 22.5) / 45) % 8
        cardinal_direction = directions[index]
        
        # Add wind information to the plot
        wind_text = f"Wind: {cardinal_direction} ({self.wind_speed:.1f} km/h)"
        plt.title(wind_text)

        def animate(frame):
            self.update()
            img.set_data(self.grid)
            return [img]

        ani = FuncAnimation(fig, animate, frames=50, interval=500, repeat=True)
        plt.show()

def monte_carlo_simulation(trials=100, grid_size=20):
    """Run Monte Carlo simulations and collect data."""
    results = {
        'soil_dryness': [],
        'vegetation_density': [],
        'air_humidity': [],
        'shade_coverage': [],
        'wind_speed': [],
        'burn_ratio': []
    }

    for _ in range(trials):
        sim = WildfireSimulation(grid_size=grid_size, ignite_random=False)
        avg_soil = np.mean(sim.soil_dryness)
        avg_veg = np.mean(sim.vegetation_density)
        avg_hum = np.mean(sim.air_humidity)
        avg_shade = np.mean(sim.shade_coverage)
        wind_speed = sim.wind_speed
        burn_ratio = sim.run_simulation(steps=20)

        results['soil_dryness'].append(avg_soil)
        results['vegetation_density'].append(avg_veg)
        results['air_humidity'].append(avg_hum)
        results['shade_coverage'].append(avg_shade)
        results['wind_speed'].append(wind_speed)
        results['burn_ratio'].append(burn_ratio)

    return results

def visualize_results(results):
    """Create visualizations to analyze relationships between environmental factors and burn ratio."""
    # Set up the figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Impact of Environmental Factors on Wildfire Spread', fontsize=16)
    
    # Flatten the axes array for easier iteration
    axs = axs.flatten()
    
    factors = ['soil_dryness', 'vegetation_density', 'air_humidity', 'shade_coverage', 'wind_speed']
    factor_names = ['Soil Dryness', 'Vegetation Density', 'Air Humidity', 'Shade Coverage', 'Wind Speed']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#d62728']
    
    # Create scatter plots with regression lines for each factor
    for i, (factor, name, color) in enumerate(zip(factors, factor_names, colors)):
        x = results[factor]
        y = results['burn_ratio']
        
        # Calculate regression line
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Create scatter plot
        axs[i].scatter(x, y, color=color, alpha=0.6, label=f'Data Points')
        
        # Plot regression line
        line_x = np.linspace(min(x), max(x), 100)
        line_y = slope * line_x + intercept
        axs[i].plot(line_x, line_y, color='red', linestyle='--', 
                   label=f'Regression Line\nR² = {r_value**2:.3f}')
        
        # Add labels and legend
        axs[i].set_xlabel(name)
        axs[i].set_ylabel('Burn Ratio')
        axs[i].set_title(f'Effect of {name} on Wildfire Spread')
        axs[i].legend()
        axs[i].grid(True, linestyle='--', alpha=0.7)
        
        # Annotate with slope and R-squared
        text = f"Slope: {slope:.3f}\nR²: {r_value**2:.3f}\np-value: {p_value:.4f}"
        axs[i].text(0.05, 0.95, text, transform=axs[i].transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create a summary bar plot in the last subplot showing relative importance
    importance = []
    factor_labels = []
    
    for factor, name in zip(factors, factor_names):
        _, _, r_value, _, _ = linregress(results[factor], results['burn_ratio'])
        importance.append(abs(r_value))
        factor_labels.append(name)
    
    # Sort by importance
    sorted_indices = np.argsort(importance)[::-1]  # descending order
    sorted_importance = [importance[i] for i in sorted_indices]
    sorted_labels = [factor_labels[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    axs[5].barh(range(len(sorted_labels)), sorted_importance, color=sorted_colors)
    axs[5].set_yticks(range(len(sorted_labels)))
    axs[5].set_yticklabels(sorted_labels)
    axs[5].set_xlabel('Absolute Correlation Coefficient (|R|)')
    axs[5].set_title('Relative Importance of Factors')
    axs[5].set_xlim(0, 1)
    
    # Add correlation values as text
    for i, value in enumerate(sorted_importance):
        axs[5].text(value + 0.01, i, f'{value:.3f}', va='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.show()

def analyze_multi_trial_simulation(num_trials=5, mc_trials=100, grid_size=20):
    """Run multiple Monte Carlo simulations and aggregate results to reduce randomness."""
    all_results = None
    
    for _ in range(num_trials):
        results = monte_carlo_simulation(trials=mc_trials, grid_size=grid_size)
        
        if all_results is None:
            all_results = results
        else:
            for key in results:
                all_results[key].extend(results[key])
    
    print(f"Completed {num_trials} sets of Monte Carlo simulations with {mc_trials} trials each.")
    print(f"Total data points: {len(all_results['burn_ratio'])}")
    
    return all_results

def create_heatmap_analysis(results):
    """Create a heatmap of correlations between all variables."""
    # Convert results to numpy arrays
    factors = ['soil_dryness', 'vegetation_density', 'air_humidity', 'shade_coverage', 'wind_speed', 'burn_ratio']
    data = np.array([results[factor] for factor in factors]).T
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data.T)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Correlation Coefficient', rotation=-90, va="bottom")
    
    # Add ticks and labels
    factor_names = ['Soil Dryness', 'Vegetation Density', 'Air Humidity', 'Shade Coverage', 'Wind Speed', 'Burn Ratio']
    ax.set_xticks(np.arange(len(factor_names)))
    ax.set_yticks(np.arange(len(factor_names)))
    ax.set_xticklabels(factor_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(factor_names)
    
    # Add correlation values in the cells
    for i in range(len(factor_names)):
        for j in range(len(factor_names)):
            text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                          ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.7 else "white")
    
    plt.title("Correlation Matrix of Environmental Factors and Burn Ratio")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run a single visualization example
    print("Running a single wildfire visualization example...")
    sim = WildfireSimulation(grid_size=20)
    sim.visualize()
    
    # Run Monte Carlo simulations with more trials for better statistical analysis
    print("\nRunning Monte Carlo simulations to analyze factor influence...")
    all_results = analyze_multi_trial_simulation(num_trials=3, mc_trials=100, grid_size=20)
    
    # Create visualizations
    print("\nCreating visualization of results...")
    visualize_results(all_results)
    
    # Create correlation heatmap
    print("\nCreating correlation heatmap...")
    create_heatmap_analysis(all_results)