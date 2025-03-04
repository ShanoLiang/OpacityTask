import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time

# Directory to save plots and parameters
SAVE_DIR = "generated_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_smooth_curve(num_points=100, freq_factors=None, amp_factors=None, x_shift=0.0, y_shift=0.0, noise_level=0.0, wave_combinations=None):
    """
    Generate a smooth random curved line with multiple wave combinations.

    Parameters:
    - num_points: Number of points for the curve.
    - freq_factors: List of frequency multipliers for wave components.
    - amp_factors: List of amplitude multipliers for wave components.
    - x_shift: Value to shift the curve along the x-axis.
    - y_shift: Value to shift the curve along the y-axis.
    - noise_level: Standard deviation of random noise.
    - wave_combinations: List of wave combinations for each frequency component.

    Returns:
    - x: X values.
    - y: Y values.
    """
    if freq_factors is None:
        freq_factors = [1, 2, 3]
    if amp_factors is None:
        amp_factors = [1, 0.5, 0.25]
    if wave_combinations is None:
        wave_combinations = [['sin', 'sinc'], ['cos', 'tanh'], ['sin', 'exp']]

    x = np.linspace(0, 10, num_points)
    y = np.zeros_like(x)

    for a, f, waves in zip(amp_factors, freq_factors, wave_combinations):
        for w in waves:
            if w == 'sin':
                y += a * np.sin(f * (x + x_shift))
            elif w == 'cos':
                y += a * np.cos(f * (x + x_shift))
            elif w == 'sinc':
                y += a * np.sinc(f * ((x + x_shift)))
            elif w == 'tanh':
                y += a * np.tanh(f * ((x + x_shift)))
            elif w == 'exp':
                y += a * np.exp(-0.5 * f * ((x + x_shift)))

    y += (np.random.normal(scale=noise_level, size=len(x)) + y_shift)

    return x, y

def plot_multiple_curves(num_curves=3, save=True):
    """
    Generate and plot multiple smooth random curved lines with shaded areas.
    Save the plot and parameters used for recreation, including per-curve alpha values.

    Parameters:
    - num_curves: Number of curves to generate.
    - save: Whether to save the generated plot and parameters.
    """
    plt.figure(figsize=(8, 5))

    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet']
    curves = []
    parameters = []
    timestamp = int(time.time())

    for i in range(num_curves):
        freq_factors = np.random.randint(0.8, 2, size=4).tolist()
        amp_factors = np.random.uniform(0.5, 10, size=4).tolist()
        noise_level = np.random.uniform(0, 0.02)
        x_shift = i * 2
        y_shift = np.random.uniform(0, 3)
        alpha = 0.3 

        wave_combinations = [['sin', 'sinc'], ['cos', 'tanh'], ['sin', 'sinc']]
        x, y = generate_smooth_curve(freq_factors=freq_factors, amp_factors=amp_factors, noise_level=noise_level, 
                                     x_shift=x_shift, y_shift=y_shift, wave_combinations=wave_combinations)
        curves.append((x, y, alpha))

        # Store parameters
        parameters.append({
            "freq_factors": freq_factors,
            "amp_factors": amp_factors,
            "noise_level": noise_level,
            "x_shift": x_shift,
            "y_shift": y_shift,
            "alpha": alpha,
            "wave_combinations": wave_combinations
        })

    y_min = min(np.min(y) for _, y, _ in curves)

    for i, (x, y, alpha) in enumerate(curves):
        color = colors[i % len(colors)]
        plt.plot(x, y, color=color, linewidth=2, label=f"Curve {i+1}")
        plt.fill_between(x, y, y_min, color=color, alpha=alpha)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Multiple Smooth Random Curved Lines with Combined Waves")
    plt.legend()
    plt.grid(True)

    if save:
        # Save plot
        plot_filename = os.path.join(SAVE_DIR, f"curve_plot_{timestamp}.png")
        plt.savefig(plot_filename)
        # Save parameters to JSON
        param_filename = os.path.join(SAVE_DIR, f"curve_parameters_{timestamp}.json")
        with open(param_filename, "w") as f:
            json.dump(parameters, f, indent=4)

    plt.show()

def recreate_plot_from_saved(filename,overwrite = False,overwriteAlpha=0):
    """
    Recreate a previously generated plot from a saved parameters JSON file.

    Parameters:
    - filename: The JSON file containing parameters to recreate the plot.
    """
    param_filepath = os.path.join(SAVE_DIR, filename)

    if not os.path.exists(param_filepath):
        print(f"No saved parameters found for {filename}.")
        return
    
    with open(param_filepath, "r") as f:
        parameters = json.load(f)

    plt.figure(figsize=(8, 5))
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet']
    curves = []

    for i, param in enumerate(parameters):
        x, y = generate_smooth_curve(
            freq_factors=param["freq_factors"],
            amp_factors=param["amp_factors"],
            noise_level=param["noise_level"],
            x_shift=param["x_shift"],
            y_shift=param["y_shift"],
            wave_combinations=param["wave_combinations"]
        )
        if(overwrite is False):
            alpha = param["alpha"]  # Retrieve saved alpha value
        else:
            alpha = overwriteAlpha
        curves.append((x, y, alpha))

    y_min = min(np.min(y) for _, y, _ in curves)

    for i, (x, y, alpha) in enumerate(curves):
        color = colors[i % len(colors)]
        plt.plot(x, y, color=color, linewidth=2, label=f"Curve {i+1}")
        plt.fill_between(x, y, y_min, color=color, alpha=alpha)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Recreated Smooth Random Curved Lines")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, f"curve_plot_{filename}_{overwriteAlpha*100}.png"))
    plt.show()


# Generate and save a new plot
# plot_multiple_curves(num_curves=4, save=True)

saved_files = [f for f in os.listdir(SAVE_DIR) if f.startswith("curve_parameters_") and f.endswith(".json")]
latest_file = sorted(saved_files, reverse=True)[0]
recreate_plot_from_saved(latest_file,True,0.5)