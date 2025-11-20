# 23f1000470@ds.study.iitm.ac.in
# analysis.py
# Marimo-style notebook (a plain Python file with Jupyter cell markers '# %%')
# - This file is runnable in Jupyter/VSCode interactive using the '# %%' cells.
# - It demonstrates variable dependencies across cells, an interactive slider,
#   dynamic markdown output, and comments documenting data flow.

# %% [cell 1]
# Cell 1: Imports and base data generation
# Data flow note:
#   - We create a base synthetic dataset here: `x`, `y_base`.
#   - Downstream cells will depend on these variables and transform them
#     according to widget inputs (slope, noise).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import ipywidgets as widgets
from ipywidgets import Output

# For reproducibility
RNG_SEED = 42
np.random.seed(RNG_SEED)

# Generate a simple synthetic dataset
n = 200
x = np.linspace(0, 10, n)
true_intercept = 1.5
true_slope = 2.0
# y_base is the deterministic part — downstream code will add widget-controlled noise/slope changes
y_base = true_intercept + true_slope * x

# Put into dataframe for convenience
df_base = pd.DataFrame({"x": x, "y_base": y_base})

# Show a quick preview (will render in notebook)
display(Markdown("**Cell 1 — base dataset preview (first 5 rows)**"))
display(df_base.head())

# %% [cell 2]
# Cell 2: Interactive controls (slider widget) and plotting output
# Data flow note:
#   - This cell reads `df_base`, `y_base` and updates a plot when the widget changes.
#   - The widget controls `slope_adjust` and `noise_level`. The plot and downstream
#     summary text depend on these values.
#
# Requirements satisfied here:
#   - Interactive slider widget (ipywidgets)
#   - Dynamic plot and output
slope_slider = widgets.FloatSlider(
    value=1.0,
    min=0.0,
    max=4.0,
    step=0.05,
    description='Slope mult:',
    continuous_update=True,
    readout_format='.2f'
)

noise_slider = widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=3.0,
    step=0.05,
    description='Noise σ:',
    continuous_update=True,
    readout_format='.2f'
)

# Output areas: one for the plot, one for dynamic markdown summary
plot_out = Output()
md_out = Output()

def update_plot_and_summary(slope_mult, noise_sigma):
    """
    Updates viz and summary based on:
      - slope_mult: multiplier applied to the base slope (true_slope)
      - noise_sigma: standard deviation of Gaussian noise added to y
    Side effects:
      - writes to `plot_out` and `md_out` Output widgets.
    """
    # Compute dependent variable y_current (depends on df_base and widget values)
    # This demonstrates variable dependency across cells: df_base -> y_current
    current_slope = true_slope * slope_mult
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=len(df_base))
    y_current = true_intercept + current_slope * df_base["x"].values + noise

    # prepare a temporary dataframe for display / diagnostics
    df_current = df_base.copy()
    df_current["y_current"] = y_current

    # Update the plot
    with plot_out:
        plot_out.clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df_current["x"], df_current["y_current"], alpha=0.6, label="observations")
        ax.plot(df_current["x"], true_intercept + current_slope * df_current["x"], label="model (no noise)", linewidth=2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Interactive relationship (slope={current_slope:.2f}, noise={noise_sigma:.2f})")
        ax.legend()
        plt.show()

    # Compute summary statistics and show dynamic markdown
    with md_out:
        md_out.clear_output(wait=True)
        # Basic linear regression fit (least squares) to show how slope estimate changes
        A = np.vstack([df_current["x"].values, np.ones(len(df_current))]).T
        m_hat, b_hat = np.linalg.lstsq(A, df_current["y_current"], rcond=None)[0]
        mse = np.mean((df_current["y_current"] - (m_hat * df_current["x"].values + b_hat))**2)

        summary_md = f"""
**Dynamic summary**
- Requested slope multiplier: **{slope_mult:.2f}** → current model slope = **{current_slope:.2f**}  
- Estimated slope (OLS fit): **{m_hat:.3f}**  
- Estimated intercept (OLS fit): **{b_hat:.3f}**  
- Mean squared error: **{mse:.3f}**

(These numbers update whenever you move the sliders.)
"""
        display(Markdown(summary_md))

# Wire widgets to update function with interactive.observe or interactive_output
ui = widgets.HBox([slope_slider, noise_slider])
out = widgets.VBox([plot_out, md_out])

def _on_change(change):
    # We ignore the `change` payload and just use current values
    update_plot_and_summary(slope_slider.value, noise_slider.value)

# Attach handlers
slope_slider.observe(_on_change, names='value')
noise_slider.observe(_on_change, names='value')

# Initialize once
update_plot_and_summary(slope_slider.value, noise_slider.value)

# Display the UI for the user
display(Markdown("## Cell 2 — Interactive controls and visualization"))
display(ui)
display(out)

# %% [cell 3]
# Cell 3: Downstream analysis that depends on the widget-controlled state
# Data flow note:
#   - This cell demonstrates additional computations that read the *current* slider
#     values and compute a derived metric: an SNR-like ratio and a simple diagnostic table.
#   - It depends on slope_slider.value and noise_slider.value defined above (variable dependency).
#
# Important: In a linear, single-file Marimo-style notebook, execution order matters:
#   - Run Cell 1 -> Cell 2 -> Cell 3 for expected behavior.
display(Markdown("## Cell 3 — Derived metrics (dependent on widget state)"))

def derived_metrics(slope_mult, noise_sigma):
    # Effective slope
    eff_slope = true_slope * slope_mult
    # approximate signal amplitude (slope * range of x)
    signal_amp = eff_slope * (df_base["x"].max() - df_base["x"].min())
    # approximate SNR-like measure: signal_amp / (3 * noise_sigma)  (3σ as typical spread)
    snr_like = signal_amp / (3 * noise_sigma) if noise_sigma > 0 else np.inf

    metrics = {
        "effective_slope": eff_slope,
        "signal_amplitude": signal_amp,
        "noise_std": noise_sigma,
        "snr_like": snr_like
    }
    return metrics

metrics = derived_metrics(slope_slider.value, noise_slider.value)
metrics_df = pd.DataFrame([metrics]).T
metrics_df.columns = ["value"]

display(Markdown("**Derived metrics table**"))
display(metrics_df)

display(Markdown(
    "### Notes on data flow\n"
    "- `df_base`, `true_slope`, and `true_intercept` are created in Cell 1.\n"
    "- Cell 2 reads those and produces `y_current` depending on widget inputs (`slope_slider`, `noise_slider`).\n"
    "- Cell 3 reads the current widget values (so it must be executed after Cell 2 in an interactive run) and computes derived metrics.\n"
))
