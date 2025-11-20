# 23f1000470@ds.study.iitm.ac.in
# analysis.py — Marimo reactive notebook
# - Demonstrates relationship between variables with interactive widgets.
# - Cells are intentionally atomic and document data flow.

# %% [cell 1]
# Cell 1 — base data and constants
# Data flow note:
#  - This cell constructs the base dataset (x, true_intercept, true_slope, df_base).
#  - Downstream cells MUST read these names; when widgets change, downstream cells
#    will re-evaluate automatically in Marimo.
import numpy as np
import pandas as pd
import io, base64
import matplotlib.pyplot as plt

RNG_SEED = 42
np.random.seed(RNG_SEED)

n = 200
x = np.linspace(0, 10, n)
true_intercept = 1.5
true_slope = 2.0

# deterministic part of the data — downstream will add widget-controlled noise/slope
y_base = true_intercept + true_slope * x
df_base = pd.DataFrame({"x": x, "y_base": y_base})

# small preview (self-documenting)
print("Cell 1: df_base preview (first 5 rows)")
print(df_base.head())


# %% [cell 2]
# Cell 2 — interactive widgets and computed observations
# Data flow note:
#  - This cell depends on names from Cell 1: x, true_intercept, true_slope, df_base.
#  - Widgets control slope multiplier and noise sigma; changing them causes this
#    cell (and dependent cells) to re-run in Marimo.
import marimo as mo

# interactive slider: multiplier applied to the true_slope
slope_mult = mo.ui.slider(0.0, 4.0, value=1.0, step=0.01, label="Slope multiplier")
# interactive slider: noise standard deviation
noise_sigma = mo.ui.slider(0.0, 3.0, value=0.5, step=0.01, label="Noise σ")
# seed control to make noise reproducible (optional)
seed = mo.ui.number(0, 99999, value=RNG_SEED, label="Random seed (change to re-seed)")

# Compute dependent variable y_current (reactive: recomputes when sliders change)
np.random.seed(int(seed.value))
current_slope = true_slope * float(slope_mult.value)
noise = np.random.normal(loc=0.0, scale=float(noise_sigma.value), size=len(df_base))
y_current = true_intercept + current_slope * df_base["x"].values + noise

# Create a small helper to render a PNG plot and embed it in markdown
def plot_to_base64_png(x_vals, y_vals, line_x=None, line_y=None, title=""):
    fig, ax = plt.subplots(figsize=(6,3.3), tight_layout=True)
    ax.scatter(x_vals, y_vals, alpha=0.6, s=18, label="observations")
    if line_x is not None and line_y is not None:
        ax.plot(line_x, line_y, linewidth=2.0, label="model (no noise)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return b64

# prepare line (no-noise) for visual reference
line_y = true_intercept + current_slope * df_base["x"].values
img_b64 = plot_to_base64_png(df_base["x"].values, y_current, line_x=df_base["x"].values, line_y=line_y,
                             title=f"Interactive relationship (slope={current_slope:.3f}, noise={float(noise_sigma.value):.3f})")

# Dynamic markdown: shows widget states, summary stats and embeds the plot
mo.md(f"""
## Cell 2 — Controls & Visualization

- **Slope multiplier (widget):** `{slope_mult}`  
- **Noise σ (widget):** `{noise_sigma}`  
- **Random seed (widget):** `{seed}`

**Computed values (live):**
- effective slope = **{current_slope:.4f}**
- first 5 y_current values = `{[float(v) for v in y_current[:5]]}`

**Plot (updates when sliders change):**  
<img src="data:image/png;base64,{img_b64}" alt="plot" style="max-width:100%; border-radius:6px;">

(Notes: this cell reads `df_base` from Cell 1 and writes `y_current` which other cells may consume.)
""")


# %% [cell 3]
# Cell 3 — derived metrics and downstream analysis (depends on Cell 2)
# Data flow note:
#  - This cell reads y_current (from Cell 2) and computes an OLS fit + MSE.
#  - If sliders in Cell 2 change, this cell will re-run automatically in Marimo.
A = np.vstack([df_base["x"].values, np.ones(len(df_base))]).T
m_hat, b_hat = np.linalg.lstsq(A, y_current, rcond=None)[0]
mse = np.mean((y_current - (m_hat * df_base["x"].values + b_hat))**2)

# Show results in dynamic markdown so users immediately see effect of widgets
mo.md(f"""
## Cell 3 — Downstream analysis (OLS & diagnostics)

- Estimated slope (OLS): **{m_hat:.4f}**  
- Estimated intercept (OLS): **{b_hat:.4f}**  
- Mean squared error: **{mse:.6f}**

**Data flow recap**
- Cell 1 creates `df_base`, `true_slope`, `true_intercept`.
- Cell 2 reads those and produces `y_current` using widget values: `slope_mult`, `noise_sigma`, `seed`.
- Cell 3 reads `y_current` and computes OLS / diagnostics.
""")
