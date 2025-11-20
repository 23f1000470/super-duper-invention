# Marimo-style Interactive Analysis Notebook

This repository contains `analysis.py`, a Marimo-style interactive notebook (plain Python file with Jupyter cell markers) demonstrating relationships between variables with interactive widgets.

## Features
- Email included as a comment: `23f1000470@ds.study.iitm.ac.in`
- Multiple cells with clear variable dependencies
- Interactive sliders (ipywidgets) controlling slope and noise
- Dynamic Markdown output based on widget state
- Comments documenting data flow

## How to run
1. Create a new Python environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -U pip
   pip install numpy pandas matplotlib ipywidgets
