# Week 6 — Clustering and Optimization

This folder contains assignments and scripts focused on clustering (K-Means and DBSCAN) and a simple optimization exercise using cluster centroids.

## Contents
- 6540131_Assignment4.ipynb — K-Means on a city lifestyle dataset: scaling, elbow method, PCA visualization, and cluster interpretation.
- 6540131_Assignment5.ipynb — Synthetic dataset generation, DBSCAN from scratch (NumPy), parameter study (9 cases), and profit-oriented optimization using cluster centroids.
- city_lifestyle_dataset.csv — Source dataset used in Assignment 4 (e.g., columns include air_quality_index, happiness_score, etc.).
- DBscan_inclass.py — In-class DBSCAN example script.
- seed.py — Utility script (e.g., reproducibility helpers).

## Requirements
- Python 3.10.11 or 10+
- Packages: numpy, pandas, matplotlib, scikit-learn

Install into a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn
```

## Quick Start

### Assignment 4: K-Means (Notebook)
- Open 6540131_Assignment4.ipynb in VS Code.
- Ensure city_lifestyle_dataset.csv is in the same folder (week6).
- Run all cells in order:
  1) Prepare and scale data
  2) Pre-clustering visualization (Air Quality vs Happiness)
  3) Elbow method to determine optimal `k`
  4) PCA-based cluster visualization with centroids
  5) Cluster interpretation (group means)

Expected outputs: elbow curve, PCA scatter with colored clusters, and a summary table of cluster means.

### Assignment 5: DBSCAN + Optimization (Notebook)
- Open 6540131_Assignment5.ipynb in VS Code.
- Run all cells in order:
  1) Generate and standardize synthetic data
  2) DBSCAN from scratch (NumPy) + core/border/noise analysis
  3) Parameter grid (3×3) with plots and summary table
  4) Profit optimization: choose best parameters, compute centroids (original coordinates), pick best two micro-kitchen locations, estimate savings

Expected outputs: scaled scatter, DBSCAN clustering plot, 3×3 comparison grid, printed optimization results (best pair, locations, savings) and final visualization.

### Script: DBscan_inclass.py
Run the example script directly from the project root (or `week6`):

```bash
python3 week6/DBscan_inclass.py
```

## Tips
- If plots don’t appear, ensure interactive plotting is enabled in your environment.
- Run notebooks from the folder containing the data to avoid path issues.
- For macOS, use `python3` in terminals.

## Notes
- The notebooks are self-contained and will reproduce figures and tables when executed cell-by-cell.
- city_lifestyle_dataset.csv is only required for Assignment 4.
