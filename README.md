# Beyond the Pipeline: Assessing Gender-Neutrality in Board Appointments in Corporate Europe — Code

This repository accompanies the submitted paper **“Beyond the Pipeline: Assessing Gender-Neutrality in Board Appointments in Corporate Europe.”**

It contains the code used for the quantitative analysis, packaged as a user-friendly Python library: **`implicitquotas`**.

## Contents
- **`implicitquotas/`** — Core package used in the analysis.
- **`00_demo.ipynb`** — Demonstrates the package’s core functionality and replicates selected results as a usage example.
- **`01_replication.ipynb`** — Code to reproduce the calculations reported in the article.

## Data Availability
The underlying dataset is proprietary and cannot be shared. To enable experimentation with the codebase, we provide a **synthetic dataset** with the same schema/structure.



## implicitquotas
**implicitquotas** is a Python package for analyzing *implicit quotas* in organizational data — for example, the number of women on boards relative to what would be expected given the available candidate pool.  

It provides a workflow to:

1. **Build null probabilities (`p_s`)** from candidate pools or external benchmarks.  
2. **Perform statistical tests** of observed counts against binomial expectations.  
3. **Handle panel data** with robust variance estimation.  
4. **Taylored analyses** conditional on many groups, such as years, or countries.  
5. **Adjust p-values** for multiple comparisons.  
6. **Visualize results** as interpretable heatmaps.

---