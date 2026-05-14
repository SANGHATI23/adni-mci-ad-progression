# adni-mci-ad-progression
Project models Alzheimer’s disease progression as a longitudinal brain-network transition process using ADNI structural MRI data.The pipeline includes MCI cohort construction,progression-feature engineering,longitudinal atrophy modeling, graph-theoretical network analysis, sensitivity testing,and survival modeling for time-to-conversion prediction.
# ADNI Longitudinal MCI-to-Alzheimer's Progression Modeling

Longitudinal brain network modeling for early prediction of MCI-to-Alzheimer’s progression using ADNI structural MRI data.

## Status
In Progress

## Dataset
ADNI (Alzheimer’s Disease Neuroimaging Initiative)

Access required: https://adni.loni.usc.edu/

---

# Research Question

Can longitudinal MRI brain-change patterns predict MCI-to-AD conversion more accurately than static baseline snapshots, and does conversion correspond to measurable disruption of structural brain covariance networks?

This project reframes Alzheimer’s disease progression as a longitudinal network-transition process rather than a static MRI classification problem.

---

# Key Results

| Component | Result |
|---|---|
| Cohort | 859 baseline MCI patients |
| Strict cohort | pMCI = 199, sMCI = 660 |
| Final MRI cohort | 260 patients with matched MRI + outcome labels |
| Static MRI model ROC-AUC | 0.693 |
| Longitudinal progression model ROC-AUC | 0.736 |
| Balanced Accuracy | 0.700 |
| pMCI Recall | 0.80 |
| Cox Survival C-index | 0.731 |
| Significant survival feature | Ventricular expansion (HR = 1.26, p = 0.010) |
| Borderline survival feature | Entorhinal degeneration (HR = 0.80, p = 0.062) |

---

# Scientific Motivation

Most Alzheimer’s disease machine learning studies frame the problem as static classification from a single MRI scan.

This project instead models disease progression as:

- longitudinal structural degeneration
- dynamic network transition
- coordinated loss of brain-network organization
- progression-risk trajectory modeling

The goal is to identify biologically interpretable early-warning structural network changes associated with MCI-to-AD conversion.

---

# Core Biological Findings

The strongest longitudinal degeneration signals involve:

- entorhinal cortex
- hippocampus
- temporal cortex
- ventricular expansion

These regions align with established Alzheimer’s neurodegeneration pathways.

Key progression findings:

- temporal lobe degeneration strongly predicts conversion
- entorhinal degeneration emerges as an early progression signal
- ventricular expansion predicts earlier conversion timing
- converters show reduced structural network integration

---

# Network Analysis

Brain-change covariance networks were constructed using 12-month MRI delta features.

## Graph-Theoretical Findings

| Metric | pMCI | sMCI |
|---|---|---|
| Density | 0.762 | 0.857 |
| Clustering Coefficient | 0.568 | 0.609 |
| Global Efficiency | 0.881 | 0.929 |

## Interpretation

Stable MCI patients exhibit:

- stronger structural coordination
- higher network efficiency
- more integrated degeneration patterns

Progressive MCI patients exhibit:

- reduced network integration
- weaker entorhinal-temporal coupling
- fragmented degeneration dynamics

This supports a network-disintegration model of Alzheimer’s progression.

---

# Statistical Validation

Permutation testing and edge-level validation were performed.

## Important Result

Edge-level differences did not survive FDR correction after multiple-comparison adjustment.

However:

- biologically meaningful edges repeatedly emerged
- directional stability remained consistent
- entorhinal-temporal and ventricular relationships were reproducible across sensitivity analyses

This suggests stable longitudinal degeneration trends despite limited statistical power.

---

# Survival Modeling

Cox proportional hazards modeling demonstrated:

- ventricular expansion significantly predicts earlier conversion timing
- entorhinal degeneration shows strong progression-risk trends
- longitudinal degeneration patterns contain clinically meaningful timing information beyond binary classification

## Clinical Interpretation

This project models:

- HOW FAST patients convert
- not only WHETHER they convert

This provides a more clinically relevant framework for progression-risk stratification.

---

# Pipeline Structure

```text
src/
  01_load_adni_data.py
  02_label_mci_conversion.py
  03_extract_mri_features.py
  04_merge_dataset.py
  05_baseline_model.py
  06_progression_features.py
  07_progression_model.py
  08_network_analysis.py
  09_graph_metrics.py
  10_statistical_validation.py
  11_survival_modeling.py
  13_network_sensitivity_analysis.py
  14_survival_network_analysis.py

data/results/
  mci_conversion_labels.csv
  mri_features_clean.csv
  final_mri_dataset.csv
  progression_dataset.csv
  pmci_brain_change_network.csv
  smci_brain_change_network.csv
  pmci_minus_smci_network_difference.csv
  pmci_node_metrics.csv
  smci_node_metrics.csv
  network_global_metrics.csv
  edge_statistical_validation.csv
  statistical_validation_network.csv
  network_sensitivity_edge_results.csv
  network_sensitivity_direction_summary.csv
  cox_mri_delta_survival_results.csv

Figures
Recommended figures included in this repository:
pMCI vs sMCI network comparison
difference heatmap
graph-theoretical comparison
survival analysis outputs
Main Scientific Contribution
This project proposes that Alzheimer’s progression can be characterized as:
a transition from a coordinated structural brain network to a fragmented and less efficient degeneration state.
The work combines:
longitudinal MRI progression modeling
graph-theoretical network analysis
sensitivity validation
survival modeling
clinically interpretable progression-risk analysis
Limitations
Edge-level differences did not survive FDR correction
Sample size remains moderate for network neuroscience analysis
Structural covariance is not direct anatomical connectivity
External validation has not yet been performed
Kaplan-Meier analysis and publication-quality figure generation remain in progress
Tools and Libraries
Python
pandas
NumPy
scikit-learn
lifelines
NetworkX
matplotlib
seaborn
Future Directions
Kaplan-Meier survival visualization
External validation (AIBL / NACC)
Diffusion MRI integration
Resting-state fMRI network validation
Multimodal biomarker integration
Manuscript preparation for workshop/conference submission
Citation
If using ADNI data, please follow official ADNI acknowledgment requirements:
https://adni.loni.usc.edu/
Author
Sanghati Basu
MS Healthcare Informatics
University of Illinois Springfield
