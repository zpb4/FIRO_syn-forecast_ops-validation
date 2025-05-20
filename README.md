# FIRO and Synthetic Forecast Operational Validation
This repository houses the FIRO simulation model and probabilistic verification routines to support operational validation of synthetic ensemble forecasts. This repository is associated with the WRR manuscript (2025) 'Synthetic ensemble forecasts: operations-based evaluation and inter-model comparison for reservoir systems across California'. Running this code requires synthetic ensemble forecasts that can be run from the two repositories below, which also include the native HEFS ensemble forecasts and observations:
    
Synthetic Forecast version 1 [https://github.com/zpb4/Synthetic-Forecast-v1-FIRO-DISES](https://github.com/zpb4/Synthetic-Forecast-v1-FIRO-DISES)   
Synthetic Forecast version 2 [https://github.com/zpb4/Synthetic-Forecast-v2-FIRO-DISES](https://github.com/zpb4/Synthetic-Forecast-v2-FIRO-DISES)   

## Workflow   
Once data are available from the synthetic forecasting repos and available to the code in this repo in netcdf format, the user can train FIRO policies for each modeled site, simulate from those policies across both HEFS and synthetic HEFS ensemble forecasts, and analyze operational performance via plotting routines that are associated with the figure sequence in the associated manuscript. Note that workflow steps that are numbered must be run in the order indicated.

The metadata that were generated for the WRR manuscript and can be used to run the plotting routines directly without the lengthy processing steps are available in the following Zenodo repository:  https://doi.org/10.5281/zenodo.11127417

Most scripts below require runs to be accomplished separately for each location ('loc' variable) and site ('site' variable) used in the WRR study. The location/site combinations are listed below:   
ADO/ADOC1 - Prado Dam   
LAM/LAMC1 - Lake Mendocino   
NHG/NHGC1 - New Hogan Lake   
YRS/NBBC1 - New Bullards Bar dam   
YRS/ORDC1 - Oroville Dam (primary analysis site)   
### FIRO policy training and simulation
These scripts train and simulate from FIRO policies
1. ./src/train_param.py - train the FIRO risk curve for a given site and reservoir configuration
2. ./src/ens_simulate.py - simulate from the FIRO policy for specified number of synthetic HEFS samples

### Forecast verification
These scripts train and simulate from FIRO policies
1. ./src/extract_syn-forecasts.py - extract the netcdf synthetic forecast files to a numpy array (data intensive)
2. ./src/calc-ecrps.py - calculates the eCRPS statistic for all lead times 

### Helper scripts
- ./src/model.py - simulation model
- ./src/ensemble_verification_functions.py - simulation model
- ./src/syn_util.py - utility functions for synthetic forecasts
  
### Plotting
Plotting routines for operational validation associated with figure sequence from associated manuscript
- ./src/plot/plot_fig3_forc-ensemble_verification.py - Figure 3
- ./src/plot/plot_fig4_8-panel_risk-curve_ext-event-timeseries.py - Figure 4
- ./src/plot/plot_fig5_comb-S-R_ops-val-metrics_clean.py - Figure 5
- ./src/plot/plot_fig6_4panel_rel-ld-hist.py - Figure 6
- ./src/plot/plot_fig7_2-panel_ops-val_cross-metric_polar.py - Figure 7
- ./src/plot/plot_fig8_risk_curves_5-panel.py - Figure 8
- ./src/plot/plot_fig9_multisite-multiconfig_R-pqq.py - Figure 9
- ./src/plot/plot_fig10_multisite-multiconfig_ops-val-polar.py - Figure 10
- ./src/plot/plot_SI_risk-curve_example.py - SI figure S3.2.


### Contact
Zach Brodeur, zpb4@cornell.edu
