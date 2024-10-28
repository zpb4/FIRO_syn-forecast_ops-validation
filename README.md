# FIRO and Synthetic Forecast Operational Validation
This repository houses the FIRO simulation model and probabilistic verification routines to support operational validation of synthetic ensemble forecasts. This repository is associated with the WRR manuscript (in review) 'Synthetic ensemble forecasts: operational evaluation and inter-model comparison for stylized reservoir systems across California'. Running this code requires synthetic ensemble forecasts that can be run from the two repositories below, which also include the native HEFS ensemble forecasts and observations:
    
Synthetic Forecast version 1 [https://github.com/zpb4/Synthetic-Forecast-v1-FIRO-DISES](https://github.com/zpb4/Synthetic-Forecast-v1-FIRO-DISES)   
Synthetic Forecast version 2 [https://github.com/zpb4/Synthetic-Forecast-v2-FIRO-DISES](https://github.com/zpb4/Synthetic-Forecast-v2-FIRO-DISES)   

## Workflow   
Once data are available from the synthetic forecasting repos and ported to this repo in netcdf format, the user can train FIRO policies for each modeled site, simulate from those policies across both HEFS and synthetic HEFS ensemble forecasts, and analyze operational performance via plotting routines that are associated with the figure sequence in the associated manuscript
### FIRO policy training and simulation
These scripts train and simulate from FIRO policies
- ./src/train_param.py - train the FIRO risk curve for a given site and reservoir configuration
- ./src/ens_simulate.py - simulate from the FIRO policy for specified number of synthetic HEFS samples


Helper scripts
- ./src/model.py - simulation model
- ./src/util.py - utility functions
- ./src/syn_util.py - utility functions for synthetic forecasts
### Plotting
Plotting routines for operational validation associated with figure sequence from associated manuscript
- ./src/plot/plot_fig4_8-panel_risk-curve_ext-event-timeseries.py - Figure 4
- ./src/plot/plot_fig5_comb-S-R_ops-val-metrics_clean.py - Figure 5
- ./src/plot/plot_fig6_4panel_rel-ld-hist.py - Figure 6
- ./src/plot/plot_fig7_2-panel_ops-val_cross-metric_polar.py - Figure 7
- ./src/plot/plot_fig8_risk_curves_5-panel.py - Figure 8
- ./src/plot/plot_fig9_multisite-multiconfig_R-pqq.py - Figure 9
- ./src/plot/plot_fig10_multisite-multiconfig_ops-val-polar.py - Figure 10
- ./src/plot/plot_SI_risk-curve_example.py - SI


### Contact
Zach Brodeur, zpb4@cornell.edu
