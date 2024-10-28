
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import model
import netcdf_extract
from scipy.optimize import differential_evolution as DE
from time import localtime, strftime
from numba import njit
import pickle

loc = 'YRS'
site = 'ORDC1'
sd = '1990-10-01' 
ed = '2019-08-15'

#////////////////////////////////////////////////////////////////////////////////
#define scaling ratios based on ORO baseline

Q_oro,dowy,df_idx = netcdf_extract.extract_obs(sd,ed,'../Synthetic-Forecast-v2-FIRO-DISES',loc=loc,site=site)
Q,Qf,dowy,tocs_inp,df_idx = model.extract(sd,ed,forecast_type=opt_forecast,syn_sample=syn_samp,Rsyn_path=syn_path2,syn_vers=syn_vers2,forecast_param='a',loc=loc,site=site,opt_pcnt=syn_vers2_pct,gen_setup=syn_vers2_setup,K=K_scale)
tocs = model.get_tocs(dowy,K_scale)

