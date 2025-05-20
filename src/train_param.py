
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import model
import syn_util
from scipy.optimize import differential_evolution as DE
from time import localtime, strftime
from numba import njit
import pickle

idx = int(sys.argv[1])-1

sd = '1990-10-01' 
ed = '2019-08-15'

loc = 'YRS'
site = 'ORDC1'
vers_out = '5fold'
max_lds = 14
kcfs_to_tafd = 2.29568411*10**-5 * 86400

Q,dowy,df_idx = syn_util.extract_obs(sd,ed,'../Synthetic-Forecast-v2-FIRO-DISES',loc=loc,site=site)

res_params = pd.read_csv('./data/reservoir-storage-safe-rel.csv')
res_idx = np.where(res_params['Reservoir']==site)
K = res_params['Capacity (TAF)'][res_idx[0]].values[0]
Rmax = res_params['Safe Release (CFS)'][res_idx[0]].values[0] / 1000 * kcfs_to_tafd
ramping_rate = res_params['Ramping (CFS)'][res_idx[0]].values[0] / 1000 * kcfs_to_tafd 
#ramping_rate = Rmax
K_ratio_min = 1
Rmax_ratio_min = 0.65   # ADOC1 = 0.75, LAMC1 = 0.6, NHGC1 = 0.25, NBBC1 = 0.45, ORDC1 = 0.65

reservoir_params = {'K': K, 'Rmax': Rmax, 'ramping_rate': ramping_rate, 'Kmin': K_ratio_min, 'Rmaxmin': Rmax_ratio_min}
pickle.dump(reservoir_params,open('data/%s/%s/reservoir-params.pkl' %(loc,site),'wb'))

p = idx
rr_match_Rmax = True   # if T, scale ramping rates with Rmax ratios, if F, leave at 1

Rmax_ratio_mid = 1-(1-Rmax_ratio_min)/2
K_ratio_mid = 1-(1-K_ratio_min)/2

K_ratios = np.full(3,fill_value=[1,K_ratio_mid,K_ratio_min])
Rmax_ratios = np.full(3,fill_value=[1,Rmax_ratio_mid,Rmax_ratio_min])


K_scale = K * K_ratios[p]
Rmax_scale = Rmax * Rmax_ratios[p]
if rr_match_Rmax == True:
    rr_ratios = Rmax_ratios
    ramping_rate_scale = ramping_rate * rr_ratios[p]
elif rr_match_Rmax == False:
    rr_ratios = np.ones(9)
    ramping_rate_scale = ramping_rate * rr_ratios[p]

syn_vers2 = 'v2'
syn_vers2_pct = 0.99
syn_vers2_setup = '5fold'
syn_path2 = '../Synthetic-Forecast-%s-FIRO-DISES' %(syn_vers2) # path to R synthetic forecast repo for 'r-gen' setting below

opt_forecast = 'hefs' # what forecast to optimize to? either 'hefs' for actual HEFS hindcast or 'syn' for a synthetic sample
syn_samp = 'hefs'   # if using a 'syn' sample, this specifies which one to use
tocs_reset = 'none'   # 'tocs' to reset to baseline tocs at beginning of WY, 'firo' to reset to firo pool, 'none' to use continuous storage profile
fixed_pool = True
fixed_pool_value = 0.5
seed = 1

Q,Qf,dowy,tocs_inp,df_idx = model.extract(sd,ed,forecast_type=opt_forecast,syn_sample=syn_samp,Rsyn_path=syn_path2,syn_vers=syn_vers2,forecast_param='a',loc=loc,site=site,opt_pcnt='',obj_pwr='',opt_strat='',gen_setup=syn_vers2_setup,K=K_scale)
tocs = model.get_tocs(dowy,K_scale)
Qf = Qf[:,:,:max_lds] # just use 14 lead days
ne = np.shape(Qf)[1]
nl = max_lds


if opt_forecast == 'perfect':
    Qf = xr.open_dataset('data/perfect_forecast.nc')['inflow']
    Qf = Qf.values
else:
    Qf = Qf

# sum and sort the forecasts    
Qf_summed = np.cumsum(Qf, axis=2)
Qf_summed_sorted = np.sort(Qf_summed, axis = 1)


@njit
def create_param_risk_curve(x):
    lds = max_lds
    lo = x[0]
    hi = x[1]
    pwr = x[2]
    no_risk = int(x[3])
    all_risk = int(x[4])
    if all_risk > 0:
        hi = 1

    if no_risk > 0:
        lo = 0

    ld_sset = np.arange(lds-(no_risk+all_risk)+2)+1
    ld_sset2 = np.arange(lds-(no_risk+all_risk)+1)+1
    ld_sset3 = np.arange(lds-(no_risk+all_risk))+1
    if pwr != 0:
        risk_curve_sset = (np.exp(pwr*ld_sset)-np.exp(pwr)) / (np.exp(pwr*ld_sset[len(ld_sset)-1])-np.exp(pwr))
        risk_curve_sset2 = (np.exp(pwr*ld_sset2)-np.exp(pwr)) / (np.exp(pwr*ld_sset2[len(ld_sset2)-1])-np.exp(pwr))
        risk_curve_sset3 = (np.exp(pwr*ld_sset3)-np.exp(pwr)) / (np.exp(pwr*ld_sset3[len(ld_sset3)-1])-np.exp(pwr))
    if pwr == 0:
        risk_curve_sset = np.interp(ld_sset,np.array([1,ld_sset[-1]]),np.array([0,1]))
        risk_curve_sset2 = np.interp(ld_sset2,np.array([1,ld_sset2[-1]]),np.array([0,1]))
        risk_curve_sset3 = np.interp(ld_sset3,np.array([1,ld_sset3[-1]]),np.array([0,1]))
    if no_risk > 0 and all_risk > 0:
        risk_curve = np.concatenate((np.zeros(no_risk), risk_curve_sset[1:-1],np.ones(all_risk)))
    if no_risk > 0 and all_risk == 0:
        risk_curve = np.concatenate((np.zeros(no_risk), risk_curve_sset2[1:])) * hi
    if no_risk == 0 and all_risk > 0:
        risk_curve_sset = (risk_curve_sset - risk_curve_sset[1]) / (1-risk_curve_sset[1]) * (1-lo)
        risk_curve = np.concatenate((risk_curve_sset+lo, np.ones(all_risk)))[1:-1]
    if no_risk == 0 and all_risk == 0:
        risk_curve_sset = (risk_curve_sset3 - risk_curve_sset3[0]) / (1-risk_curve_sset3[0]) * (1-lo) * hi
        #risk_curve_sset = risk_curve_sset3 * (hi - lo) + lo
        risk_curve = (risk_curve_sset+lo)[:]
        #risk_curve = risk_curve_sset

    return risk_curve

@njit
def opt_wrapper(x):  # x is an array of the decision variables, where x[0] is the size of firo_pool from 0-0.25
    #global Q,Q_MSG,Qf,Qf_MSG,dowy,tocs,df_idx
    risk_thresholds = create_param_risk_curve(x[1:])
    ix = ((1 - risk_thresholds) * (ne)).astype(np.int32)-1
    S, R, firo, spill, Q_cp, rel_leads = model.simulate(firo_pool=x[0], ix=ix, Q=Q, Qf=Qf_summed_sorted, dowy=dowy, tocs=tocs, K = K_scale, Rmax = Rmax_scale, ramping_rate = ramping_rate_scale, policy='firo', tocs_reset='none')

    obj = model.objective(S, firo, Q_cp, K_scale, Rmax_scale, spill)
    print(obj)
  
    return obj


bounds = [(0,0.5)] + [(0,0.99),(0,1),(-2,2),(0,6),(0,6)] # decision variable bounds
if fixed_pool == True:
    bounds[0] = (fixed_pool_value,fixed_pool_value)

opt = DE(opt_wrapper, bounds = bounds, disp = False, maxiter = 10000, polish = True,tol = 0.0001, init='latinhypercube',popsize=100)
#opt = DE(opt_wrapper, bounds = bounds, disp = False, maxiter = 1000, seed = seed, polish = True, tol = 0.0000001, init='sobol',integrality=([False,False,False,False,True,True]),mutation=(1,1.5),popsize=30,recombination=0.35)
#opt = DE(opt_wrapper, bounds = bounds, disp = False, maxiter = 10000, seed = seed, polish = True, tol = 0.0001, init='latinhypercube',mutation=(1,1.5),popsize=30,recombination=0.35,workers=-1)
print(opt.fun)

opt_params = opt.x[1:]
opt_params[3] = int(opt_params[3])
opt_params[4] = int(opt_params[4])
if opt_params[3] > 0:
    opt_params[0] = 0
if opt_params[4] > 0:
    opt_params[1] = 1
    
# # how big is the firo pool?
print('Firo pool size:' +str(opt.x[0]))
print('Opt params:'+' lo='+str(round(opt.x[1],3))+' hi='+str(round(opt.x[2],3))+' pwr='+str(round(opt.x[3],3))+' no_risk='+str(int(opt.x[4]))+' all_risk='+str(int(opt.x[5])))

risk_thresholds = np.zeros(14)
ix = ((1 - risk_thresholds) * (ne)).astype(np.int32)-1
S, R, firo, spill, Q_cp, rel_leads = model.simulate_nonjit(firo_pool=0.5, ix=ix, Q=Q, Qf=Qf_summed_sorted, dowy=dowy, tocs=tocs, K = K_scale, Rmax = Rmax_scale, ramping_rate = ramping_rate_scale, policy='firo', tocs_reset='none')

obj = model.objective(S, firo, Q_cp, K_scale, Rmax_scale, spill)


# save trained parameters
if opt_forecast == 'perfect':
    params = {'obj': opt.fun, 'firo_pool': opt.x[0], 'lo': opt_params[0],'hi': opt_params[1], 'pwr': opt_params[2], 'no_risk': opt_params[3], 'all_risk': opt_params[4]}
    pickle.dump(params,open('data/%s/%s/perfect_param-risk-thresholds_tocs-reset=%s_fixed=%s-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.pkl'%(loc,site,tocs_reset,fixed_pool,fixed_pool_value,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed),'wb'))


params = {'obj': opt.fun, 'firo_pool': opt.x[0], 'lo': opt_params[0],'hi': opt_params[1], 'pwr': opt_params[2], 'no_risk': opt_params[3], 'all_risk': opt_params[4]}
pickle.dump(params,open('data/%s/%s/%s_param-risk-thresholds_tocs-reset=%s_fixed=%s-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.pkl'%(loc,site,syn_samp,tocs_reset,fixed_pool,fixed_pool_value,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed),'wb'))


#------------------------------------------------------END-------------------------------------