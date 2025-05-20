import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from syn_util import water_day
from numba import njit
import calendar

kcfs_to_tafd = 2.29568411*10**-5 * 86400
#K_base = 3524 # TAF
#Rmax_base = 150 * kcfs_to_tafd # estimate - from MBK, this is correct

# ramping rate from ORO WCM is 10,000 cfs every two hours for an increase and 5,000 cfs every two hours for a decrease
#ramping_rate_base = 120 * kcfs_to_tafd
def extract(sd,ed,forecast_type,syn_sample,Rsyn_path,syn_vers,forecast_param,loc,site,opt_pcnt,obj_pwr,opt_strat,gen_setup,K):

    if forecast_type=='hefs':
        path = '%s/out/%s/Qf-%s.nc' % (Rsyn_path,loc,forecast_type)
    elif forecast_type=='syn' and syn_vers=='v1':
        path = '%s/out/%s/Qf-%s_%s_%s.nc' % (Rsyn_path,loc,forecast_type+forecast_param,site,gen_setup)
    elif forecast_type=='syn' and syn_vers=='v2':
        path = '%s/out/%s/Qf-%s_pcnt=%s_objpwr=%s_optstrat=%s_%s_%s.nc' % (Rsyn_path,loc,forecast_type,opt_pcnt,obj_pwr,opt_strat,site,gen_setup)
        
    da = xr.open_dataset(path)[forecast_type]
    df = pd.read_csv('%s/data/%s/observed_flows.csv' %(Rsyn_path,loc), index_col=0, parse_dates=True)[sd:ed]

    df = df * kcfs_to_tafd
    Q = df[site].values 

    dowy = np.array([water_day(d,calendar.isleap(d.year)) for d in df.index])
    tocs = get_tocs(dowy,K)
    
    ado = {'ADOC1':0}
    nhg = {'MSGC1L':0,'NHGC1':1}
    lam = {'HOPC1L':0,'LAMC1':1,'UKAC1':2}
    yrs = {'MRYC1L':0,'NBBC1':1,'ORDC1':2}
    locs = {'ADO':ado,'NHG':nhg,'LAM':lam,'YRS':yrs}
    
    site_id = locs[loc][site]
    
    # (ensemble: 4, site: 2, date: 15326, trace: 42, lead: 15)
    if forecast_type == 'hefs':
        Qf = da.sel(ensemble=0, site=site_id, date=df.index).values * kcfs_to_tafd # np.array (time, trace, lead)
    if forecast_type == 'syn':
        Qf = da.sel(ensemble=int(syn_sample[5:])-1, site=site_id, date=df.index).values * kcfs_to_tafd # np.array (time, trace, lead)
    
    #recommend not presorting ensembles because it will mix and match ensemble members
    #Qf.sort(axis = 1)
    #Qf_MSG.sort(axis = 1)
    df_idx = df.index
    
    return Q,Qf,dowy,tocs,df_idx

# helper functions
@njit
def get_tocs(d,K):
    tp = np.array([0, 7, 140, 340, 366], dtype=np.float64)
    sp = np.array([K, 0.5*K, 0.5*K, K, K], dtype=np.float64)
    return np.interp(d, tp, sp)

@njit
def firo_curve(d, firo_pool,K):
  tp = np.array([0, 7, 140, 340, 366], dtype=np.float64)
  sp = np.array([K, (0.5+firo_pool)*K, (0.5+firo_pool)*K, K, K], dtype=np.float64)
  return np.interp(d, tp, sp)

@njit
def create_risk_curve(x):
  # convert 0-1 values to non-decreasing risk curve
    x_copy = np.copy(x)
    for i in range(1, len(x_copy)):
        x_copy[i] = x_copy[i-1] + (1 - x_copy[i-1]) * x_copy[i]
    return x_copy

@njit
def clip(x, l, u):
    return max(min(x, u), l)


@njit
def daily_opt(S0, Smax, fmax, Q, Qf_summed_sorted, ix):
    ne,nl = Qf_summed_sorted.shape
    rel_ld = 0
    Qf_quantiles = np.zeros(nl)
    # iterate over lead times
    for l in range(nl):
        if ix[l] != -1:
            Qf_quantiles[l] = Qf_summed_sorted[ix[l],l]
        elif ix[l] == -1:
            Qf_quantiles[l] = 0
    
    #a risk threshold of near 100% means you accept all risk and don't react to forecasts at all
    ##Qf_quantiles[ix==0] = 0
    # vectorized computation of future storage
    Sf_q = S0 + Qf_quantiles
    
    # vectorized computation to find where future storage exceeds fmax and calculate the required release
    releases = np.maximum(Sf_q - fmax,0) / (np.arange(nl) + 1)
    
    # find the max release and the corresponding lead time
    R = np.max(releases)
    if R>0:
        rel_ld = np.argmax(releases)+1
    
    return R, rel_ld


@njit
def simulate(firo_pool, ix, Q, Qf, dowy, tocs, K, Rmax, ramping_rate, policy='baseline', tocs_reset='tocs', summer_rel_rule='firo'):
  T = len(Q)
  S = np.full(T+1, np.nan)
  R = np.full(T, np.nan)
  Q_cp = np.full(T, np.nan) # downstream flow at the control point
  Q_sim = np.zeros(T+1)
  Q_sim[:T] = Q
  spill = np.zeros(T)
  rel_leads = np.zeros(T)
  tocs = get_tocs(dowy,K)
  
  ##manual change to use either flat tocs or precip-based tocs
  #tocs = get_tocs(dowy) # storage guide curve that corresponds to conservative >=12 in antecedent precip
  #tocs = tocs            # use precip-based tocs
  
  firo = firo_curve(dowy, firo_pool,K) # firo pool guide curve
  S[0] = np.unique(tocs[dowy==7])[0] #start at baseline winter TOCS for all simulation configurations
  #S[0] = K #start full
  S[0] = (0.5+firo_pool)*K # start at firo pool limit
  for t in range(T):
    if dowy[t] == 0 and tocs_reset == 'tocs':
      S[t] = np.unique(tocs[dowy==7])[0] # start every WY at winter TOCS
      
    elif dowy[t] == 0 and tocs_reset == 'firo':
      S[t] = np.unique(firo[dowy==7])[0] # start every WY at winter FIRO pool
    
    elif dowy[t] == 0 and tocs_reset == 'none':
      S[t] = S[t]
      
    if policy == 'baseline':
      R[t] = clip(S[t] + Q[t] - tocs[t], 0, Rmax) #ensure releases will not exceeed downstream Rmax

    elif policy == 'firo' and summer_rel_rule == 'firo':
      R[t],rel_leads[t] = daily_opt(S[t], tocs[t], firo[t], Q_sim[t+1], Qf[t,:,:], ix) # added the daily firo pool value
      if R[t] < (S[t] + Q_sim[t+1] - K): # added to try to fix spill issue 1/10/24
          R[t] = S[t] + Q_sim[t+1] - K # override release calculated from daily_opt to ensure no spill when res is full
      if R[t] > Rmax: # added 4/24/24
          R[t] = Rmax
    
    elif policy == 'firo' and summer_rel_rule == 'baseline':
      #if in summer pool, execute baseline operations
      if dowy[t]==0 or dowy[t]>=220:
          R[t] = clip(S[t] + Q_sim[t+1] - tocs[t], 0, Rmax) #ensure releases will not exceeed downstream Rmax
          
      #o/w execute the firo rule
      else:
          R[t],rel_leads[t] = daily_opt(S[t], tocs[t], firo[t], Q_sim[t+1], Qf[t,:,:], ix) # added the daily firo pool value
          if R[t] < (S[t] + Q_sim[t+1] - K): # added to try to fix spill issue 1/10/24
              R[t] = S[t] + Q_sim[t+1] - K # override release calculated from daily_opt to ensure no spill when res is full
          if R[t] > Rmax: 
              R[t] = Rmax
              
    if R[t] > S[t] + Q_sim[t+1]: # release limited by water available
      R[t] = S[t] + Q_sim[t+1]
      
    if np.abs(R[t]-R[t-1]) > ramping_rate: #release limited by ramping rate (unlikely to be restrictive)
      R[t] = R[t-1] + np.sign((R[t]-R[t-1])) * ramping_rate
          
    if S[t] + Q_sim[t+1] - R[t] > K: # spill
      spill[t] = S[t] + Q_sim[t+1] - R[t] - K
    
    S[t+1] = S[t] + Q_sim[t+1] - R[t] - spill[t]
    Q_cp[t] = R[t] # downstream flow at the control point

  return S[:T],R,firo,spill,Q_cp,rel_leads

def simulate_nonjit(firo_pool, ix, Q, Qf, dowy, tocs, K, Rmax, ramping_rate, policy='baseline', tocs_reset='tocs', summer_rel_rule='firo'):
  T = len(Q)
  S = np.full(T+1, np.nan)
  R = np.full(T, np.nan)
  Q_cp = np.full(T, np.nan) # downstream flow at the control point
  Q_sim = np.zeros(T+1)
  Q_sim[:T] = Q
  spill = np.zeros(T)
  rel_leads = np.zeros(T)
  tocs = get_tocs(dowy,K)
  
  ##manual change to use either flat tocs or precip-based tocs
  #tocs = get_tocs(dowy) # storage guide curve that corresponds to conservative >=12 in antecedent precip
  #tocs = tocs            # use precip-based tocs
  
  firo = firo_curve(dowy, firo_pool,K) # firo pool guide curve
  S[0] = np.unique(tocs[dowy==7])[0] #start at baseline winter TOCS for all simulation configurations
  #S[0] = K #start full
  S[0] = (0.5+firo_pool)*K # start at firo pool limit
  for t in range(T):
    if dowy[t] == 0 and tocs_reset == 'tocs':
      S[t] = np.unique(tocs[dowy==7])[0] # start every WY at winter TOCS
      
    elif dowy[t] == 0 and tocs_reset == 'firo':
      S[t] = np.unique(firo[dowy==7])[0] # start every WY at winter FIRO pool
    
    elif dowy[t] == 0 and tocs_reset == 'none':
      S[t] = S[t]
      
    if policy == 'baseline':
      R[t] = clip(S[t] + Q[t] - tocs[t], 0, Rmax) #ensure releases will not exceeed downstream Rmax

    elif policy == 'firo' and summer_rel_rule == 'firo':
      R[t],rel_leads[t] = daily_opt(S[t], tocs[t], firo[t], Q_sim[t+1], Qf[t,:,:], ix) # added the daily firo pool value
      if R[t] < (S[t] + Q_sim[t+1] - K): # added to try to fix spill issue 1/10/24
          R[t] = S[t] + Q_sim[t+1] - K # override release calculated from daily_opt to ensure no spill when res is full
      if R[t] > Rmax: # added 4/24/24
          R[t] = Rmax
    
    elif policy == 'firo' and summer_rel_rule == 'baseline':
      #if in summer pool, execute baseline operations
      if dowy[t]==0 or dowy[t]>=220:
          R[t] = clip(S[t] + Q_sim[t+1] - tocs[t], 0, Rmax) #ensure releases will not exceeed downstream Rmax
          
      #o/w execute the firo rule
      else:
          R[t],rel_leads[t] = daily_opt(S[t], tocs[t], firo[t], Q_sim[t+1], Qf[t,:,:], ix) # added the daily firo pool value
          if R[t] < (S[t] + Q_sim[t+1] - K): # added to try to fix spill issue 1/10/24
              R[t] = S[t] + Q_sim[t+1] - K # override release calculated from daily_opt to ensure no spill when res is full
          if R[t] > Rmax: 
              R[t] = Rmax
              
    if R[t] > S[t] + Q_sim[t+1]: # release limited by water available
      R[t] = S[t] + Q_sim[t+1]
      
    if np.abs(R[t]-R[t-1]) > ramping_rate: #release limited by ramping rate (unlikely to be restrictive)
      R[t] = R[t-1] + np.sign((R[t]-R[t-1])) * ramping_rate
          
    if S[t] + Q_sim[t+1] - R[t] > K: # spill
      spill[t] = S[t] + Q_sim[t+1] - R[t] - K
    
    S[t+1] = S[t] + Q_sim[t+1] - R[t] - spill[t]
    Q_cp[t] = R[t] # downstream flow at the control point

  return S[:T],R,firo,spill,Q_cp,rel_leads

@njit
def objective(S,firo,Q_cp,K,Rmax,spill):
  obj = 0
  obj -= S.mean()/K # maximize average storage (percent of total capacity)
  #obj += np.sum(S > firo) * 100 # penalize for going above firo pool limit
  obj += np.sum(Q_cp > Rmax) * 100 # penalize for releases above downstream max
  obj += np.sum(spill > 0) * 100 # penalize for any spill at all

  return obj

@njit
def multiobjective(S,R,firo,Q_cp,Rmax,spill):

  S_obj = 0
  S_obj -= S.mean()/3524 # maximize average storage (percent of total capacity)
  #obj += np.sum(S > firo) * 100 # penalize for going above firo pool limit
  S_obj += np.sum(spill > 0) * 100 # penalize for any spill at all

  R_obj = 0
  R_obj += np.sum(Q_cp > (Rmax-100)) # minimize the number of high frequency releases
  R_obj += np.sum(Q_cp > Rmax) * 100 # penalize for releases above downstream max

  return S_obj, R_obj

"""
def plot_results(Q,S,R,tocs,firo,spill,Q_cp,df_idx,title):
    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    plt.plot(df_idx, tocs, c='gray')
    plt.plot(df_idx, firo, c = 'green')
    plt.plot(df_idx, S, c='blue')
    plt.axhline(K_base, color='red')
    plt.ylabel('TAF')
    plt.ylim([0, K_base+50])
    plt.gcf().autofmt_xdate()
    plt.legend(['TOCS','FIRO Pool','Storage'], loc = 'lower right')

    plt.subplot(3,1,2)
    plt.plot(df_idx, Q / kcfs_to_tafd)
    # plt.plot(df.index, (R+spill) / cfs_to_tafd)
    plt.plot(df_idx, Q_cp / kcfs_to_tafd)
    plt.axhline(Rmax_base / kcfs_to_tafd, color='red')
    plt.ylabel('kcfs')
    plt.gcf().autofmt_xdate()
    plt.legend(['Inflow', 'Q_cp', 'Max safe release'], loc = 'lower right')
    
    plt.subplot(3,1,3)
    plt.plot(df_idx, spill)
    plt.legend(['Spill'], loc = 'lower right')
    plt.suptitle(title)
    plt.show()
"""
