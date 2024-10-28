import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import xarray as xr
from util import water_day
import cvxpy as cvx
from numba import njit
import calendar

# using cvxpy for the inner loop prevents numba compiling

kcfs_to_tafd = 2.29568411*10**-5 * 86400
K = 3524

def extract_obs(sd,ed,Rsyn_path,loc,site):
    
    df = pd.read_csv('%s/data/%s/observed_flows.csv' %(Rsyn_path,loc), index_col=0, parse_dates=True)[sd:ed]

    df = df * kcfs_to_tafd
    df_idx = df.index
    Q = df[site].values 

    dowy = np.array([water_day(d,calendar.isleap(d.year)) for d in df.index])
    
    return Q,dowy,df_idx

def extract_obs_scale(sd,ed,Rsyn_path,loc,site,scale_site,event_no,rtn_period):
    
    df = pd.read_csv('%s/data/%s/observed_flows_scaled_%s_evt=%s_rtn=%s.csv' %(Rsyn_path,loc,scale_site,event_no,rtn_period), index_col=0, parse_dates=True)[sd:ed]

    df = df * kcfs_to_tafd
    df_idx = df.index
    Q = df[site].values 

    dowy = np.array([water_day(d,calendar.isleap(d.year)) for d in df.index])
    
    return Q,dowy,df_idx
    
def extract(sd,ed,forecast_type,syn_sample,Rsyn_path,syn_vers,forecast_param,loc,site,opt_pcnt,gen_setup):

    if forecast_type=='hefs':
        path = '%s/out/%s/Qf-%s.nc' % (Rsyn_path,loc,forecast_type)
    elif forecast_type=='syn' and syn_vers=='v1':
        path = '%s/out/%s/Qf-%s_%s_%s.nc' % (Rsyn_path,loc,forecast_type+forecast_param,site,gen_setup)
    elif forecast_type=='syn' and syn_vers=='v2':
        path = '%s/out/%s/Qf-%s_pcnt=%s_%s_%s.nc' % (Rsyn_path,loc,forecast_type,opt_pcnt,site,gen_setup)
        
    da = xr.open_dataset(path)[forecast_type]
    df = pd.read_csv('%s/data/%s/observed_flows.csv' %(Rsyn_path,loc), index_col=0, parse_dates=True)[sd:ed]

    df = df * kcfs_to_tafd
    Q = df[site].values 

    dowy = np.array([water_day(d,calendar.isleap(d.year)) for d in df.index])
    tocs = get_tocs(dowy)
    
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

def extract_scale(sd,ed,forecast_type,syn_sample,Rsyn_path,syn_vers,forecast_param,loc,site,opt_pcnt,gen_setup,scale_site,event_no,rtn_period):

    if forecast_type=='hefs':
        path = '%s/out/%s/Qf-%s_scaled_%s_evt=%s_rtn=%s.nc' % (Rsyn_path,loc,forecast_type,scale_site,event_no,rtn_period)
    elif forecast_type=='syn' and syn_vers=='v1':
        path = '%s/out/%s/Qf-%s_%s_%s.nc' % (Rsyn_path,loc,forecast_type+forecast_param,site,gen_setup)
    elif forecast_type=='syn' and syn_vers=='v2':
        path = '%s/out/%s/Qf-%s_pcnt=%s_%s_%s_scaled_%s_evt=%s_rtn=%s.nc' % (Rsyn_path,loc,forecast_type,opt_pcnt,site,gen_setup,scale_site,event_no,rtn_period)
        
    da = xr.open_dataset(path)[forecast_type]
    df = pd.read_csv('%s/data/%s/observed_flows_scaled_%s_evt=%s_rtn=%s.csv' %(Rsyn_path,loc,scale_site,event_no,rtn_period), index_col=0, parse_dates=True)[sd:ed]

    df = df * kcfs_to_tafd
    Q = df[site].values 

    dowy = np.array([water_day(d,calendar.isleap(d.year)) for d in df.index])
    tocs = get_tocs(dowy)
    
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

def extract86(sd,ed,Rsyn_path,loc,site):
    path = '%s/out/%s/Qf-hefs86.nc' % (Rsyn_path,loc)
    da = xr.open_dataset(path)['hefs']
    df = pd.read_csv('%s/data/%s/observed_flows.csv' %(Rsyn_path,loc), index_col=0, parse_dates=True)[sd:ed]

    df = df * kcfs_to_tafd
    Q = df[site].values 
    
    ado = {'ADOC1':0}
    nhg = {'MSGC1L':0,'NHGC1':1}
    lam = {'HOPC1L':0,'LAMC1':1,'UKAC1':2}
    yrs = {'MRYC1L':0,'NBBC1':1,'ORDC1':2}
    locs = {'ADO':ado,'NHG':nhg,'LAM':lam,'YRS':yrs}
    
    site_id = locs[loc][site]

    dowy = np.array([water_day(d,calendar.isleap(d.year)) for d in df.index])
    tocs = get_tocs(dowy)
    
    Qf = da.sel(ensemble=0, site=site_id, date=df.index).values * kcfs_to_tafd # np.array (time, trace, lead)

    df_idx = df.index
    
    return Q,Qf,dowy,tocs,df_idx


def get_tocs(d): 
  tp = [0, 60, 170, 252, 366]
  sp = [K, 0.5*K, 0.5*K, K, K] # Note: actual NHG tocs min is 152 TAF, about 0.48 * K
  return np.interp(d, tp, sp)

def create_param_risk_curve(x,lds):
    lo = x[0]
    hi = x[1]
    pwr = x[2]
    no_risk = int(x[3])
    all_risk = int(x[4])
    if all_risk > 0:
        hi = 1

    if no_risk > 0:
        lo = 0

    ld_sset = np.arange(lds-(no_risk+all_risk)+1)
    risk_curve_sset = (np.exp(pwr*ld_sset)-np.exp(pwr)) / (np.exp(pwr*ld_sset[len(ld_sset)-1])-np.exp(pwr))
    if no_risk > 0 and all_risk > 0:
        risk_curve = np.concatenate((np.zeros(no_risk), risk_curve_sset[1:],np.ones(all_risk)))
    if no_risk > 0 and all_risk == 0:
        risk_curve = np.concatenate((np.zeros(no_risk), risk_curve_sset[1:])) * hi
    if no_risk == 0 and all_risk > 0:
        risk_curve_sset = (risk_curve_sset - risk_curve_sset[1]) / (1-risk_curve_sset[1]) * (1-lo)
        risk_curve = np.concatenate((risk_curve_sset+lo, np.ones(all_risk)))[1:]
    if no_risk == 0 and all_risk == 0:
        risk_curve_sset = (risk_curve_sset - risk_curve_sset[1]) / (1-risk_curve_sset[1]) * (1-lo) * hi
        risk_curve = (risk_curve_sset+lo)[1:]

    return risk_curve

def create_risk_curve(x):
	# convert 0-1 values to non-decreasing risk curve
    x_copy = np.copy(x)
    for i in range(1, len(x_copy)):
        x_copy[i] = x_copy[i-1] + (1 - x_copy[i-1]) * x_copy[i]
    return x_copy




