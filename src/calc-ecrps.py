# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:41:05 2024

@author: zpb4
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))
import numpy as np
import pandas as pd
import syn_util
import ensemble_verification_functions as verify
from time import localtime, strftime
from datetime import datetime

now=datetime.now()
print('ecrps start',now.strftime("%H:%M:%S"))

# to simulate a policy, replace 'firo_pool' and 'risk_thresholds' with those found in the train.py file
kcfs_to_tafd = 2.29568411*10**-5 * 86400
K = 317 # TAF
Rmax = 12.5 * kcfs_to_tafd # estimate - from MBK

loc='YRS'
site='NBBC1'
vers='5fold'

upr_pcnt = (0.99,1)
lwr_pcnt = (0,0.99)

sd_syn = '1985-10-15' 
ed_syn = '2019-08-15'

idx_syn = pd.date_range(start = sd_syn, end = ed_syn )

sd = '1990-10-01' 
ed = '2019-08-15'
sl_idx = idx_syn.slice_indexer(sd,ed)
save_figs = True  # T to show plots, F to save .png files to ./plot repo

nsamps = 100

Q_hefs_trn,Qf_hefs_inp,dowy_hefs_trn,tocs,df_idx_hefs = syn_util.extract(sd,ed,forecast_type='hefs',syn_sample='',Rsyn_path='../Synthetic-Forecast-v1-FIRO-DISES',syn_vers='',forecast_param='',loc=loc,site=site,opt_pcnt='',obj_pwr='',opt_strat='',gen_setup='')
Qf_hefs_trn = verify.onesamp_forecast_rearrange(Qf_hefs_inp)

wy_vec = df_idx_hefs.year.values
wy_vec[np.isin(df_idx_hefs.month,[10,11,12])] = wy_vec[np.isin(df_idx_hefs.month,[10,11,12])]+1


Qf_v1_inp = np.load('data/%s/%s/%s-%s_Qf_syn-forecastv1%s_nsamp=%s.npz' %(loc,site,loc,site,vers,nsamps))['arr']
Qf_v1_trn = verify.multisamp_forecast_rearrange(Qf_v1_inp[:,sl_idx,:,:])

Qf_v2_inp = np.load('data/%s/%s/%s-%s_Qf_syn-forecastv2%s_nsamp=%s.npz' %(loc,site,loc,site,vers,nsamps))['arr']
Qf_v2_trn = verify.multisamp_forecast_rearrange(Qf_v2_inp[:,sl_idx,:,:])

#4. eCRPS diagrams
Qf_hefs = Qf_hefs_trn[:,:,:]
Q_hefs = Q_hefs_trn
Qf_v1 = Qf_v1_trn[:,:,:,:]
Qf_v2 = Qf_v2_trn[:,:,:,:]
dowy_hefs = dowy_hefs_trn

ref_st = '1985-10-15'
ref_end = '2019-08-15'
lds = np.arange(np.shape(Qf_hefs)[2])

sset_idx = np.arange(len(Q_hefs_trn))


Q_ref,dowy_ref,df_idx_ref = syn_util.extract_obs(ref_st, ref_end, Rsyn_path='../Synthetic-Forecast-v1-FIRO-DISES',loc=loc,site=site)
Qf_ref_ens_inp = verify.create_obs_ref_ens(Q_ref, dowy_ref, sd, ed)
Qf_ref_ens = Qf_ref_ens_inp[:,:]

ecrps_ss_hefs = np.empty((2,len(lds)))
ecrps_ss_v1 = np.empty((2,len(lds),nsamps))
ecrps_ss_v2 = np.empty((2,len(lds),nsamps))

for i in range(len(lds)):
    Qf_v1_ecrps = Qf_v1[:,:,:,lds[i]]
    Qf_v2_ecrps = Qf_v2[:,:,:,lds[i]]

    ecrps_hefs = verify.onesamp_ecrps(ensemble = Qf_hefs[sset_idx,:,lds[i]], tgt = Q_hefs[sset_idx], pcntile = upr_pcnt)
    ecrps_ref = verify.onesamp_ecrps(ensemble = Qf_ref_ens[sset_idx,:], tgt = Q_hefs[sset_idx], pcntile = upr_pcnt)

    ecrps_v1 = verify.multisamp_ecrps(ensemble = Qf_v1_ecrps[:,sset_idx,:], tgt = Q_hefs[sset_idx], pcntile = upr_pcnt)
    ecrps_v2 = verify.multisamp_ecrps(ensemble = Qf_v2_ecrps[:,sset_idx,:], tgt = Q_hefs[sset_idx], pcntile = upr_pcnt)

    ecrps_ss_hefs[0,i] = 1 - (ecrps_hefs[1] / ecrps_ref[1])
    ecrps_ss_v1[0,i,:] = verify.ecrps_ss(ens_ecrps=ecrps_v1, ref_ecrps=ecrps_ref[1])
    ecrps_ss_v2[0,i,:] = verify.ecrps_ss(ens_ecrps=ecrps_v2, ref_ecrps=ecrps_ref[1])
    
    ecrps_hefs = verify.onesamp_ecrps(ensemble = Qf_hefs[sset_idx,:,lds[i]], tgt = Q_hefs[sset_idx], pcntile = lwr_pcnt)
    ecrps_ref = verify.onesamp_ecrps(ensemble = Qf_ref_ens[sset_idx,:], tgt = Q_hefs[sset_idx], pcntile = lwr_pcnt)

    ecrps_v1 = verify.multisamp_ecrps(ensemble = Qf_v1_ecrps[:,sset_idx,:], tgt = Q_hefs[sset_idx], pcntile = lwr_pcnt)
    ecrps_v2 = verify.multisamp_ecrps(ensemble = Qf_v2_ecrps[:,sset_idx,:], tgt = Q_hefs[sset_idx], pcntile = lwr_pcnt)

    ecrps_ss_hefs[1,i] = 1 - (ecrps_hefs[1] / ecrps_ref[1])
    ecrps_ss_v1[1,i,:] = verify.ecrps_ss(ens_ecrps=ecrps_v1, ref_ecrps=ecrps_ref[1])
    ecrps_ss_v2[1,i,:] = verify.ecrps_ss(ens_ecrps=ecrps_v2, ref_ecrps=ecrps_ref[1])
    now=datetime.now()
    print(i,now.strftime("%H:%M:%S"))
    
np.savez_compressed('data/%s/%s/%s-%s_ecrps-ss_syn-forecastv1%s_nsamp=%s.npz' %(loc,site,loc,site,vers,nsamps), arr=ecrps_ss_v1)
np.savez_compressed('data/%s/%s/%s-%s_ecrps-ss_syn-forecastv2%s_nsamp=%s.npz' %(loc,site,loc,site,vers,nsamps), arr=ecrps_ss_v2)

np.savez_compressed('data/%s/%s/%s-%s_ecrps-ss_hefs.npz' %(loc,site,loc,site), arr=ecrps_ss_hefs)

now=datetime.now()
print('ecrps end',now.strftime("%H:%M:%S"))