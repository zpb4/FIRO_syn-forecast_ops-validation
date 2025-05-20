
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model
import syn_util
from time import localtime, strftime
from datetime import datetime
import pickle

now=datetime.now()
print('sim start',now.strftime("%H:%M:%S"))

idx = int(sys.argv[1])-1

kcfs_to_tafd = 2.29568411*10**-5 * 86400
max_lds = 14

sd = '1990-10-01' 
ed = '2019-08-15'

loc = 'YRS'
site = 'NBBC1'
svers = '5fold'
opt_samp = 'hefs'
nsamps = 100
seed = 1

res_params = pickle.load(open('data/%s/%s/reservoir-params.pkl'%(loc,site),'rb'),encoding='latin1')

K = res_params['K']
Rmax = res_params['Rmax']
ramping_rate = res_params['ramping_rate']
K_ratio_min = res_params['Kmin']
Rmax_ratio_min = res_params['Rmaxmin']

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

syn_vers1 = 'v1'
syn_vers1_param = 'a'
syn_vers1_setup = '5fold'
syn_path1 = '../Synthetic-Forecast-%s-FIRO-DISES' %(syn_vers1) # path to R synthetic forecast repo for 'r-gen' setting below

syn_vers2 = 'v2'
syn_vers2_pct = 0.99
syn_vers2_objpwr = 0
syn_vers2_optstrat = 'ecrps-dts'
syn_vers2_setup = '5fold-test'
syn_path2 = '../Synthetic-Forecast-%s-FIRO-DISES' %(syn_vers2) # path to R synthetic forecast repo for 'r-gen' setting below

sim_forecast = 'syn' # what forecast to optimize to? either 'hefs' for actual HEFS hindcast or 'syn' for a synthetic sample
syn_samp = 'Synth1'   # if using a 'syn' sample, this specifies which one to use
tocs_reset = 'none'   # 'tocs' to reset to baseline tocs at beginning of WY, 'firo' to reset to firo pool, 'none' to use continuous storage profile
fixed_pool = True
fixed_pool_value = 0.5


pars = pickle.load(open('data/%s/%s/%s_param-risk-thresholds_tocs-reset=%s_fixed=%s-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.pkl'%(loc,site,opt_samp,tocs_reset,fixed_pool,fixed_pool_value,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed),'rb'),encoding='latin1')
risk_curve = syn_util.create_param_risk_curve((pars['lo'],pars['hi'],pars['pwr'],pars['no_risk'],pars['all_risk']),lds=max_lds)

Q,Qf,dowy,tocs,df_idx = model.extract(sd,ed,forecast_type=sim_forecast,syn_sample=syn_samp,Rsyn_path=syn_path2,syn_vers=syn_vers2,forecast_param='a',loc=loc,site=site,opt_pcnt=syn_vers2_pct,obj_pwr=syn_vers2_objpwr,opt_strat=syn_vers2_optstrat,gen_setup=syn_vers2_setup,K=K_scale)
Qf = Qf[:,:,:max_lds]
ne = np.shape(Qf)[1]
Qf_summed = np.cumsum(Qf, axis=2)
Qf_summed_sorted = np.sort(Qf_summed, axis = 1)
#tocs_inp = model.get_tocs(dowy)
ix = ((1 - risk_curve) * (ne)).astype(np.int32)-1
S, R, firo, spill, Q_cp, rel_leads = model.simulate(firo_pool=pars['firo_pool'], ix=ix, Q=Q, Qf=Qf_summed_sorted, dowy=dowy, tocs=tocs, K = K_scale, Rmax = Rmax_scale, ramping_rate = ramping_rate_scale, policy='firo', tocs_reset='none')

comb_arrayv1=np.empty((nsamps,np.shape(Q)[0],7))
comb_arrayv2=np.empty((nsamps,np.shape(Q)[0],7))

for i in range(nsamps):
    syn_samp = 'Synth'+str(i+1) #if using a 'syn' sample, this specifies which one to use
    #version 1
    Q,Qf,dowy,tocs,df_idx = model.extract(sd,ed,forecast_type=sim_forecast,syn_sample=syn_samp,Rsyn_path=syn_path1,syn_vers=syn_vers1,forecast_param=syn_vers1_param,loc=loc,site=site,opt_pcnt='',obj_pwr='',opt_strat='',gen_setup=syn_vers1_setup,K=K_scale)
    Qf = Qf[:,:,:max_lds]
    Qf_summed = np.cumsum(Qf, axis=2)
    Qf_summed_sorted = np.sort(Qf_summed, axis = 1)
    ix = ((1 - risk_curve) * (ne)).astype(np.int32)-1
    S, R, firo, spill, Q_cp, rel_leads = model.simulate(firo_pool=pars['firo_pool'], ix=ix, Q=Q, Qf=Qf_summed_sorted, dowy=dowy, tocs=tocs, K = K_scale, Rmax = Rmax_scale, ramping_rate = ramping_rate_scale, policy='firo', tocs_reset='none')
    comb_arrayv1[i,:,0]=S
    comb_arrayv1[i,:,1]=R
    comb_arrayv1[i,:,2]=tocs
    comb_arrayv1[i,:,3]=firo
    comb_arrayv1[i,:,4]=spill
    comb_arrayv1[i,:,5]=Q_cp
    comb_arrayv1[i,:,6]=rel_leads
    #version 2
    Q,Qf,dowy,tocs,df_idx = model.extract(sd,ed,forecast_type=sim_forecast,syn_sample=syn_samp,Rsyn_path=syn_path2,syn_vers=syn_vers2,forecast_param='',loc=loc,site=site,opt_pcnt=syn_vers2_pct,obj_pwr=syn_vers2_objpwr,opt_strat=syn_vers2_optstrat,gen_setup=syn_vers2_setup,K=K_scale)
    Qf = Qf[:,:,:max_lds]
    Qf_summed = np.cumsum(Qf, axis=2)
    Qf_summed_sorted = np.sort(Qf_summed, axis = 1)
    ix = ((1 - risk_curve) * (ne)).astype(np.int32)-1
    S, R, firo, spill, Q_cp, rel_leads = model.simulate(firo_pool=pars['firo_pool'], ix=ix, Q=Q, Qf=Qf_summed_sorted, dowy=dowy, tocs=tocs, K = K_scale, Rmax = Rmax_scale, ramping_rate = ramping_rate_scale, policy='firo', tocs_reset='none')
    comb_arrayv2[i,:,0]=S
    comb_arrayv2[i,:,1]=R
    comb_arrayv2[i,:,2]=tocs
    comb_arrayv2[i,:,3]=firo
    comb_arrayv2[i,:,4]=spill
    comb_arrayv2[i,:,5]=Q_cp
    comb_arrayv2[i,:,6]=rel_leads
    now=datetime.now()
    print('syn-samp',i,now.strftime("%H:%M:%S"))

np.savez_compressed('data/%s/%s/sim-array_synforc-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.npz' %(loc,site,syn_vers1+svers,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed), arr=comb_arrayv1)
np.savez_compressed('data/%s/%s/sim-array_synforc-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.npz' %(loc,site,syn_vers2+svers,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed), arr=comb_arrayv2)

now=datetime.now()
print('sim end',now.strftime("%H:%M:%S"))


#----------------------------------------end--------------------------------------------