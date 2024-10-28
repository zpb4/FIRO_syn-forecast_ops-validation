# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:47:45 2024

@author: zpb4
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util
import syn_util
from time import localtime, strftime
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle

col_cb = sns.color_palette('colorblind')

kcfs_to_tafd = 2.29568411*10**-5 * 86400
#K = 3524 # TAF
Rmax = 150 * kcfs_to_tafd # estimate - from MBK
ramping_rate = 30868/1000 * kcfs_to_tafd # cfs to kcfs to tafd

max_lds = 14

locs = ['ADO','LAM','NHG','YRS','YRS']
sites = ['ADOC1','LAMC1','NHGC1','NBBC1','ORDC1']
configs = [0,1,2]  #selection of configs from 3 x 3 matrix
#config_lab = ['$base\_constr$','$mod\_constr$','$high\_constr$']
config_lab = ['$base$','$mod$','$high$']
fig_lab = ['a)','b)','c)','d)','e)']


opt_forecast = 'hefs' # what forecast to optimize to? either 'hefs' for actual HEFS hindcast or 'syn' for a synthetic sample
syn_samp = 'hefs'   # if using a 'syn' sample, this specifies which one to use
tocs_reset = 'none'   # 'tocs' to reset to baseline tocs at beginning of WY, 'firo' to reset to firo pool, 'none' to use continuous storage profile
fixed_pool = True
fixed_pool_value = 0.5
seed = 1

rr_match_Rmax = True   # if T, scale ramping rates with Rmax ratios, if F, leave at 1

#Rmax_ratio_mid = 1-(1-Rmax_ratio_min)/2
#K_ratio_mid = 1-(1-K_ratio_min)/2

#K_ratios = np.full(9,fill_value=[1,1,1,K_ratio_mid,K_ratio_mid,K_ratio_mid,K_ratio_min,K_ratio_min,K_ratio_min])
#Rmax_ratios = np.full(9,fill_value=[1,Rmax_ratio_mid,Rmax_ratio_min,1,Rmax_ratio_mid,Rmax_ratio_min,1,Rmax_ratio_mid,Rmax_ratio_min])

risk_curve_arr = np.zeros((len(sites),len(configs),max_lds))

for i in range(len(sites)):

    loc = locs[i]
    site = sites[i]

    res_params = pickle.load(open('data/%s/%s/reservoir-params.pkl'%(loc,site),'rb'),encoding='latin1')

    K = res_params['K']
    Rmax = res_params['Rmax']
    ramping_rate = res_params['ramping_rate']
    K_ratio_min = res_params['Kmin']
    Rmax_ratio_min = res_params['Rmaxmin']
    
    Rmax_ratio_mid = 1-(1-Rmax_ratio_min)/2
    K_ratio_mid = 1-(1-K_ratio_min)/2
    
    K_ratios = np.full(3,fill_value=[1,K_ratio_mid,K_ratio_min])
    Rmax_ratios = np.full(3,fill_value=[1,Rmax_ratio_mid,Rmax_ratio_min])
    
    for p in enumerate(configs):
    
        K = K * K_ratios[p[1]]
        Rmax = Rmax * Rmax_ratios[p[1]]
    
        if rr_match_Rmax == True:
            rr_ratios = Rmax_ratios
            ramping_rate = ramping_rate * rr_ratios[p[1]]
        elif rr_match_Rmax == False:
            rr_ratios = np.ones(9)

        #load and print risk curve
        pars = pickle.load(open('data/%s/%s/%s_param-risk-thresholds_tocs-reset=%s_fixed=%s-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.pkl'%(loc,site,syn_samp,tocs_reset,fixed_pool,fixed_pool_value,round(K_ratios[p[1]],2),round(Rmax_ratios[p[1]],2),round(rr_ratios[p[1]],2),seed),'rb'),encoding='latin1')
        risk_curve = syn_util.create_param_risk_curve((pars['lo'],pars['hi'],pars['pwr'],pars['no_risk'],pars['all_risk']),lds=max_lds)
    
        risk_curve_arr[i,p[0],:] = risk_curve

sns.set_theme()
sns.set_style('ticks')
sns.set_context('paper')

fig = plt.figure(layout='constrained',figsize=(9,2.25))
gs0 = fig.add_gridspec(1,5)
cols = [2,4,3]

for p in enumerate(sites):
    
    ax1 = fig.add_subplot(gs0[p[0]])
    for k in enumerate(configs):
        ax1.plot(np.arange(max_lds+1)[1:],risk_curve_arr[p[0],k[0],:],color=col_cb[cols[k[0]]])
    if p[0] == 0:
        ax1.legend(config_lab,loc='upper left',fontsize='medium')
    ax1.set_xlim([1,max_lds])
    ax1.set_ylim([0,1])
    ax1.set_xlabel('Lead time (days)')
    ax1.text(max_lds-2.5,0.85,fig_lab[p[0]],fontsize='xx-large',fontweight='bold')
    if p[0] == 0:
        ax1.set_ylabel('Risk threshold')
    #ax1.text(2,0.95,'$K_{ratio}$: '+str(round(K_ratios[p[1]],2)))
    #ax1.text(2,0.9,'$Rmax_{ratio}$: '+str(round(Rmax_ratios[p[1]],2)))
    #ax1.set_title(config_lab[p[0]] + ':  $K^{frac}$ = %s $R^{frac}_{max}$ = %s' %(round(K_ratios[p[1]],2),round(Rmax_ratios[p[1]],2)))
    ax1.set_title(sites[p[0]],fontsize='x-large',fontweight='bold')
    if p[0] > 0:
        ax1.yaxis.set_ticklabels([])


#plt.suptitle(site,fontsize='xx-large',fontweight='bold')
#fig.savefig('./figures/3-panel_risk-curve-comp_Kmin=%s_Rmin=%s_seed-%s.png' %(round(K_ratio_min,2),round(Rmax_ratio_min,2),seed),dpi=300,bbox_inches='tight')
fig.savefig('./figures/5-panel_risk-curve-comp_seed-%s.png' %(seed),dpi=300,bbox_inches='tight')
plt.show()


#-------------------------------------------------------------end------------------------------------------------------------------------
