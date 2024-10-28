# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:25:26 2024

@author: zpb4
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../Synthetic-Forecast_Verification/src'))
sys.path.insert(0, os.path.abspath('./src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from matplotlib.patches import Polygon
from math import pi
import seaborn as sns
import model
import syn_util
import ensemble_verification_functions as verify
from scipy.stats import rankdata
from scipy.stats import kendalltau as ktau
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from time import localtime, strftime
from datetime import datetime
import matplotlib.dates as mdates
import pickle

col_cb = sns.color_palette('colorblind')
#sns.palplot(col_cb)  #plot coloblind palette for comparison
colv1 = sns.color_palette('PuRd',10)
colv2 = sns.color_palette('YlOrBr',10)

cv2 = colv1[4]  #version 2 is pink-purplish color
cv1 = colv2[6]  #version 1 is orangey-brown color

cv1_pal4 = colv1[2:6]
cv2_pal4 = colv2[4:8]
sns.palplot(cv2_pal4)

cv1_pal4 = sns.color_palette('PuRd',4)
cv2_pal4 = sns.color_palette('YlOrBr',4)

#sns.palplot((col_cb[4],cv1,col_cb[3],cv2)) #base colors pretty close to 'colorblind' palette

kcfs_to_tafd = 2.29568411*10**-5 * 86400
#K = 3524 # TAF
#Rmax = 150 * kcfs_to_tafd # estimate - from MBK
#ramping_rate = 30868/1000 * kcfs_to_tafd # cfs to kcfs to tafd
max_lds = 14

sd = '1990-10-01' 
ed = '2019-08-15'

locs = ['ADO','LAM','NHG','YRS','YRS']
sites = ['ADOC1','LAMC1','NHGC1','NBBC1','ORDC1']
configs = [0,1,2]  #selection of configs from 3 x 3 matrix
config_lab = ['$base$','$mod$','$high$']
plot_idx = np.reshape(np.arange(len(sites)*len(configs)),(len(sites),len(configs)))
abc = list(map(chr, range(97, 123)))
abc_labs = list((abc[i]+')') for i in range(len(abc)))


svers = 'test'

opt_forecast = 'hefs'
syn_samp = 'hefs'
pcntiles = (0.25,0.5,0.75,0.95) 
pcntile_idx = verify.pcntile_fun(pcntiles)
pct_disp = 0.0
nsamps = 100

rr_match_Rmax = True   # if T, scale ramping rates with Rmax ratios, if F, leave at 1

#K_ratios = np.full(9,fill_value=[1,1,1,K_ratio_mid,K_ratio_mid,K_ratio_mid,K_ratio_min,K_ratio_min,K_ratio_min])
#Rmax_ratios = np.full(9,fill_value=[1,Rmax_ratio_mid,Rmax_ratio_min,1,Rmax_ratio_mid,Rmax_ratio_min,1,Rmax_ratio_mid,Rmax_ratio_min])

#K_ratios = np.ones(9)
#Rmax_ratios = np.full(9,fill_value=[1,.9,.8,.7,.6,.5,.4,.3,.2])

syn_vers1 = 'v1'
syn_vers1_param = 'a'
syn_vers1_setup = 'cal'
syn_path1 = '../Synthetic-Forecast-%s-FIRO-DISES' %(syn_vers1) # path to R synthetic forecast repo for 'r-gen' setting below

syn_vers2 = 'v2'
syn_vers2_pct = 0.99
syn_vers2_setup = '5fold'
syn_path2 = '../Synthetic-Forecast-%s-FIRO-DISES' %(syn_vers2) # path to R synthetic forecast repo for 'r-gen' setting below

sim_forecast = 'syn' # what forecast to optimize to? either 'hefs' for actual HEFS hindcast or 'syn' for a synthetic sample
syn_samp = 'Synth1'   # if using a 'syn' sample, this specifies which one to use
tocs_reset = 'none'   # 'tocs' to reset to baseline tocs at beginning of WY, 'firo' to reset to firo pool, 'none' to use continuous storage profile
fixed_pool = True
fixed_pool_value = 0.5
seed = 1

sns.set_theme()
sns.set_style('ticks')
sns.set_context('paper')
plt.rcParams['legend.frameon'] = 'False'

#fig = plt.figure(layout='constrained',figsize=(10,6))
fig = plt.figure(layout='constrained',figsize=(7.5,10))
gs0 = fig.add_gridspec(len(sites),len(configs))

def fig_title(
    fig: matplotlib.figure.Figure, txt: str, loc=(0.5,0.98), fontdict=None, **kwargs
):
    """Alternative to fig.suptitle that behaves like ax.set_title.
    DO NOT use with suptitle.

    See also:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.suptitle.html
    https://stackoverflow.com/a/77063164/8954109
    """
    if fontdict is not None:
        kwargs = {**fontdict, **kwargs}
    if "fontsize" not in kwargs and "size" not in kwargs:
        kwargs["fontsize"] = plt.rcParams["axes.titlesize"]

    if "fontweight" not in kwargs and "weight" not in kwargs:
        kwargs["fontweight"] = plt.rcParams["figure.titleweight"]

    if "verticalalignment" not in kwargs:
        kwargs["verticalalignment"] = "top"
    if "horizontalalignment" not in kwargs:
        kwargs["horizontalalignment"] = "center"

    # Tell the layout engine that our text is using space at the top of the figure
    # so that tight_layout does not break.
    # Is there a more direct way to do this?
    fig.suptitle(" ")
    text = fig.text(loc[0], loc[1], txt, transform=fig.transFigure, in_layout=True, **kwargs)

    return text

#load risk curve data
for s in enumerate(sites):
    loc = locs[s[0]]
    site = sites[s[0]]
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

        K_scale = K * K_ratios[p[1]]
        Rmax_scale = Rmax * Rmax_ratios[p[1]]
        if rr_match_Rmax == True:
            rr_ratios = Rmax_ratios
            ramping_rate_scale = ramping_rate * rr_ratios[p[1]]
        elif rr_match_Rmax == False:
            rr_ratios = np.ones(9)
            ramping_rate_scale = ramping_rate * rr_ratios[p[1]]
    
        pars = pickle.load(open('data/%s/%s/%s_param-risk-thresholds_tocs-reset=%s_fixed=%s-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.pkl'%(loc,site,opt_forecast,tocs_reset,fixed_pool,fixed_pool_value,round(K_ratios[p[1]],2),round(Rmax_ratios[p[1]],2),round(rr_ratios[p[1]],2),seed),'rb'),encoding='latin1')
        risk_curve = syn_util.create_param_risk_curve((pars['lo'],pars['hi'],pars['pwr'],pars['no_risk'],pars['all_risk']),lds=max_lds)

        #load synthetic forecast simulation data
        data1 = np.load('out/%s/%s/sim-array_synforc-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.npz' %(loc,site,syn_vers1+svers,round(K_ratios[p[1]],2),round(Rmax_ratios[p[1]],2),round(rr_ratios[p[1]],2),seed))
        sim_data1 = data1['arr']

        data2 = np.load('out/%s/%s/sim-array_synforc-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.npz' %(loc,site,syn_vers2+svers,round(K_ratios[p[1]],2),round(Rmax_ratios[p[1]],2),round(rr_ratios[p[1]],2),seed))
        sim_data2 = data2['arr']

        """
        #sim_data components across dim 3 for reference
        comb_array[i,:,0]=S
        comb_array[i,:,1]=R
        comb_array[i,:,2]=tocs
        comb_array[i,:,3]=firo
        comb_array[i,:,4]=spill
        comb_array[i,:,5]=Q_cp
        comb_array[i,:,6]=rel_ld
        """

        #extract and simulate for HEFS-firo and baseline
        Q,Qf_hefs,dowy,tocs,df_idx = model.extract(sd,ed,forecast_type=opt_forecast,syn_sample=syn_samp,Rsyn_path=syn_path2,syn_vers=syn_vers2,forecast_param='a',loc=loc,site=site,opt_pcnt=syn_vers2_pct,gen_setup=syn_vers2_setup,K=K_scale)
        Qf_hefs = Qf_hefs[:,:,:max_lds]
        ne = np.shape(Qf_hefs)[1]
        Qf_summed = np.cumsum(Qf_hefs, axis=2)
        Qf_summed_sorted = np.sort(Qf_summed, axis = 1)
        #risk_curve = np.zeros(14)
        ix = ((1 - risk_curve) * (ne)).astype(np.int32)-1
        S_hefs, R_hefs, firo_hefs, spill_hefs, Q_cp_hefs, rel_ld_hefs = model.simulate_nonjit(firo_pool=pars['firo_pool'], ix=ix, Q=Q, Qf=Qf_summed_sorted, dowy=dowy, tocs=tocs, K = K_scale, Rmax = Rmax_scale, ramping_rate = ramping_rate_scale, policy='firo', tocs_reset='none')
        S_base, R_base, firo_base, spill_base, Q_cp_base, rel_ld_base = model.simulate_nonjit(firo_pool=pars['firo_pool'], ix=ix, Q=Q, Qf=Qf_summed_sorted, dowy=dowy, tocs=tocs, K = K_scale, Rmax = Rmax_scale, ramping_rate = ramping_rate_scale, policy='baseline', tocs_reset='none')
        
        #------------------------------------
        #1. plot statistical reliability

        #sset_idx = np.where((dowy > 7) & (dowy < 140)  & (Q > 0))[0]
        sset_idx = np.where(Q > np.sort(Q)[int(pct_disp*len(Q))])[0]
        #sset_idx = np.where(R_hefs > np.sort(R_hefs)[int(pct_disp*len(R_hefs))])[0]
        
        ax1 = fig.add_subplot(gs0[plot_idx[s[0],p[0]]])

        R_zi1,R_Ri1,R_Rel1 = verify.pit_pred(sim_data1[:,sset_idx,1],R_hefs[sset_idx])
        R_zi2,R_Ri2,R_Rel2 = verify.pit_pred(sim_data2[:,sset_idx,1],R_hefs[sset_idx])
        
        ax1.scatter(R_zi1,R_Ri1,s=15,color=cv1,edgecolors='gray',linewidths=0.1,alpha=0.75)
        ax1.scatter(R_zi2,R_Ri2,s=15,color=cv2,edgecolors='gray',linewidths=0.1,alpha=0.75)
        if s[0]==0 and p[0]==0: 
            ax1.legend(['syn-M1','syn-M2'],loc='upper left',fontsize='large')
        if len(sset_idx) > 1000:
            alp = 0.1 
        elif len(sset_idx) <= 1000:
            alp = 0.5
        ax1.scatter(R_zi1,R_Ri1,s=15,color=cv1,edgecolors='gray',linewidths=0.1,alpha=alp)
        ax1.scatter(R_zi2,R_Ri2,s=15,color=cv2,edgecolors='gray',linewidths=0.1,alpha=alp)
        ax1.axline((0,0),(1,1),linewidth=2,color='gray',linestyle='--')
        ax1.text(0.6,0.35,r'$\pi_{rel}: $'+str(round(R_Rel1,3)),color=cv1,fontsize='large',fontweight='bold')
        ax1.text(0.6,0.225,r'$\pi_{rel}: $'+str(round(R_Rel2,3)),color=cv2,fontsize='large',fontweight='bold')
        #ax1.text(0.6,0.225,r'$\tau: $'+str(round(verify.tau_stat(R_zi1)[0],3)),color=cv1,fontsize='medium',fontweight='bold')
        #ax1.text(0.6,0.15,r'$\tau: $'+str(round(verify.tau_stat(R_zi2)[0],3)),color=cv2,fontsize='medium',fontweight='bold')
        ax1.set_xlim([0,1])
        ax1.set_ylim([0,1])
        if s[0] == (len(sites)-1):
            ax1.set_xlabel('$PIT_t$',fontsize='large')
        if s[0] == 0:
            ax1.set_title(config_lab[p[0]],loc='center',fontsize='xx-large',fontweight='bold')
        if p[0] == 0:
            #ax1.set_title(sites[s[0]],loc='left',fontsize='xx-large',fontweight='bold')
            ax2 = ax1.twinx()
            ax2.yaxis.set_label_position('left')
            ax2.spines['left'].set_position(('axes', -0.35))
            ax2.spines['left'].set_visible(False)
            ax2.set_yticks([])
            tt2 = '$\mathrm{\mathbf{%s}}_{%s}$' %(sites[s[0]],int(pct_disp*100))
            ax2.set_ylabel(tt2, rotation=90, size='xx-large',
                   ha='center', va='center',fontweight='bold')
            
        if p[0] == 0:
            ax1.set_ylabel('$rank(PIT_t)/N$')
        ax1.text(0.85,0.05,abc_labs[plot_idx[s[0],p[0]]],fontsize='xx-large',fontweight='bold')
        if s[0] != (len(sites)-1):
            ax1.xaxis.set_ticklabels([])
        if p[0] != 0:
            ax1.yaxis.set_ticklabels([])

fig.savefig('./figures/multsite-multconfig_pqq-rel_R_svers-%s_sites=%s_configs=%s_disp-pct=%s_seed=%s.png' %(svers,sites,configs,pct_disp,seed),dpi=300,bbox_inches='tight')
       
#sys.modules[__name__].__dict__.clear()
#----------------------------------end-----------------------------
