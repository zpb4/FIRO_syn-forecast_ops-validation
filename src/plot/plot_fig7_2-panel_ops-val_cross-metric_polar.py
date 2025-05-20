# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:49:33 2024

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

cv2 = colv1[4]  #version 1 is pink-purplish color
cv1 = colv2[6]  #version 2 is orangey-brown color

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

loc = 'YRS'
site = 'ORDC1'
svers = '5fold'
p = 0
opt_forecast = 'hefs'
syn_samp = 'hefs'
pcntiles = (0.25,0.5,0.75,0.95) 
pcntile_idx = verify.pcntile_fun(pcntiles)
pct_disp1 = 0.9
pct_disp2 = 0.99
pct_thresh1 = 0.9
pct_thresh2 = 0.99
show_vals = False
nsamps = 100

res_params = pickle.load(open('data/%s/%s/reservoir-params.pkl'%(loc,site),'rb'),encoding='latin1')

K = res_params['K']
Rmax = res_params['Rmax']
ramping_rate = res_params['ramping_rate']
K_ratio_min = res_params['Kmin']
Rmax_ratio_min = res_params['Rmaxmin']

rr_match_Rmax = True   # if T, scale ramping rates with Rmax ratios, if F, leave at 1

Rmax_ratio_mid = 1-(1-Rmax_ratio_min)/2
K_ratio_mid = 1-(1-K_ratio_min)/2

K_ratios = np.full(3,fill_value=[1,K_ratio_mid,K_ratio_min])
Rmax_ratios = np.full(3,fill_value=[1,Rmax_ratio_mid,Rmax_ratio_min])

#K_ratios = np.full(9,fill_value=[1,1,1,K_ratio_mid,K_ratio_mid,K_ratio_mid,K_ratio_min,K_ratio_min,K_ratio_min])
#Rmax_ratios = np.full(9,fill_value=[1,Rmax_ratio_mid,Rmax_ratio_min,1,Rmax_ratio_mid,Rmax_ratio_min,1,Rmax_ratio_mid,Rmax_ratio_min])

#K_ratios = np.ones(9)
#Rmax_ratios = np.full(9,fill_value=[1,.9,.8,.7,.6,.5,.4,.3,.2])

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


#load risk curve data
pars = pickle.load(open('data/%s/%s/%s_param-risk-thresholds_tocs-reset=%s_fixed=%s-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.pkl'%(loc,site,opt_forecast,tocs_reset,fixed_pool,fixed_pool_value,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed),'rb'),encoding='latin1')
risk_curve = syn_util.create_param_risk_curve((pars['lo'],pars['hi'],pars['pwr'],pars['no_risk'],pars['all_risk']),lds=max_lds)

#load synthetic forecast simulation data
data1 = np.load('data/%s/%s/sim-array_synforc-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.npz' %(loc,site,syn_vers1+svers,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed))
sim_data1 = data1['arr']

data2 = np.load('data/%s/%s/sim-array_synforc-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.npz' %(loc,site,syn_vers2+svers,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed))
sim_data2 = data2['arr']

#extract and simulate for HEFS-firo and baseline
Q,Qf_hefs,dowy,tocs,df_idx = model.extract(sd,ed,forecast_type=opt_forecast,syn_sample=syn_samp,Rsyn_path=syn_path2,syn_vers=syn_vers2,forecast_param='a',loc=loc,site=site,opt_pcnt='',obj_pwr='',opt_strat='',gen_setup=syn_vers2_setup,K=K_scale)
Qf_hefs = Qf_hefs[:,:,:max_lds]
ne = np.shape(Qf_hefs)[1]
Qf_summed = np.cumsum(Qf_hefs, axis=2)
Qf_summed_sorted = np.sort(Qf_summed, axis = 1)
#risk_curve = np.zeros(14)
ix = ((1 - risk_curve) * (ne)).astype(np.int32)-1
S_hefs, R_hefs, firo_hefs, spill_hefs, Q_cp_hefs, rel_ld_hefs = model.simulate_nonjit(firo_pool=pars['firo_pool'], ix=ix, Q=Q, Qf=Qf_summed_sorted, dowy=dowy, tocs=tocs, K = K_scale, Rmax = Rmax_scale, ramping_rate = ramping_rate_scale, policy='firo', tocs_reset='none')
S_base, R_base, firo_base, spill_base, Q_cp_base, rel_ld_base = model.simulate_nonjit(firo_pool=pars['firo_pool'], ix=ix, Q=Q, Qf=Qf_summed_sorted, dowy=dowy, tocs=tocs, K = K_scale, Rmax = Rmax_scale, ramping_rate = ramping_rate_scale, policy='baseline', tocs_reset='none')

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

#-----------------------------------------------------------------------------------------------------
sns.set_theme()
sns.set_style('ticks')
sns.set_context('paper')
plt.rcParams['legend.frameon'] = 'False'


sset_idx = np.where(Q > np.sort(Q)[int(pct_disp1*len(Q))])[0]
#sset_idx = np.where(R_hefs > np.sort(R_hefs)[int(pct_disp1*len(R_hefs))])[0]
sset_idx2 = np.where(R_hefs > 0)[0]

R_thresh = np.sort(R_hefs)[int(pct_thresh1*len(Q))]
bns = 5
bins = [0.5,3.5,6.5,9.5,14.5]

theta = np.linspace(0, 2 * np.pi, 5)

Spit,Srnk,Srel1 = verify.pit_pred(sim_data1[:,sset_idx,0],S_hefs[sset_idx]) #rel
Spit,Srnk,Srel2 = verify.pit_pred(sim_data2[:,sset_idx,0],S_hefs[sset_idx]) #rel
Rpit,Rrnk,Rrel1 = verify.pit_pred(sim_data1[:,sset_idx,1],R_hefs[sset_idx]) #rel
Rpit,Rrnk,Rrel2 = verify.pit_pred(sim_data2[:,sset_idx,1],R_hefs[sset_idx]) #rel
bs_rel1,bs_res1,bs_unc1 = verify.brier_scores(ensemble=sim_data1[:,sset_idx2,1], tgt=R_hefs[sset_idx2], bins=bns, threshold=R_thresh)
bs_rel2,bs_res2,bs_unc2 = verify.brier_scores(ensemble=sim_data2[:,sset_idx2,1], tgt=R_hefs[sset_idx2], bins=bns, threshold=R_thresh)
chisq1,p1,freq_ens1,freq_tgt,freq_err1 = verify.ens_chisq_bin_specify(ensemble=sim_data1[:,sset_idx,6], tgt=rel_ld_hefs[sset_idx],bins=bins) # no of cats is number of leads + 1 for the '0 lead' non-FIRO constraint based releases
chisq2,p2,freq_ens2,freq_tgt,freq_err2 = verify.ens_chisq_bin_specify(ensemble=sim_data2[:,sset_idx,6], tgt=rel_ld_hefs[sset_idx],bins=bins) # no of cats is number of leads + 1 for the '0 lead' non-FIRO constraint based releases


fig = plt.figure(layout='constrained',figsize=(6,6))
gs0 = fig.add_gridspec(1,2)
ax1 = fig.add_subplot(gs0[0],polar=True)

#matplotlib.rcParams['text.usetex']=True

if show_vals == True:
    Srel_nm = '\n'.join((
        r'$S_{\pi_{rel}}$',
        r'$[%s]$' %(round(max(Srel1,Srel2),3))
        ))

    Rrel_nm = '\n'.join((
        r'$R_{\pi_{rel}}$',
        r'$[%s]$' %(round(max(Rrel1,Rrel2),3))
        ))

    Rbsrel_nm = '\n'.join((
        r'$R_{BS_{rel}}$',
        r'$[%s]$' %(round(max(bs_rel1,bs_rel2)*1e2,3))
        ))

    chisq_nm = '\n'.join((
        r'$R^{ld}_{chi^2}$',
        r'$[%s]$' %(round(max(chisq1,chisq2),2))
        ))

    names = [Srel_nm,Rrel_nm,Rbsrel_nm,chisq_nm]

else:
    names = ['$S_{\pi_{rel}}$','$R_{\pi_{rel}}$','$R_{BS_{rel}}$','$R^{ld}_{\chi^2}$']

lines, labels = plt.thetagrids(range(0, 360, int(360/4)), (names),fontsize='large')

ax1.plot(theta,(Srel1/max(Srel1,Srel2),Rrel1/max(Rrel1,Rrel2),bs_rel1/max(bs_rel1,bs_rel2),chisq1/max(chisq1,chisq2),Srel1/max(Srel1,Srel2)),color=cv1,linewidth=2)
ax1.plot(theta,(Srel2/max(Srel1,Srel2),Rrel2/max(Rrel1,Rrel2),bs_rel2/max(bs_rel1,bs_rel2),chisq2/max(chisq1,chisq2),Srel2/max(Srel1,Srel2)),color=cv2,linewidth=2)
ax1.legend(labels=('syn-M1', 'syn-M2'), loc=(-0.5,.95),fontsize='large')
tt1 = '$\mathrm{\mathbf{%s}}_{%s}$' %(site,'top'+str(round((1-pct_disp1)*100))+'\%')
ax1.set_title(tt1,fontsize='x-large',fontweight='bold')
ax1.text(3.9,1.5,'a)',fontsize='xx-large',fontweight='bold')
if show_vals == True:
    ax1.tick_params(pad=10)
#fig_title(fig,'%s - %s' %(site,int(pct_disp1*100))+'%',loc=(-0.01,0.725),fontsize='x-large',fontweight='bold',rotation=90,ha='center',va='center')


sset_idx = np.where(Q > np.sort(Q)[int(pct_disp2*len(Q))])[0]
#sset_idx = np.where(R_hefs > np.sort(R_hefs)[int(pct_disp2*len(R_hefs))])[0]
sset_idx2 = np.where(R_hefs > 0)[0]
R_thresh = np.sort(R_hefs)[int(pct_thresh2*len(Q))]

Spit,Srnk,Srel1 = verify.pit_pred(sim_data1[:,sset_idx,0],S_hefs[sset_idx]) #rel
Spit,Srnk,Srel2 = verify.pit_pred(sim_data2[:,sset_idx,0],S_hefs[sset_idx]) #rel
Rpit,Rrnk,Rrel1 = verify.pit_pred(sim_data1[:,sset_idx,1],R_hefs[sset_idx]) #rel
Rpit,Rrnk,Rrel2 = verify.pit_pred(sim_data2[:,sset_idx,1],R_hefs[sset_idx]) #rel
bs_rel1,bs_res1,bs_unc1 = verify.brier_scores(ensemble=sim_data1[:,sset_idx2,1], tgt=R_hefs[sset_idx2], bins=bns, threshold=R_thresh)
bs_rel2,bs_res2,bs_unc2 = verify.brier_scores(ensemble=sim_data2[:,sset_idx2,1], tgt=R_hefs[sset_idx2], bins=bns, threshold=R_thresh)
chisq1,p1,freq_ens1,freq_tgt,freq_err1 = verify.ens_chisq_bin_specify(ensemble=sim_data1[:,sset_idx,6], tgt=rel_ld_hefs[sset_idx],bins=bins) # no of cats is number of leads + 1 for the '0 lead' non-FIRO constraint based releases
chisq2,p2,freq_ens2,freq_tgt,freq_err2 = verify.ens_chisq_bin_specify(ensemble=sim_data2[:,sset_idx,6], tgt=rel_ld_hefs[sset_idx],bins=bins) # no of cats is number of leads + 1 for the '0 lead' non-FIRO constraint based releases



ax2 = fig.add_subplot(gs0[1],polar=True)

if show_vals == True:
    Srel_nm = '\n'.join((
        r'$S_{\pi_{rel}}$',
        r'$[%s]$' %(round(max(Srel1,Srel2),3))
        ))

    Rrel_nm = '\n'.join((
        r'$R_{\pi_{rel}}$',
        r'$[%s]$' %(round(max(Rrel1,Rrel2),3))
        ))

    Rbsrel_nm = '\n'.join((
        r'$R_{BS_{rel}}$',
        r'$[%s]$' %(round(max(bs_rel1,bs_rel2)*1e2,3))
        ))

    chisq_nm = '\n'.join((
        r'$R^{ld}_{chi^2}$',
        r'$[%s]$' %(round(max(chisq1,chisq2),2))
        ))

    names = [Srel_nm,Rrel_nm,Rbsrel_nm,chisq_nm]

else:
    names = ['$S_{\pi_{rel}}$','$R_{\pi_{rel}}$','$R_{BS_{rel}}$','$R^{ld}_{\chi^2}$']
    
lines, labels = plt.thetagrids(range(0, 360, int(360/4)), (names),fontsize='large')

ax2.plot(theta,(Srel1/max(Srel1,Srel2),Rrel1/max(Rrel1,Rrel2),bs_rel1/max(bs_rel1,bs_rel2),chisq1/max(chisq1,chisq2),Srel1/max(Srel1,Srel2)),color=cv1,linewidth=2)
ax2.plot(theta,(Srel2/max(Srel1,Srel2),Rrel2/max(Rrel1,Rrel2),bs_rel2/max(bs_rel1,bs_rel2),chisq2/max(chisq1,chisq2),Srel2/max(Srel1,Srel2)),color=cv2,linewidth=2)
#ax2.legend(labels=('sHEFS-V1', 'sHEFS-V2'), loc=(-0.2,0.95))
tt2 = '$\mathrm{\mathbf{%s}}_{%s}$' %(site,'top'+str(round((1-pct_disp2)*100))+'\%')
ax2.set_title(tt2,fontsize='x-large',fontweight='bold')
ax2.text(3.9,1.5,'b)',fontsize='xx-large',fontweight='bold')
if show_vals == True:
    ax2.tick_params(pad=10)
#fig_title(fig,'%s - %s' %(site,int(pct_disp1*100))+'%',loc=(-0.01,0.725),fontsize='x-large',fontweight='bold',rotation=90,ha='center',va='center')


fig.savefig('./figs/%s/%s/1x2-ops-val-polar_cross-metric_svers-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s_disp-pct1=%s,pct2=%s_show-vals=%s.png' %(loc,site,svers,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed,pct_disp1,pct_disp2,show_vals),dpi=300,bbox_inches='tight')


#sys.modules[__name__].__dict__.clear()



#-----------------------------------------------------------------END--------------------------------------------------------------------