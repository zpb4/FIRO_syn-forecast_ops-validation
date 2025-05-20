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
import matplotlib as matplotlib
import matplotlib.pyplot as plt
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

sns.palplot((col_cb[4],cv1,col_cb[3],cv2)) #base colors pretty close to 'colorblind' palette
sns.palplot(col_cb)

kcfs_to_tafd = 2.29568411*10**-5 * 86400
#K = 3524 # TAF
#Rmax = 150 * kcfs_to_tafd # estimate - from MBK
#ramping_rate = 30868/1000 * kcfs_to_tafd # cfs to kcfs to tafd
max_lds = 14

sd = '1990-10-01' 
ed = '2019-08-15'

loc = 'YRS'
site = 'NBBC1'
svers = '5fold'
p = 0
opt_forecast = 'hefs'
syn_samp = 'hefs'
pcntiles = (0.25,0.5,0.75,0.95) 
pcntile_idx = verify.pcntile_fun(pcntiles)
pct_disp1 = 0.9
pct_disp2 = 0.99
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

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

#---------------------------------------------------
#Reliability plots for release leads, Brier score, and chisquared test
#rel_ld_idx = np.where((dowy_hefs > 60) & (dowy_hefs < 170))[0] # & (Q_hefs > 0))[0]
rel_ld_idx1 = np.where(Q > np.sort(Q)[int(pct_disp1*len(Q))])[0]
#rel_ld_idx1 = np.where(R_hefs > np.sort(R_hefs)[int(pct_disp1*len(R_hefs))])[0]

rel_ld_sset = rel_ld_hefs[rel_ld_idx1]
ens_rel_ld_sset1 = sim_data1[:,rel_ld_idx1,6]
ens_rel_ld_sset2 = sim_data2[:,rel_ld_idx1,6]

#rel_ld_ssetd = rel_ld_hefsd[rel_ld_idx]
#ens_rel_ld_sset1d = sim_data1d[:,rel_ld_idx,6]
#ens_rel_ld_sset2d = sim_data2d[:,rel_ld_idx,6]

#rel_ld_sset = rel_ld_hefs[:]
#ens_rel_ld_sset1 = sim_data1[:,:,6]
#ens_rel_ld_sset2 = sim_data2[:,:,6]

bins = [0.5,3.5,6.5,9.5,14.5]
#bins = [0.5,5.5,10.5,15.5]
nb = len(bins)-1
bn_labs = ('ld 1-3','ld 4-6','ld 7-9','ld 10-14')
#bn_labs = ('ld 1-5','ld 5-10','ld 11-15')

chisq1,p1,freq_ens1,freq_tgt,freq_err1 = verify.ens_chisq_bin_specify(ensemble=ens_rel_ld_sset1, tgt=rel_ld_sset,bins=bins) # no of cats is number of leads + 1 for the '0 lead' non-FIRO constraint based releases
bscore1,bskill1 = verify.brier_score_binsmulticat(ensemble=ens_rel_ld_sset1, tgt=rel_ld_sset, bins=bins)

chisq2,p2,freq_ens2,freq_tgt,freq_err2 = verify.ens_chisq_bin_specify(ensemble=ens_rel_ld_sset2, tgt=rel_ld_sset,bins=bins) # no of cats is number of leads + 1 for the '0 lead' non-FIRO constraint based releases
bscore2,bskill2 = verify.brier_score_binsmulticat(ensemble=ens_rel_ld_sset2, tgt=rel_ld_sset, bins=bins)

#chisq1d,p1d,freq_ens1d,freq_tgtd,freq_err1d = verify.ens_chisq_bin_specify(ensemble=ens_rel_ld_sset1d, tgt=rel_ld_ssetd,bins=bins) # no of cats is number of leads + 1 for the '0 lead' non-FIRO constraint based releases
#bscore1d,bskill1d = verify.brier_score_binsmulticat(ensemble=ens_rel_ld_sset1d, tgt=rel_ld_ssetd, bins=bins)

#chisq2d,p2d,freq_ens2d,freq_tgtd,freq_err2d = verify.ens_chisq_bin_specify(ensemble=ens_rel_ld_sset2d, tgt=rel_ld_ssetd,bins=bins) # no of cats is number of leads + 1 for the '0 lead' non-FIRO constraint based releases
#bscore2d,bskill2d = verify.brier_score_binsmulticat(ensemble=ens_rel_ld_sset2d, tgt=rel_ld_ssetd, bins=bins)


sns.set_theme()
sns.set_style('ticks')
sns.set_context('paper')
plt.rcParams['legend.frameon'] = 'False'
plt.rcParams['figure.figsize'] = [8,8]

bns=4

x1 = np.arange(nb)+0.75
x2 = np.arange(nb)+1.25

y1 = freq_tgt
y2 = freq_ens1

yerr1 = freq_err1

fig = plt.figure(layout='constrained',figsize=(6,6))
gs0 = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs0[0])

#plt.rcParams['figure.figsize'] = [4,3]
ax1.bar(x1,y1,color='blue',width=0.4)
ax1.bar(x2,y2,color=cv1,width=0.4,yerr=yerr1)
ax1.legend(['HEFS','syn-M1'],loc='upper center',fontsize='x-large')
#plt.xlabel('Release leads')
ax1.set_title('Release leads $(R^{ld})$ histogram',fontsize='large')
ax1.set_ylabel('Density',fontsize='large')
ax1.set_ylim([0,0.35])
ax1.text(2.5,0.225,r'$\chi^2: $'+str(round(chisq1,2)),ha='center',fontsize='x-large')
if (1-p1) > 1E-4:
    ax1.text(2.5,0.185,'$p: $'+str(round(1-p1,4)),ha='center',fontsize='x-large')
else:
    ax1.text(2.5,0.185,'$p\lll0$',ha='center',fontsize='x-large')
ax1.set_xticks(ticks=(x1+0.25))
ax1.xaxis.set_ticklabels([])
ax1.tick_params(axis='both',which='major',labelsize='large')
ax1.text(nb+0.25,0.31,'a)',fontsize='xx-large',fontweight='bold')
#tt1 = '%s - %s' %(site,int(pct_disp1*100))+'%'
tt1 = '$\mathrm{\mathbf{%s}}_{%s}$' %(site,'top'+str(round((1-pct_disp1)*100))+'\%')
fig_title(fig,tt1,loc=(-0.01,0.725),fontsize='xx-large',fontweight='bold',rotation=90,ha='center',va='center')

ax2 = fig.add_subplot(gs0[1])

y1 = freq_tgt
y2 = freq_ens2

yerr2 = freq_err2

#plt.rcParams['figure.figsize'] = [4,3]
ax2.bar(x1,y1,color='blue',width=0.4)
ax2.bar(x2,y2,color=cv2,width=0.4,yerr=yerr2)
ax2.legend(['HEFS','syn-M2'],loc='upper center',fontsize='x-large')
#plt.xlabel('Release leads')
ax2.set_title('Release leads $(R^{ld})$ histogram',fontsize='large')
#ax2.set_ylabel('Density')
ax2.set_ylim([0,0.35])
ax2.text(2.5,0.225,r'$\chi^2: $'+str(round(chisq2,2)),ha='center',fontsize='x-large')
if (1-p2) > 1E-4:
    ax2.text(2.5,0.185,'$p: $'+str(round(1-p2,4)),ha='center',fontsize='x-large')
else:
    ax2.text(2.5,0.185,'$p\lll0$',ha='center',fontsize='x-large')
ax2.text(nb+0.25,0.31,'b)',fontsize='xx-large',fontweight='bold')
ax2.set_xticks(ticks=(x1+0.25))
ax2.xaxis.set_ticklabels([])
ax2.yaxis.set_ticklabels([])
ax2.tick_params(axis='both',which='major',labelsize='large')

rel_ld_idx2 = np.where(Q > np.sort(Q)[int(pct_disp2*len(Q))])[0]
#rel_ld_idx2 = np.where(R_hefs > np.sort(R_hefs)[int(pct_disp2*len(R_hefs))])[0]

rel_ld_sset = rel_ld_hefs[rel_ld_idx2]
ens_rel_ld_sset1 = sim_data1[:,rel_ld_idx2,6]
ens_rel_ld_sset2 = sim_data2[:,rel_ld_idx2,6]

chisq1,p1,freq_ens1,freq_tgt,freq_err1 = verify.ens_chisq_bin_specify(ensemble=ens_rel_ld_sset1, tgt=rel_ld_sset,bins=bins) # no of cats is number of leads + 1 for the '0 lead' non-FIRO constraint based releases
bscore1,bskill1 = verify.brier_score_binsmulticat(ensemble=ens_rel_ld_sset1, tgt=rel_ld_sset, bins=bins)

chisq2,p2,freq_ens2,freq_tgt,freq_err2 = verify.ens_chisq_bin_specify(ensemble=ens_rel_ld_sset2, tgt=rel_ld_sset,bins=bins) # no of cats is number of leads + 1 for the '0 lead' non-FIRO constraint based releases
bscore2,bskill2 = verify.brier_score_binsmulticat(ensemble=ens_rel_ld_sset2, tgt=rel_ld_sset, bins=bins)

ax3 = fig.add_subplot(gs0[2])

y1 = freq_tgt
y2 = freq_ens1

yerr1 = freq_err1

ax3.bar(x1,y1,color='blue',width=0.4)
ax3.bar(x2,y2,color=cv1,width=0.4,yerr=yerr1)
#ax3.legend(['HEFS','sHEFS-V1'],loc='upper center')
#plt.xlabel('Release leads')
#ax3.set_title('Release leads histogram')
ax3.set_ylabel('Density',fontsize='large')
ax3.set_ylim([0,0.35])
ax3.text(2.5,0.3,r'$\chi^2: $'+str(round(chisq1,2)),ha='center',fontsize='x-large')
if (1-p1) > 1E-4:
    ax3.text(2.5,0.26,'$p: $'+str(round(1-p1,4)),ha='center',fontsize='x-large')
else:
    ax3.text(2.5,0.26,'$p\lll0$',ha='center',fontsize='x-large')
ax3.text(nb+0.25,0.31,'c)',fontsize='xx-large',fontweight='bold')
ax3.set_xticks(ticks=(x1+0.25))
ax3.xaxis.set_ticklabels(bn_labs)
ax3.tick_params(axis='both',which='major',labelsize='large')
#tt2 = '%s - %s' %(site,int(pct_disp2*100))+'%'
tt2 = '$\mathrm{\mathbf{%s}}_{%s}$' %(site,'top'+str(round((1-pct_disp2)*100))+'\%')
fig_title(fig,tt2,loc=(-0.01,0.25),fontsize='xx-large',fontweight='bold',rotation=90,ha='center',va='center')

ax4 = fig.add_subplot(gs0[3])

y1 = freq_tgt
y2 = freq_ens2

yerr1 = freq_err2

ax4.bar(x1,y1,color='blue',width=0.4)
ax4.bar(x2,y2,color=cv2,width=0.4,yerr=yerr1)
#ax4.legend(['HEFS','sHEFS-V2'],loc='upper center')
#plt.xlabel('Release leads')
#ax4.set_title('Release leads histogram')
#ax4.set_ylabel('Density')
ax4.set_ylim([0,0.35])
ax4.text(2.5,0.3,r'$\chi^2: $'+str(round(chisq2,2)),ha='center',fontsize='x-large')
if (1-p2) > 1E-4:
    ax4.text(2.5,0.26,'$p: $'+str(round(1-p2,4)),ha='center',fontsize='x-large')
else:
    ax4.text(2.5,0.26,'$p\lll0$',ha='center',fontsize='x-large')
ax4.text(nb+0.25,0.31,'d)',fontsize='xx-large',fontweight='bold')
ax4.set_xticks(ticks=(x1+0.25))
ax4.xaxis.set_ticklabels(bn_labs)
ax4.yaxis.set_ticklabels([])
ax4.tick_params(axis='both',which='major',labelsize='large')

fig.savefig('./figs/%s/%s/2x2-rel-lead-plot_binned_svers-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s_disp-pct1=%s,pct2=%s.png' %(loc,site,svers,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed,pct_disp1,pct_disp2),dpi=300,bbox_inches='tight')


#sys.modules[__name__].__dict__.clear()

#----------------------------------end-----------------------------
