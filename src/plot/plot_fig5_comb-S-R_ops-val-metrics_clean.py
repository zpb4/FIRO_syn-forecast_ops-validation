# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:25:26 2024

@author: zpb4
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../Synthetic-Forecast_Verification/src'))
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

loc = 'YRS'
site = 'NBBC1'
svers = 'test'
p = 0
opt_forecast = 'hefs'
syn_samp = 'hefs'
pcntiles = (0.25,0.5,0.75,0.95) 
pcntile_idx = verify.pcntile_fun(pcntiles)
pct_disp1 = 0.9
pct_thresh1=0.9
pct_disp2 = 0.99
pct_thresh2=0.99
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

"""
#demand calculations assuming baseline policy storage with no demand based releases
dmd_par = pickle.load(open('./data/demand-data.pkl','rb'),encoding='latin1')

dmd_t = model.simulate_baseline_demand(Q=Q_hefs, Q_MSG=Q_MSG_hefs, Qf=Qf_hefs, Qf_MSG=Qf_MSG_hefs, dowy=dowy_hefs, tocs=tocs_inp, dmd_params=dmd_par)[6]
#o/w, do not include demand in optimization
no_dmd_t = np.zeros_like(Q_hefs)
"""

#load synthetic forecast simulation data
data1 = np.load('out/%s/%s/sim-array_synforc-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.npz' %(loc,site,syn_vers1+svers,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed))
sim_data1 = data1['arr']

data2 = np.load('out/%s/%s/sim-array_synforc-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s.npz' %(loc,site,syn_vers2+svers,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed))
sim_data2 = data2['arr']

"""
#load synthetic forecast simulation data
mod_versd = mod_vers + 'd'
data = np.load('data/comb_sim-array_synforc-%s_modvers-%s.npz' %(syn_vers1+svers1,mod_versd))
sim_data1d = data['arr']

data = np.load('data/comb_sim-array_synforc-%s_modvers-%s.npz' %(syn_vers2+svers2,mod_versd))
sim_data2d = data['arr']
"""
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


S_rel1 = np.empty((len(pcntiles),np.shape(S_hefs)[0]))
R_rel1 = np.empty((len(pcntiles),np.shape(S_hefs)[0]))
Q_cp_rel1 = np.empty((len(pcntiles),np.shape(S_hefs)[0]))
rel_ld_rel1 = np.empty((len(pcntiles),np.shape(S_hefs)[0]))

S_rel2 = np.empty((len(pcntiles),np.shape(S_hefs)[0]))
R_rel2 = np.empty((len(pcntiles),np.shape(S_hefs)[0]))
Q_cp_rel2 = np.empty((len(pcntiles),np.shape(S_hefs)[0]))
rel_ld_rel2 = np.empty((len(pcntiles),np.shape(S_hefs)[0]))

"""
S_rel1d = np.empty((len(pcntiles),np.shape(S_hefs)[0]))
R_rel1d = np.empty((len(pcntiles),np.shape(S_hefs)[0]))
Q_cp_rel1d = np.empty((len(pcntiles),np.shape(S_hefs)[0]))
rel_ld_rel1d = np.empty((len(pcntiles),np.shape(S_hefs)[0]))

S_rel2d = np.empty((len(pcntiles),np.shape(S_hefs)[0]))
R_rel2d = np.empty((len(pcntiles),np.shape(S_hefs)[0]))
Q_cp_rel2d = np.empty((len(pcntiles),np.shape(S_hefs)[0]))
rel_ld_rel2d = np.empty((len(pcntiles),np.shape(S_hefs)[0]))

"""

for i in range(len(S_hefs)):
    for k in range(len(pcntiles)):
        S_rel1[k,i] = verify.percentile_rel(sim_data1[:,i,0],S_hefs[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        R_rel1[k,i] = verify.percentile_rel(sim_data1[:,i,1],R_hefs[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        Q_cp_rel1[k,i] = verify.percentile_rel(sim_data1[:,i,5],Q_cp_hefs[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        rel_ld_rel1[k,i] = verify.percentile_rel(sim_data1[:,i,6],rel_ld_hefs[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        
        S_rel2[k,i] = verify.percentile_rel(sim_data2[:,i,0],S_hefs[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        R_rel2[k,i] = verify.percentile_rel(sim_data2[:,i,1],R_hefs[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        Q_cp_rel2[k,i] = verify.percentile_rel(sim_data2[:,i,5],Q_cp_hefs[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        rel_ld_rel2[k,i] = verify.percentile_rel(sim_data2[:,i,6],rel_ld_hefs[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        
        """
        S_rel1d[k,i] = verify.percentile_rel(sim_data1d[:,i,0],S_hefsd[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        R_rel1d[k,i] = verify.percentile_rel(sim_data1d[:,i,1],R_hefsd[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        Q_cp_rel1d[k,i] = verify.percentile_rel(sim_data1d[:,i,5],Q_cp_hefsd[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        rel_ld_rel1d[k,i] = verify.percentile_rel(sim_data1d[:,i,6],rel_ld_hefsd[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        
        S_rel2d[k,i] = verify.percentile_rel(sim_data2d[:,i,0],S_hefsd[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        R_rel2d[k,i] = verify.percentile_rel(sim_data2d[:,i,1],R_hefsd[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        Q_cp_rel2d[k,i] = verify.percentile_rel(sim_data2d[:,i,5],Q_cp_hefsd[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        rel_ld_rel2d[k,i] = verify.percentile_rel(sim_data2d[:,i,6],rel_ld_hefsd[i],pcntile_idx[k][0],pcntile_idx[k][1])[2]
        """
   

#/////////////////////////////////////////////////////////////////////////////
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

"""
2 x 2 plot of stats reliability, PQQ plot, event-based reliability, and BSE diagram
"""
#------------------------------------
#1. 2 x 2 plot of S, R, Q_cp, rel_lds reliability for different percentiles

#sset_idx = np.where((dowy > 7) & (dowy < 140)  & (Q > 0))[0]
sset_idx = np.where(Q > np.sort(Q)[int(pct_disp1*len(Q))])[0]
#sset_idx = np.where(R_hefs > np.sort(R_hefs)[int(pct_disp1*len(R_hefs))])[0]

#t = np.sort(R_hefs[sset_idx])

sns.set_theme()
sns.set_style('ticks')
sns.set_context('paper')
plt.rcParams['legend.frameon'] = 'False'

x = np.zeros(len(pcntiles)+1)
y = np.zeros(len(pcntiles)+1)
x[1:] = pcntiles

Qcp_rel_out1 = np.zeros_like(x)
Qcp_rel_out2 = np.zeros_like(x)
#Qcp_rel_out1d = np.zeros_like(x)
#Qcp_rel_out2d = np.zeros_like(x)

for i in range(len(pcntiles)):
    Qcp_rel_out1[i+1] = np.sum(Q_cp_rel1[i,sset_idx])/len(sset_idx)
    Qcp_rel_out2[i+1] = np.sum(Q_cp_rel2[i,sset_idx])/len(sset_idx)
    #Qcp_rel_out1d[i+1] = np.sum(Q_cp_rel1d[i,sset_idx])/len(sset_idx)
    #Qcp_rel_out2d[i+1] = np.sum(Q_cp_rel2d[i,sset_idx])/len(sset_idx)
    
#fig = plt.figure(layout='constrained',figsize=(10,6))
fig = plt.figure(layout='constrained',figsize=(9,11))
gs0 = fig.add_gridspec(4,3)
ax1 = fig.add_subplot(gs0[0])

#plt.title('Simulation w/o Demand',rotation=90,loc='center',y=0.5,fontsize='xx-large',fontweight='bold')
#panel 1: Statistical reliability plot
ax1.plot(x,Qcp_rel_out1,color=cv1,linewidth=3)
ax1.plot(x,Qcp_rel_out2,color=cv2,linewidth=3)
ax1.legend(['syn-M1','syn-M2'],loc='upper left',fontsize='large')
ax1.set_xlim([0,1])
ax1.set_ylim([0,1])
ax1.axline((0,0),slope=1,color='gray',alpha=0.5,linestyle='--')
ax1.set_title('Release CP plot',fontsize='large')

#ax1.set_xlabel('Coverage interval')
ax1.set_ylabel('Coverage probability (CP)',fontsize='large')
ax1.text(0.95,0.05,'a)',fontsize='xx-large',fontweight='bold',ha='right')
ax1.xaxis.set_ticklabels([])
ax1.tick_params(axis='both',which='major',labelsize='large')
ax1.xaxis.set_ticks([0,0.2,0.4,0.6,0.8,1])
#ax1.yaxis.set_ticklabels([])
#ax1.text(-.3,0.5,'Simulation w/o Demand',fontsize='xx-large',fontweight='bold',rotation=90,ha='center',va='center')
tt1 = '$\mathrm{\mathbf{%s}}_{%s}$' %(site,int(pct_disp1*100))
fig_title(fig,tt1,loc=(-0.02,0.86),fontsize='xx-large',fontweight='bold',rotation=90,ha='center',va='center')

Q_cp_zi1,Q_cp_Ri1,Q_cp_Rel1 = verify.pit_pred(sim_data1[:,sset_idx,5],Q_cp_hefs[sset_idx])
Q_cp_zi2,Q_cp_Ri2,Q_cp_Rel2 = verify.pit_pred(sim_data2[:,sset_idx,5],Q_cp_hefs[sset_idx])

#panel 2: PQQ plot
ax2 = fig.add_subplot(gs0[1])

ax2.scatter(Q_cp_zi1[0],Q_cp_Ri1[0],s=15,color=cv1,edgecolors='gray',linewidths=0.1,alpha=0.5)
ax2.scatter(Q_cp_zi2[0],Q_cp_Ri2[0],s=15,color=cv2,edgecolors='gray',linewidths=0.1,alpha=0.5)
#ax2.legend(['sHEFS-V1','sHEFS-V2'],loc='upper left')
if len(sset_idx) > 1000:
    alp = 0.1 
elif len(sset_idx) <= 1000:
    alp = 0.5
ax2.scatter(Q_cp_zi1,Q_cp_Ri1,s=15,color=cv1,edgecolors='gray',linewidths=0.1,alpha=alp)
ax2.scatter(Q_cp_zi2,Q_cp_Ri2,s=15,color=cv2,edgecolors='gray',linewidths=0.1,alpha=alp)
ax2.axline((0,0),(1,1),linewidth=2,color='gray',linestyle='--')
ax2.text(0.55,0.3,r'$\pi_{rel}: $'+str(round(Q_cp_Rel1,3)),color=cv1,fontsize='x-large',fontweight='bold')
ax2.text(0.55,0.2,r'$\pi_{rel}: $'+str(round(Q_cp_Rel2,3)),color=cv2,fontsize='x-large',fontweight='bold')
#ax2.text(0.6,0.2,r'$\tau: $'+str(round(verify.tau_stat(Q_cp_zi1)[0],3)),color=cv1,fontsize='large',fontweight='bold')
#ax2.text(0.6,0.125,r'$\tau: $'+str(round(verify.tau_stat(Q_cp_zi2)[0],3)),color=cv2,fontsize='large',fontweight='bold')
ax2.set_xlim([0,1])
ax2.set_ylim([0,1])
ax2.set_title('Release PIT plot',fontsize='large')
#ax2.set_xlabel('Obs quantile')
ax2.xaxis.set_ticklabels([])
ax2.yaxis.set_ticklabels([])
ax2.tick_params(axis='both',which='major',labelsize='large')
ax2.set_ylabel('$rank(PIT_t)/N$',fontsize='large')
ax2.text(0.95,0.05,'b)',fontsize='xx-large',fontweight='bold',ha='right')
ax2.tick_params(axis='both',which='major',labelsize='large')
ax2.xaxis.set_ticks([0,0.2,0.4,0.6,0.8,1])

Q_cp_thresh = np.sort(Q_cp_hefs)[int(pct_thresh1*len(Q))]
bns = 5

#sset_idx = np.where(R_hefs > np.sort(R_hefs)[int(pct_disp1*len(R_hefs))])[0]
sset_idx = np.where(R_hefs > 0)[0]

p_obs_y1,p_y1,uc_prob1,ens_uc_prob1,prob_vec1 = verify.rel_plot_calcs(ensemble=sim_data1[:,sset_idx,5], tgt=Q_cp_hefs[sset_idx], bins=bns, threshold=Q_cp_thresh)
#bscore1,bskill1 = verify.brier_score_skill(ensemble=sim_data1[:,sset_idx,5], tgt=Q_cp_hefs[sset_idx], threshold=Q_cp_thresh)
bs_rel1,bs_res1,bs_unc1 = verify.brier_scores(ensemble=sim_data1[:,sset_idx,5], tgt=Q_cp_hefs[sset_idx], bins=bns, threshold=Q_cp_thresh)


p_obs_y2,p_y2,uc_prob2,ens_uc_prob2,prob_vec2 = verify.rel_plot_calcs(ensemble=sim_data2[:,sset_idx,5], tgt=Q_cp_hefs[sset_idx], bins=bns, threshold=Q_cp_thresh)
#bscore2,bskill2 = verify.brier_score_skill(ensemble=sim_data2[:,sset_idx,5], tgt=Q_cp_hefs[sset_idx], threshold=Q_cp_thresh)
bs_rel2,bs_res2,bs_unc2 = verify.brier_scores(ensemble=sim_data2[:,sset_idx,5], tgt=Q_cp_hefs[sset_idx], bins=bns, threshold=Q_cp_thresh)

vert = 0.025
pts_y = np.array([[vert,uc_prob1],[0,uc_prob1 + vert/np.tan(pi/3)],[0,uc_prob1 - vert/np.tan(pi/3)]])
pts_x1 = np.array([[ens_uc_prob1,vert],[ens_uc_prob1 + vert/np.tan(pi/3),0],[ens_uc_prob1 - vert/np.tan(pi/3),0]])
pts_x2 = np.array([[ens_uc_prob2,vert],[ens_uc_prob2 + vert/np.tan(pi/3),0],[ens_uc_prob2 - vert/np.tan(pi/3),0]])

ax3 = fig.add_subplot(gs0[2])
ax3.plot(p_y1,p_obs_y1,color=cv1,linewidth=3)
ax3.plot(p_y2,p_obs_y2,color=cv2,linewidth=3)
#ax3.legend(['sHEFS-V1','sHEFS-V2'],loc='upper left')
ax3.axline((0,0),(1,1),linewidth=2,color='gray',linestyle='--')
ax3.set_title('Release reliability diagram',fontsize='large')
#ax3.set_xlabel('$p(y)$')
ax3.xaxis.set_ticklabels([])
ax3.yaxis.set_ticklabels([])
ax3.set_ylabel('$\overline{o}_i$',fontsize='large')
py = Polygon(pts_y,closed=True,fc='white',ec='black')
ax3.add_patch(py)
px1 = Polygon(pts_x1,closed=True,fc='white',ec=cv1)
px2 = Polygon(pts_x2,closed=True,fc='white',ec=cv2)
ax3.add_patch(px1)
ax3.add_patch(px2)
ax3.set_xlim([0,1])
ax3.set_ylim([0,1])
ax3.text(0.05,0.9,'$R_{90}$: '+str(round(Q_cp_thresh,2))+' TAF',fontsize='x-large',fontweight='bold')
ax3.text(0.05,0.775,'$BS_{rel}$: '+str(round(bs_rel1*1e2,4)),color=cv1,fontsize='x-large',fontweight='bold')
ax3.text(0.05,0.675,'$BS_{rel}$: '+str(round(bs_rel2*1e2,4)),color=cv2,fontsize='x-large',fontweight='bold')
ax3.text(0.95,0.05,'c)',fontsize='xx-large',fontweight='bold',ha='right')
ax3.tick_params(axis='both',which='major',labelsize='large')
ax3.xaxis.set_ticks([0,0.2,0.4,0.6,0.8,1])

sset_idx = np.where(Q > np.sort(Q)[int(pct_disp2*len(Q))])[0]
#sset_idx = np.where(R_hefs > np.sort(R_hefs)[int(pct_disp2*len(R_hefs))])[0]

x = np.zeros(len(pcntiles)+1)
y = np.zeros(len(pcntiles)+1)
x[1:] = pcntiles

Qcp_rel_out1 = np.zeros_like(x)
Qcp_rel_out2 = np.zeros_like(x)
#Qcp_rel_out1d = np.zeros_like(x)
#Qcp_rel_out2d = np.zeros_like(x)

for i in range(len(pcntiles)):
    Qcp_rel_out1[i+1] = np.sum(Q_cp_rel1[i,sset_idx])/len(sset_idx)
    Qcp_rel_out2[i+1] = np.sum(Q_cp_rel2[i,sset_idx])/len(sset_idx)

ax5 = fig.add_subplot(gs0[3])

#ax1.text(-.3,0.5,'Simulation w/ Demand',fontsize='xx-large',fontweight='bold',rotation=90,ha='center',va='center')
tt2 = '$\mathrm{\mathbf{%s}}_{%s}$' %(site,int(pct_disp2*100))
fig_title(fig,tt2,loc=(-0.02,0.63),fontsize='xx-large',fontweight='bold',rotation=90,ha='center',va='center')
#panel 1: Statistical reliability plot
ax5.plot(x,Qcp_rel_out1,color=cv1,linewidth=3)
ax5.plot(x,Qcp_rel_out2,color=cv2,linewidth=3)
#ax5.legend(['sHEFS-V1','sHEFS-V2'],loc='upper left',fontsize='x-large')
ax5.set_xlim([0,1])
ax5.set_ylim([0,1])
ax5.axline((0,0),slope=1,color='gray',alpha=0.5,linestyle='--')
#ax5.set_title('$Q_{cp}$ statistical reliability')
ax5.set_xlabel('Coverage interval (CI)',fontsize='large')
ax5.set_ylabel('Coverage probability (CP)',fontsize='large')
#ax5.xaxis.set_ticklabels([])
#ax5.yaxis.set_ticklabels([])
ax5.text(0.95,0.05,'d)',fontsize='xx-large',fontweight='bold',ha='right')
ax5.tick_params(axis='both',which='major',labelsize='large')
ax5.xaxis.set_ticks([0,0.2,0.4,0.6,0.8,1])

Q_cp_zi1,Q_cp_Ri1,Q_cp_Rel1 = verify.pit_pred(sim_data1[:,sset_idx,5],Q_cp_hefs[sset_idx])
Q_cp_zi2,Q_cp_Ri2,Q_cp_Rel2 = verify.pit_pred(sim_data2[:,sset_idx,5],Q_cp_hefs[sset_idx])

#panel 2: PQQ plot
ax6 = fig.add_subplot(gs0[4])

ax6.scatter(Q_cp_zi1[0],Q_cp_Ri1[0],s=15,color=cv1,edgecolors='gray',linewidths=0.1,alpha=0.5)
ax6.scatter(Q_cp_zi2[0],Q_cp_Ri2[0],s=15,color=cv2,edgecolors='gray',linewidths=0.1,alpha=0.5)
#ax2.legend(['sHEFS-V1','sHEFS-V2'],loc='upper left')
if len(sset_idx) > 1000:
    alp = 0.1 
elif len(sset_idx) <= 1000:
    alp = 0.5
ax6.scatter(Q_cp_zi1,Q_cp_Ri1,s=15,color=cv1,edgecolors='gray',linewidths=0.1,alpha=alp)
ax6.scatter(Q_cp_zi2,Q_cp_Ri2,s=15,color=cv2,edgecolors='gray',linewidths=0.1,alpha=alp)
ax6.axline((0,0),(1,1),linewidth=2,color='gray',linestyle='--')
ax6.text(0.55,0.3,r'$\pi_{rel}: $'+str(round(Q_cp_Rel1,3)),color=cv1,fontsize='x-large',fontweight='bold')
ax6.text(0.55,0.2,r'$\pi_{rel}: $'+str(round(Q_cp_Rel2,3)),color=cv2,fontsize='x-large',fontweight='bold')
#ax6.text(0.6,0.3,r'$\tau: $'+str(round(verify.tau_stat(Q_cp_zi1)[0],3)),color=cv1,fontsize='large',fontweight='bold')
#ax6.text(0.6,0.225,r'$\tau: $'+str(round(verify.tau_stat(Q_cp_zi2)[0],3)),color=cv2,fontsize='large',fontweight='bold')
ax6.set_xlim([0,1])
ax6.set_ylim([0,1])
#ax6.set_title('$Q_{cp}$ PQQ plot')
ax6.set_xlabel('$PIT_t$',fontsize='large')
ax6.set_ylabel('$rank(PIT_t)/N$',fontsize='large')
ax6.text(0.95,0.05,'e)',fontsize='xx-large',fontweight='bold',ha='right')
#ax6.xaxis.set_ticklabels([])
ax6.yaxis.set_ticklabels([])
ax6.tick_params(axis='both',which='major',labelsize='large')
ax6.xaxis.set_ticks([0,0.2,0.4,0.6,0.8,1])

Q_cp_thresh = np.sort(Q_cp_hefs)[int(pct_thresh2*len(Q))]
bns = 5
sset_idx = np.where(R_hefs > 0)[0]

p_obs_y1,p_y1,uc_prob1,ens_uc_prob1,prob_vec1 = verify.rel_plot_calcs(ensemble=sim_data1[:,sset_idx,5], tgt=Q_cp_hefs[sset_idx], bins=bns, threshold=Q_cp_thresh)
#bscore1,bskill1 = verify.brier_score_skill(ensemble=sim_data1[:,sset_idx,5], tgt=Q_cp_hefs[sset_idx], threshold=Q_cp_thresh)
bs_rel1,bs_res1,bs_unc1 = verify.brier_scores(ensemble=sim_data1[:,sset_idx,5], tgt=Q_cp_hefs[sset_idx], bins=bns, threshold=Q_cp_thresh)

p_obs_y2,p_y2,uc_prob2,ens_uc_prob2,prob_vec2 = verify.rel_plot_calcs(ensemble=sim_data2[:,sset_idx,5], tgt=Q_cp_hefs[sset_idx], bins=bns, threshold=Q_cp_thresh)
#bscore2,bskill2 = verify.brier_score_skill(ensemble=sim_data2[:,sset_idx,5], tgt=Q_cp_hefs[sset_idx], threshold=Q_cp_thresh)
bs_rel2,bs_res2,bs_unc2 = verify.brier_scores(ensemble=sim_data2[:,sset_idx,5], tgt=Q_cp_hefs[sset_idx], bins=bns, threshold=Q_cp_thresh)

vert = 0.025
pts_y = np.array([[vert,uc_prob1],[0,uc_prob1 + vert/np.tan(pi/3)],[0,uc_prob1 - vert/np.tan(pi/3)]])
pts_x1 = np.array([[ens_uc_prob1,vert],[ens_uc_prob1 + vert/np.tan(pi/3),0],[ens_uc_prob1 - vert/np.tan(pi/3),0]])
pts_x2 = np.array([[ens_uc_prob2,vert],[ens_uc_prob2 + vert/np.tan(pi/3),0],[ens_uc_prob2 - vert/np.tan(pi/3),0]])

ax7 = fig.add_subplot(gs0[5])
ax7.plot(p_y1,p_obs_y1,color=cv1,linewidth=3)
ax7.plot(p_y2,p_obs_y2,color=cv2,linewidth=3)
#ax3.legend(['sHEFS-V1','sHEFS-V2'],loc='upper left')
ax7.axline((0,0),(1,1),linewidth=2,color='gray',linestyle='--')
#ax7.set_title('$Q_{cp}$ reliability diagram')
ax7.set_xlabel('$y_i$',fontsize='large')
ax7.set_ylabel('$\overline{o}_i$',fontsize='large')
py = Polygon(pts_y,closed=True,fc='white',ec='black')
ax7.add_patch(py)
px1 = Polygon(pts_x1,closed=True,fc='white',ec=cv1)
px2 = Polygon(pts_x2,closed=True,fc='white',ec=cv2)
ax7.add_patch(px1)
ax7.add_patch(px2)
ax7.set_xlim([0,1])
ax7.set_ylim([0,1])
ax7.text(0.05,0.9,'$R_{99}$: '+str(round(Q_cp_thresh,2))+' TAF',fontsize='x-large',fontweight='bold')
ax7.text(0.05,0.775,'$BS_{rel}$: '+str(round(bs_rel1*1e2,4)),color=cv1,fontsize='x-large',fontweight='bold')
ax7.text(0.05,0.675,'$BS_{rel}$: '+str(round(bs_rel2*1e2,4)),color=cv2,fontsize='x-large',fontweight='bold')
ax7.text(0.95,0.05,'f)',fontsize='xx-large',fontweight='bold',ha='right')
ax7.yaxis.set_ticklabels([])
ax7.tick_params(axis='both',which='major',labelsize='large')
ax7.xaxis.set_ticks([0,0.2,0.4,0.6,0.8,1])


sset_idx = np.where(Q > np.sort(Q)[int(pct_disp1*len(Q))])[0]
#sset_idx = np.where(R_hefs > np.sort(R_hefs)[int(pct_disp1*len(R_hefs))])[0]

x = np.zeros(len(pcntiles)+1)
y = np.zeros(len(pcntiles)+1)
x[1:] = pcntiles

S_rel_out1 = np.zeros_like(x)
S_rel_out2 = np.zeros_like(x)
#Qcp_rel_out1d = np.zeros_like(x)
#Qcp_rel_out2d = np.zeros_like(x)

for i in range(len(pcntiles)):
    S_rel_out1[i+1] = np.sum(S_rel1[i,sset_idx])/len(sset_idx)
    S_rel_out2[i+1] = np.sum(S_rel2[i,sset_idx])/len(sset_idx)
    #Qcp_rel_out1d[i+1] = np.sum(Q_cp_rel1d[i,sset_idx])/len(sset_idx)
    #Qcp_rel_out2d[i+1] = np.sum(Q_cp_rel2d[i,sset_idx])/len(sset_idx)
    
ax8 = fig.add_subplot(gs0[6])

#plt.title('Simulation w/o Demand',rotation=90,loc='center',y=0.5,fontsize='xx-large',fontweight='bold')
#panel 1: Statistical reliability plot
ax8.plot(x,S_rel_out1,color=cv1,linewidth=3)
ax8.plot(x,S_rel_out2,color=cv2,linewidth=3)
#ax8.legend(['sHEFS-V1','sHEFS-V2'],loc='upper left',fontsize='large')
ax8.set_xlim([0,1])
ax8.set_ylim([0,1])
ax8.axline((0,0),slope=1,color='gray',alpha=0.5,linestyle='--')
ax8.set_title('Storage CP plot',fontsize='large')

#ax8.set_xlabel('Coverage interval')
ax8.set_ylabel('Coverage probability (CP)',fontsize='large')
ax8.text(0.95,0.05,'g)',fontsize='xx-large',fontweight='bold',ha='right')
ax8.xaxis.set_ticklabels([])
ax8.tick_params(axis='both',which='major',labelsize='large')
ax8.xaxis.set_ticks([0,0.2,0.4,0.6,0.8,1])

#ax1.text(-.3,0.5,'Simulation w/o Demand',fontsize='xx-large',fontweight='bold',rotation=90,ha='center',va='center')
fig_title(fig,tt1,loc=(-0.02,0.36),fontsize='xx-large',fontweight='bold',rotation=90,ha='center',va='center')

S_zi1,S_Ri1,S_Rel1 = verify.pit_pred(sim_data1[:,sset_idx,0],S_hefs[sset_idx])
S_zi2,S_Ri2,S_Rel2 = verify.pit_pred(sim_data2[:,sset_idx,0],S_hefs[sset_idx])

#panel 2: PQQ plot
ax9 = fig.add_subplot(gs0[7])

ax9.scatter(S_zi1[0],S_Ri1[0],s=15,color=cv1,edgecolors='gray',linewidths=0.1,alpha=0.5)
ax9.scatter(S_zi2[0],S_Ri2[0],s=15,color=cv2,edgecolors='gray',linewidths=0.1,alpha=0.5)
#ax2.legend(['sHEFS-V1','sHEFS-V2'],loc='upper left')
if len(sset_idx) > 1000:
    alp = 0.1 
elif len(sset_idx) <= 1000:
    alp = 0.5
ax9.scatter(S_zi1,S_Ri1,s=15,color=cv1,edgecolors='gray',linewidths=0.1,alpha=alp)
ax9.scatter(S_zi2,S_Ri2,s=15,color=cv2,edgecolors='gray',linewidths=0.1,alpha=alp)
ax9.axline((0,0),(1,1),linewidth=2,color='gray',linestyle='--')
ax9.text(0.55,0.3,r'$\pi_{rel}: $'+str(round(S_Rel1,3)),color=cv1,fontsize='x-large',fontweight='bold')
ax9.text(0.55,0.2,r'$\pi_{rel}: $'+str(round(S_Rel2,3)),color=cv2,fontsize='x-large',fontweight='bold')
#ax9.text(0.6,0.2,r'$\tau: $'+str(round(verify.tau_stat(S_zi1)[0],3)),color=cv1,fontsize='large',fontweight='bold')
#ax9.text(0.6,0.125,r'$\tau: $'+str(round(verify.tau_stat(S_zi2)[0],3)),color=cv2,fontsize='large',fontweight='bold')
ax9.set_xlim([0,1])
ax9.set_ylim([0,1])
ax9.set_title('Storage PIT plot',fontsize='large')
#ax9.set_xlabel('Obs quantile')
ax9.xaxis.set_ticklabels([])
ax9.yaxis.set_ticklabels([])
ax9.set_ylabel('$rank(PIT_t)/N$',fontsize='large')
ax9.text(0.95,0.05,'h)',fontsize='xx-large',fontweight='bold',ha='right')
ax9.tick_params(axis='both',which='major',labelsize='large')
ax9.xaxis.set_ticks([0,0.2,0.4,0.6,0.8,1])

sset_idx = np.where(Q > np.sort(Q)[int(pct_disp2*len(Q))])[0]
#sset_idx = np.where(R_hefs > np.sort(R_hefs)[int(pct_disp2*len(R_hefs))])[0]

x = np.zeros(len(pcntiles)+1)
y = np.zeros(len(pcntiles)+1)
x[1:] = pcntiles

S_rel_out1 = np.zeros_like(x)
S_rel_out2 = np.zeros_like(x)
#Qcp_rel_out1d = np.zeros_like(x)
#Qcp_rel_out2d = np.zeros_like(x)

for i in range(len(pcntiles)):
    S_rel_out1[i+1] = np.sum(S_rel1[i,sset_idx])/len(sset_idx)
    S_rel_out2[i+1] = np.sum(S_rel2[i,sset_idx])/len(sset_idx)
    #Qcp_rel_out1d[i+1] = np.sum(Q_cp_rel1d[i,sset_idx])/len(sset_idx)
    #Qcp_rel_out2d[i+1] = np.sum(Q_cp_rel2d[i,sset_idx])/len(sset_idx)
    
ax10 = fig.add_subplot(gs0[9])

#plt.title('Simulation w/o Demand',rotation=90,loc='center',y=0.5,fontsize='xx-large',fontweight='bold')
#panel 1: Statistical reliability plot
ax10.plot(x,S_rel_out1,color=cv1,linewidth=3)
ax10.plot(x,S_rel_out2,color=cv2,linewidth=3)
#ax10.legend(['sHEFS-V1','sHEFS-V2'],loc='upper left',fontsize='large')
ax10.set_xlim([0,1])
ax10.set_ylim([0,1])
ax10.axline((0,0),slope=1,color='gray',alpha=0.5,linestyle='--')
#ax8.set_title('$S$ statistical reliability')

ax10.set_xlabel('Coverage interval (CI)',fontsize='large')
ax10.set_ylabel('Coverage probability (CP)',fontsize='large')
ax10.text(0.95,0.05,'i)',fontsize='xx-large',fontweight='bold',ha='right')
ax10.tick_params(axis='both',which='major',labelsize='large')
ax10.xaxis.set_ticks([0,0.2,0.4,0.6,0.8,1])

#ax10.xaxis.set_ticklabels([])
#ax1.text(-.3,0.5,'Simulation w/o Demand',fontsize='xx-large',fontweight='bold',rotation=90,ha='center',va='center')
fig_title(fig,tt2,loc=(-0.02,0.14),fontsize='xx-large',fontweight='bold',rotation=90,ha='center',va='center')

S_zi1,S_Ri1,S_Rel1 = verify.pit_pred(sim_data1[:,sset_idx,0],S_hefs[sset_idx])
S_zi2,S_Ri2,S_Rel2 = verify.pit_pred(sim_data2[:,sset_idx,0],S_hefs[sset_idx])

#panel 2: PQQ plot
ax11 = fig.add_subplot(gs0[10])

ax11.scatter(S_zi1[0],S_Ri1[0],s=15,color=cv1,edgecolors='gray',linewidths=0.1,alpha=0.5)
ax11.scatter(S_zi2[0],S_Ri2[0],s=15,color=cv2,edgecolors='gray',linewidths=0.1,alpha=0.5)
#ax2.legend(['sHEFS-V1','sHEFS-V2'],loc='upper left')
if len(sset_idx) > 1000:
    alp = 0.1 
elif len(sset_idx) <= 1000:
    alp = 0.5
ax11.scatter(S_zi1,S_Ri1,s=15,color=cv1,edgecolors='gray',linewidths=0.1,alpha=alp)
ax11.scatter(S_zi2,S_Ri2,s=15,color=cv2,edgecolors='gray',linewidths=0.1,alpha=alp)
ax11.axline((0,0),(1,1),linewidth=2,color='gray',linestyle='--')
ax11.text(0.55,0.3,r'$\pi_{rel}: $'+str(round(S_Rel1,3)),color=cv1,fontsize='x-large',fontweight='bold')
ax11.text(0.55,0.2,r'$\pi_{rel}: $'+str(round(S_Rel2,3)),color=cv2,fontsize='x-large',fontweight='bold')
#ax11.text(0.6,0.2,r'$\tau: $'+str(round(verify.tau_stat(S_zi1)[0],3)),color=cv1,fontsize='large',fontweight='bold')
#ax11.text(0.6,0.125,r'$\tau: $'+str(round(verify.tau_stat(S_zi2)[0],3)),color=cv2,fontsize='large',fontweight='bold')
ax11.set_xlim([0,1])
ax11.set_ylim([0,1])
#ax9.set_title('$S$ PQQ plot')
ax11.set_xlabel('$PIT_t$',fontsize='large')
#ax11.xaxis.set_ticklabels([])
ax11.yaxis.set_ticklabels([])
ax11.set_ylabel('$rank(PIT_t)/N$',fontsize='large')
ax11.text(0.95,0.05,'j)',fontsize='xx-large',fontweight='bold',ha='right')
ax11.tick_params(axis='both',which='major',labelsize='large')
ax11.xaxis.set_ticks([0,0.2,0.4,0.6,0.8,1])


plt.savefig('./figures/%s/%s/4x3_stats-rel_pqq_rel-diagram_R-S_clean_svers-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s_disp-pct=%s,%s_thresh-pct=%s,%s.png' %(loc,site,svers,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed,pct_disp1,pct_disp2,pct_thresh1,pct_thresh2),dpi=300,bbox_inches='tight')


#sys.modules[__name__].__dict__.clear()


#----------------------------------end-----------------------------
