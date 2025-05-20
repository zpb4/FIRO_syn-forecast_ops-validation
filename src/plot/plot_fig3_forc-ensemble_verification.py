# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:22:54 2024

@author: zpb4
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import syn_util
import ensemble_verification_functions as verify
from time import localtime, strftime
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import seaborn as sns

col_cb = sns.color_palette('colorblind')
#sns.palplot(col_cb)  #plot coloblind palette for comparison
colv1 = sns.color_palette('PuRd',10)
colv2 = sns.color_palette('YlOrBr',10)

cv2 = colv1[4]  #version 1 is pink-purplish color
cv1 = colv2[6]  #version 2 is orangey-brown color
chefs = col_cb[0]  #hefs is colorblind blue

# to simulate a policy, replace 'firo_pool' and 'risk_thresholds' with those found in the train.py file
kcfs_to_tafd = 2.29568411*10**-5 * 86400

loc = 'YRS'
site = 'NBBC1'
vers_out1 = '5fold'
vers_out2 = '5fold'

#extracted date index for syn-forecasts
sd_syn = '1985-10-15' 
ed_syn = '2019-08-15'

idx_syn = pd.date_range(start = sd_syn, end = ed_syn )

#date range of consideration (inclusive of all sites and HEFS avail dates)
sd = '1990-10-01' 
ed = '2019-08-15'
sl_idx = idx_syn.slice_indexer(sd,ed)
save_figs = True  # T to show plots, F to save .png files to ./plot repo

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

nsamps = 100
ld = 1
ld2 = 3
lds = (1,3,5,10)
pcnt_rnk = (0.99,1)
pcnt_bse = (0,1)
low_pcnt_ecrps = (0,0.99)
upr_pcnt_ecrps = (0.99,1)

Q_hefs_trn,Qf_hefs_inp,dowy_hefs_trn,tocs,df_idx_hefs = syn_util.extract(sd,ed,forecast_type='hefs',syn_sample='',Rsyn_path='../Synthetic-Forecast-v1-FIRO-DISES',syn_vers='',forecast_param='',loc=loc,site=site,opt_pcnt='',obj_pwr='',opt_strat='',gen_setup='')
Qf_hefs_trn = verify.onesamp_forecast_rearrange(Qf_hefs_inp)

wy_vec = df_idx_hefs.year.values
wy_vec[np.isin(df_idx_hefs.month,[10,11,12])] = wy_vec[np.isin(df_idx_hefs.month,[10,11,12])]+1

val_idx = np.arange(len(Q_hefs_trn))  #val index to use for entire dataset

Qf_v1_inp = np.load('data/%s/%s/%s-%s_Qf_syn-forecast%s_nsamp=%s.npz' %(loc,site,loc,site,syn_vers1+vers_out1,nsamps))['arr']
Qf_v1_trn = verify.multisamp_forecast_rearrange(Qf_v1_inp[:,sl_idx,:,:])
Qf_v1 = Qf_v1_trn[:,val_idx,:,:]

del Qf_v1_inp,Qf_v1_trn

Qf_v2_inp = np.load('data/%s/%s/%s-%s_Qf_syn-forecast%s_nsamp=%s.npz' %(loc,site,loc,site,syn_vers2+vers_out2,nsamps))['arr']
Qf_v2_trn = verify.multisamp_forecast_rearrange(Qf_v2_inp[:,sl_idx,:,:])
Qf_v2 = Qf_v2_trn[:,val_idx,:,:]

del Qf_v2_inp,Qf_v2_trn

#1. plot rank histograms for various leads and percentiles
Qf_hefs = Qf_hefs_trn[val_idx,:,:]
Q_hefs = Q_hefs_trn[val_idx]
dowy_hefs = dowy_hefs_trn[val_idx]
fsort = False
#sset_idx = np.where((dowy_hefs>60) & (dowy_hefs<170))[0]
sset_idx = np.arange(len(Q_hefs_trn))
ne = np.shape(Qf_hefs)[1]

Qf_v1_rhist = Qf_v1[:,:,:,ld-1]
Qf_v2_rhist = Qf_v2[:,:,:,ld-1]
rnk_hist_hefs = verify.onesamp_rnk_hist(Qf_hefs[sset_idx,:,ld-1], Q_hefs[sset_idx], pcnt_rnk, forc_sort=fsort)
rnk_hist_v1 = verify.multisamp_rnk_hist(Qf_v1_rhist[:,sset_idx,:], Q_hefs[sset_idx],pcnt_rnk,forc_sort=fsort)
rnk_hist_v1_agg = rnk_hist_v1[0].sum(axis=0) / np.sum(rnk_hist_v1[0])
rnk_hist_v2 = verify.multisamp_rnk_hist(Qf_v2_rhist[:,sset_idx,:], Q_hefs[sset_idx],pcnt_rnk,forc_sort=fsort)
rnk_hist_v2_agg = rnk_hist_v2[0].sum(axis=0) / np.sum(rnk_hist_v2[0])


Qf_v1_rhist_l2 = Qf_v1[:,:,:,ld2-1]
Qf_v2_rhist_l2 = Qf_v2[:,:,:,ld2-1]
rnk_hist_hefs_l2 = verify.onesamp_rnk_hist(Qf_hefs[sset_idx,:,ld2-1], Q_hefs[sset_idx], pcnt_rnk, forc_sort=fsort)
rnk_hist_v1_l2 = verify.multisamp_rnk_hist(Qf_v1_rhist_l2[:,sset_idx,:], Q_hefs[sset_idx],pcnt_rnk,forc_sort=fsort)
rnk_hist_v1_agg_l2 = rnk_hist_v1_l2[0].sum(axis=0) / np.sum(rnk_hist_v1_l2[0])
rnk_hist_v2_l2 = verify.multisamp_rnk_hist(Qf_v2_rhist_l2[:,sset_idx,:], Q_hefs[sset_idx],pcnt_rnk,forc_sort=fsort)
rnk_hist_v2_agg_l2 = rnk_hist_v2_l2[0].sum(axis=0) / np.sum(rnk_hist_v2_l2[0])


#2. plot cumulative rank histograms for various leads and percentiles
crnk_hist_hefs = np.zeros(ne+2)
crnk_hist_hefs[1:] = rnk_hist_hefs[1]

crnk_hist_hefs_l2 = np.zeros(ne+2)
crnk_hist_hefs_l2[1:] = rnk_hist_hefs_l2[1]

sns.set_theme()
sns.set_style('ticks')
sns.set_context('paper')

fig = plt.figure(layout='constrained',figsize=(10,10))
gs0 = fig.add_gridspec(2,2)
gs1 = gs0[0].subgridspec(2,2)
gs2 = gs0[1].subgridspec(2,2)
ne = np.shape(Qf_hefs)[1]
x = np.arange(ne+2)
unif_dens = rnk_hist_hefs[3]

#plt.subplot(4,4,1)
ax1 = fig.add_subplot(gs1[0])
ax1.plot(x,crnk_hist_hefs,color=chefs,linewidth=2)
#plt.xlabel('Ensemble Rank')
ax1.set_ylabel('Cumulative Density',fontsize='large')
ax1.set_title('Cumulative Rank Histogram',fontsize='large')
for i in range(nsamps):
    crnk_hist = np.zeros(ne+2)
    crnk_hist[1:] = rnk_hist_v1[1][i,:]
    ax1.plot(x,crnk_hist,color=cv1,linewidth=1.5,alpha=0.2)
ax1.legend(['HEFS','syn-M1'],loc='upper left',frameon=False,fontsize='large')
leg = ax1.get_legend()
leg.legend_handles[1].set_alpha(1)
ax1.axline((0,0),slope=1/(ne+1),color='gray',linestyle='--')
ax1.plot(x,crnk_hist_hefs,color=chefs,linewidth=2)
ax1.set_xlim([0,ne+1])
ax1.set_ylim([0,1])
#ax1.text(.05,.65,str(pcnt_rnk[0])+'-'+str(pcnt_rnk[1])+' percentile',transform=ax1.transAxes,ha='left',fontsize='large')
#ax1.text(.05,.65,'top '+str(int(round((1-pcnt_rnk[0])*100,0)))+'% flows',transform=ax1.transAxes,ha='left',fontsize='large')
if pcnt_rnk[0]==0 and pcnt_rnk[1]==1:
    ax1.text(.05,.65,'all flows',transform=ax1.transAxes,ha='left',fontsize='large')
else:
    ax1.text(.05,.65,'top '+str(int(round((1-pcnt_rnk[0])*100,0)))+'% flows',transform=ax1.transAxes,ha='left',fontsize='large')
ax1.text(.05,0.55,'Lead ' + str(ld),transform=ax1.transAxes,ha='left',fontsize='large')
ax1.text(.95,0.05,'a)',transform=ax1.transAxes,ha='right',va='bottom',fontsize='xx-large',fontweight='bold')
ax1.xaxis.set_ticklabels([])


#plt.subplot(4,4,2)
ax2 = fig.add_subplot(gs1[1])
ax2.plot(x,crnk_hist_hefs,color=chefs,linewidth=2)
#plt.xlabel('Ensemble Rank')
#plt.ylabel('Cumulative Density')
ax2.yaxis.set_ticklabels([])
ax2.xaxis.set_ticklabels([])
ax2.set_title('Cumulative Rank Histogram',fontsize='large')
for i in range(nsamps):
    crnk_hist = np.zeros(ne+2)
    crnk_hist[1:] = rnk_hist_v2[1][i,:]
    ax2.plot(x,crnk_hist,color=cv2,linewidth=1.5,alpha=0.2)
ax2.legend(['HEFS','syn-M2'],loc='upper left',frameon=False,fontsize='large')
leg = ax2.get_legend()
leg.legend_handles[1].set_alpha(1)
ax2.axline((0,0),slope=1/(ne+1),color='gray',linestyle='--')
ax2.plot(x,crnk_hist_hefs,color=chefs,linewidth=2)
ax2.set_xlim([0,ne+1])
ax2.set_ylim([0,1])
#ax2.text(.05,0.65,str(pcnt_rnk[0])+'-'+str(pcnt_rnk[1])+' percentile',transform=ax2.transAxes,ha='left',fontsize='large')
#ax2.text(.05,.65,'top '+str(int(round((1-pcnt_rnk[0])*100,0)))+'% flows',transform=ax1.transAxes,ha='left',fontsize='large')
if pcnt_rnk[0]==0 and pcnt_rnk[1]==1:
    ax2.text(.05,.65,'all flows',transform=ax2.transAxes,ha='left',fontsize='large')
else:
    ax2.text(.05,.65,'top '+str(int(round((1-pcnt_rnk[0])*100,0)))+'% flows',transform=ax2.transAxes,ha='left',fontsize='large')
ax2.text(.05,0.55,'Lead ' + str(ld),transform=ax2.transAxes,ha='left',fontsize='large')
ax2.text(.95,0.05,'b)',transform=ax2.transAxes,ha='right',va='bottom',fontsize='xx-large',fontweight='bold')

#plt.subplot(4,4,5)
ax3 = fig.add_subplot(gs1[2])
ax3.plot(x,crnk_hist_hefs_l2,color=chefs,linewidth=2)
ax3.set_xlabel('Ensemble Rank',fontsize='large')
ax3.set_ylabel('Cumulative Density',fontsize='large')
#plt.title('Cumulative Rank Histogram: Lead %s' %(ld))
for i in range(nsamps):
    crnk_hist = np.zeros(ne+2)
    crnk_hist[1:] = rnk_hist_v1_l2[1][i,:]
    ax3.plot(x,crnk_hist,color=cv1,linewidth=1.5,alpha=0.2)
#ax3.legend(['HEFS','sHEFS-V1'])
ax3.axline((0,0),slope=1/(ne+1),color='gray',linestyle='--')
ax3.plot(x,crnk_hist_hefs_l2,color=chefs,linewidth=2)
ax3.set_xlim([0,ne+1])
ax3.set_ylim([0,1])
#ax3.text(.05,.9,str(pcnt_rnk[0])+'-'+str(pcnt_rnk[1])+' percentile',transform=ax3.transAxes,ha='left',fontsize='large')
#ax3.text(.05,.9,'top '+str(int(round((1-pcnt_rnk[0])*100,0)))+'% flows',transform=ax1.transAxes,ha='left',fontsize='large')
if pcnt_rnk[0]==0 and pcnt_rnk[1]==1:
    ax3.text(.05,.9,'all flows',transform=ax3.transAxes,ha='left',fontsize='large')
else:
    ax3.text(.05,.9,'top '+str(int(round((1-pcnt_rnk[0])*100,0)))+'% flows',transform=ax3.transAxes,ha='left',fontsize='large')
ax3.text(.05,0.8,'Lead ' + str(ld2),transform=ax3.transAxes,ha='left',fontsize='large')
ax3.text(.95,0.05,'c)',transform=ax3.transAxes,ha='right',va='bottom',fontsize='xx-large',fontweight='bold')


#plt.subplot(4,4,6)
ax4 = fig.add_subplot(gs1[3])
ax4.plot(x,crnk_hist_hefs_l2,color=chefs,linewidth=2)
ax4.set_xlabel('Ensemble Rank',fontsize='large')
#plt.ylabel('Cumulative Density')
ax4.yaxis.set_ticklabels([])
#plt.title('Cumulative Rank Histogram: Lead %s' %(ld))
for i in range(nsamps):
    crnk_hist = np.zeros(ne+2)
    crnk_hist[1:] = rnk_hist_v2_l2[1][i,:]
    ax4.plot(x,crnk_hist,color=cv2,linewidth=1.5,alpha=0.2)
#ax4.legend(['HEFS','sHEFS-V2'])
ax4.axline((0,0),slope=1/(ne+1),color='gray',linestyle='--')
ax4.plot(x,crnk_hist_hefs_l2,color=chefs,linewidth=2)
ax4.set_xlim([0,ne+1])
ax4.set_ylim([0,1])
#ax4.text(.05,0.9,str(pcnt_rnk[0])+'-'+str(pcnt_rnk[1])+' percentile',transform=ax4.transAxes,ha='left',fontsize='large')
#ax4.text(.05,.9,'top '+str(int(round((1-pcnt_rnk[0])*100,0)))+'% flows',transform=ax4.transAxes,ha='left',fontsize='large')
if pcnt_rnk[0]==0 and pcnt_rnk[1]==1:
    ax4.text(.05,.9,'all flows',transform=ax4.transAxes,ha='left',fontsize='large')
else:
    ax4.text(.05,.9,'top '+str(int(round((1-pcnt_rnk[0])*100,0)))+'% flows',transform=ax4.transAxes,ha='left',fontsize='large')
ax4.text(.05,0.8,'Lead ' + str(ld2),transform=ax4.transAxes,ha='left',fontsize='large')
ax4.text(.95,0.05,'d)',transform=ax4.transAxes,ha='right',va='bottom',fontsize='xx-large',fontweight='bold')

#plt.savefig('./plot/ensemble/1x2_cumul-rank-histogram_ld%s_pcnt=%s-%s.png' %(ld,pcnt[0],pcnt[1]),dpi=300,bbox_inches='tight')
#plt.show()



#3. plot binned spread error diagrams for various leads and percentiles

Qf_v1_bse = Qf_v1[:,:,:,ld-1]
Qf_v2_bse = Qf_v2[:,:,:,ld-1]
bse_hefs = verify.onesamp_bse(Qf_hefs[sset_idx,:,ld-1], Q_hefs[sset_idx], bins=20, pcntile=pcnt_bse)
bse_v1 = verify.multisamp_bse(Qf_v1_bse[:,sset_idx,:], Q_hefs[sset_idx], bins=20, pcntile=pcnt_bse)
bse_v2 = verify.multisamp_bse(Qf_v2_bse[:,sset_idx,:], Q_hefs[sset_idx], bins=20, pcntile=pcnt_bse)

Qf_v1_bse_l2 = Qf_v1[:,:,:,ld2-1]
Qf_v2_bse_l2 = Qf_v2[:,:,:,ld2-1]
bse_hefs_l2 = verify.onesamp_bse(Qf_hefs[sset_idx,:,ld2-1], Q_hefs[sset_idx], bins=20, pcntile=pcnt_bse)
bse_v1_l2 = verify.multisamp_bse(Qf_v1_bse_l2[:,sset_idx,:], Q_hefs[sset_idx], bins=20, pcntile=pcnt_bse)
bse_v2_l2 = verify.multisamp_bse(Qf_v2_bse_l2[:,sset_idx,:], Q_hefs[sset_idx], bins=20, pcntile=pcnt_bse)

#axlim = max((np.max(bse_hefs),np.max(bse_v1),np.max(bse_v2),np.max(bse_hefs_l2),np.max(bse_v1_l2),np.max(bse_v2_l2)))
#axlim = max((np.max(bse_hefs),np.max(bse_v1),np.max(bse_hefs_l2),np.max(bse_v1_l2)))#,np.max(bse_v2_l2)))
axlim = max((np.max(bse_hefs),np.max(bse_hefs_l2)))*1.5

#plt.subplot(4,4,3)
ax1 = fig.add_subplot(gs2[0])
ax1.plot(bse_hefs[0,:],bse_hefs[1,:],color=chefs,linewidth=2)
#plt.xlabel('Spread ($\sqrt{\sigma^2}$)')
ax1.set_ylabel('Error ($rmse$)',fontsize='large')
ax1.set_title('BSE Diagram',fontsize='large')
for i in range(nsamps):
    ax1.plot(bse_v1[i][0,:],bse_v1[i][1,:],color=cv1,linewidth=1.5,alpha=0.2)
#ax1.legend(['HEFS','sHEFS-V1'])
ax1.axline((0,0),slope=1,color='gray',linestyle='--')
ax1.plot(bse_hefs[0,:],bse_hefs[1,:],color=chefs,linewidth=2)
ax1.set_xlim([0,axlim*1.05])
ax1.set_ylim([0,axlim*1.05])
#ax1.text(.05,.9,str(pcnt_bse[0])+'-'+str(pcnt_bse[1])+' percentile',transform=ax1.transAxes,ha='left',fontsize='large')
if pcnt_bse[0]==0 and pcnt_bse[1]==1:
    ax1.text(.05,.9,'all flows',transform=ax1.transAxes,ha='left',fontsize='large')
else:
    ax1.text(.05,.9,'top '+str(int(round((1-pcnt_bse[0])*100,0)))+'% flows',transform=ax1.transAxes,ha='left',fontsize='large')
ax1.text(.05,.8,'Lead ' + str(ld),transform=ax1.transAxes,ha='left',fontsize='large')
ax1.text(.95,0.05,'e)',transform=ax1.transAxes,ha='right',va='bottom',fontsize='xx-large',fontweight='bold')
ax1.yaxis.get_major_locator().set_params(integer=True)
ax1.xaxis.set_ticklabels([])


#plt.subplot(4,4,4)
ax2 = fig.add_subplot(gs2[1])
ax2.plot(bse_hefs[0,:],bse_hefs[1,:],color=chefs,linewidth=2)
#plt.xlabel('Spread ($\sqrt{\sigma^2}$)')
#plt.ylabel('Error ($\sqrt{mse}$)')
ax2.yaxis.set_ticklabels([])
ax2.xaxis.set_ticklabels([])
ax2.set_title('BSE Diagram',fontsize='large')
for i in range(nsamps):
    ax2.plot(bse_v2[i][0,:],bse_v2[i][1,:],color=cv2,linewidth=1.5,alpha=0.2)
#ax2.legend(['HEFS','sHEFS-V1'])
ax2.axline((0,0),slope=1,color='gray',linestyle='--')
ax2.plot(bse_hefs[0,:],bse_hefs[1,:],color=chefs,linewidth=2)
ax2.set_xlim([0,axlim*1.05])
ax2.set_ylim([0,axlim*1.05])
#ax2.text(.05,.9,str(pcnt_bse[0])+'-'+str(pcnt_bse[1])+' percentile',transform=ax2.transAxes,ha='left',fontsize='large')
if pcnt_bse[0]==0 and pcnt_bse[1]==1:
    ax2.text(.05,.9,'all flows',transform=ax2.transAxes,ha='left',fontsize='large')
else:
    ax2.text(.05,.9,'top '+str(int(round((1-pcnt_bse[0])*100,0)))+'% flows',transform=ax2.transAxes,ha='left',fontsize='large')
ax2.text(.05,.8,'Lead ' + str(ld),transform=ax2.transAxes,ha='left',fontsize='large')
ax2.text(.95,0.05,'f)',transform=ax2.transAxes,ha='right',va='bottom',fontsize='xx-large',fontweight='bold')
ax2.yaxis.get_major_locator().set_params(integer=True)

#plt.subplot(4,4,7)
ax3 = fig.add_subplot(gs2[2])
ax3.plot(bse_hefs_l2[0,:],bse_hefs_l2[1,:],color=chefs,linewidth=2)
ax3.set_xlabel('Spread ($\sigma$)',fontsize='large')
ax3.set_ylabel('Error ($rmse$)',fontsize='large')
#plt.title('BSE Diagram')
for i in range(nsamps):
    ax3.plot(bse_v1_l2[i][0,:],bse_v1_l2[i][1,:],color=cv1,linewidth=1.5,alpha=0.2)
#ax3.legend(['HEFS','sHEFS-V1'])
ax3.axline((0,0),slope=1,color='gray',linestyle='--')
ax3.plot(bse_hefs_l2[0,:],bse_hefs_l2[1,:],color=chefs,linewidth=2)
ax3.set_xlim([0,axlim*1.05])
ax3.set_ylim([0,axlim*1.05])
#ax3.text(0.05,0.9,str(pcnt_bse[0])+'-'+str(pcnt_bse[1])+' percentile',transform=ax3.transAxes,ha='left',fontsize='large')
if pcnt_bse[0]==0 and pcnt_bse[1]==1:
    ax3.text(.05,.9,'all flows',transform=ax3.transAxes,ha='left',fontsize='large')
else:
    ax3.text(.05,.9,'top '+str(int(round((1-pcnt_bse[0])*100,0)))+'% flows',transform=ax3.transAxes,ha='left',fontsize='large')
ax3.text(0.05,0.8,'Lead ' + str(ld2),transform=ax3.transAxes,ha='left',fontsize='large')
ax3.text(.95,0.05,'g)',transform=ax3.transAxes,ha='right',va='bottom',fontsize='xx-large',fontweight='bold')
ax3.yaxis.get_major_locator().set_params(integer=True)

#plt.subplot(4,4,8)
ax4 = fig.add_subplot(gs2[3])
ax4.plot(bse_hefs_l2[0,:],bse_hefs_l2[1,:],color=chefs,linewidth=2)
ax4.set_xlabel('Spread ($\sigma$)',fontsize='large')
#plt.ylabel('Error ($\sqrt{mse}$)')
ax4.yaxis.set_ticklabels([])
#plt.gca().xaxis.set_ticklabels([])
#plt.title('BSE Diagram')
for i in range(nsamps):
    ax4.plot(bse_v2_l2[i][0,:],bse_v2_l2[i][1,:],color=cv2,linewidth=1.5,alpha=0.2)
#ax4.legend(['HEFS','sHEFS-V1'])
ax4.axline((0,0),slope=1,color='gray',linestyle='--')
ax4.plot(bse_hefs_l2[0,:],bse_hefs_l2[1,:],color=chefs,linewidth=2)
ax4.set_xlim([0,axlim*1.05])
ax4.set_ylim([0,axlim*1.05])
#ax4.text(0.05,0.9,str(pcnt_bse[0])+'-'+str(pcnt_bse[1])+' percentile',transform=ax4.transAxes,ha='left',fontsize='large')
if pcnt_bse[0]==0 and pcnt_bse[1]==1:
    ax4.text(.05,.9,'all flows',transform=ax4.transAxes,ha='left',fontsize='large')
else:
    ax4.text(.05,.9,'top '+str(int(round((1-pcnt_bse[0])*100,0)))+'% flows',transform=ax4.transAxes,ha='left',fontsize='large')
ax4.text(0.05,0.8,'Lead ' + str(ld2),transform=ax4.transAxes,ha='left',fontsize='large')
ax4.text(.95,0.05,'h)',transform=ax4.transAxes,ha='right',va='bottom',fontsize='xx-large',fontweight='bold')
ax4.yaxis.get_major_locator().set_params(integer=True)


#----------------------------------------------------------------------------
#4. eCRPS diagrams

ecrps_ss_v1_inp = np.load('data/%s/%s/%s-%s_ecrps-ss_syn-forecast%s_nsamp=%s.npz' %(loc,site,loc,site,syn_vers1+vers_out1,nsamps))['arr']
ecrps_ss_v2_inp = np.load('data/%s/%s/%s-%s_ecrps-ss_syn-forecast%s_nsamp=%s.npz' %(loc,site,loc,site,syn_vers2+vers_out2,nsamps))['arr']
ecrps_ss_hefs_inp = np.load('data/%s/%s/%s-%s_ecrps-ss_hefs.npz' %(loc,site,loc,site))['arr']

ecrps_ss_hefs = np.empty((len(lds)))
ecrps_ss_v1 = np.empty((len(lds),nsamps))
ecrps_ss_v2 = np.empty((len(lds),nsamps))

for i in range(len(lds)):
    ecrps_ss_hefs[i] = ecrps_ss_hefs_inp[1,lds[i]-1]
    ecrps_ss_v1[i,:] = ecrps_ss_v1_inp[1,lds[i]-1,:]
    ecrps_ss_v2[i,:] = ecrps_ss_v2_inp[1,lds[i]-1,:]

ax1 = fig.add_subplot(gs0[2])
box1 = ax1.boxplot([ecrps_ss_v1[0,:],ecrps_ss_v1[1,:],ecrps_ss_v1[2,:],ecrps_ss_v1[3,:]],positions=[0.9,1.9,2.9,3.9],widths=0.2,whis=(0,100),patch_artist=True)
plt.setp(box1["boxes"], edgecolor=cv1, facecolor='white')
plt.setp(box1["medians"], color=cv1)
plt.setp(box1["whiskers"], color=cv1)
plt.setp(box1["caps"], color=cv1)
box2 = ax1.boxplot([ecrps_ss_v2[0,:],ecrps_ss_v2[1,:],ecrps_ss_v2[2,:],ecrps_ss_v2[3,:]],positions=[1.1,2.1,3.1,4.1],widths=0.2,whis=(0,100),patch_artist=True)
plt.setp(box2["boxes"], edgecolor=cv2, facecolor='white')
plt.setp(box2["medians"], color=cv2)
plt.setp(box2["whiskers"], color=cv2)
plt.setp(box2["caps"], color=cv2)
ax1.set_xticks([1,2,3,4],['Lead %s' %(lds[0]),'Lead %s' %(lds[1]),'Lead %s' %(lds[2]),'Lead %s' %(lds[3])],fontsize='large')
l1 = ax1.scatter((0.75,1.75,2.75,3.75),(ecrps_ss_hefs[0],ecrps_ss_hefs[1],ecrps_ss_hefs[2],ecrps_ss_hefs[3]),s=50,marker='^',color=chefs)
#syn1 = mpatches.Patch(color=cv1, label='syn-HEFS: V1')
#syn2 = mpatches.Patch(color=cv2, label='syn-HEFS: V2')
syn1 = mlines.Line2D([],[],color=cv1, label='syn-M1')
syn2 = mlines.Line2D([],[],color=cv2, label='syn-M2')
ax1.legend([l1,syn1,syn2],['HEFS','syn-M1','syn-M2'],fontsize='x-large',frameon=False)
ax1.set_xlim([0,5])
ax1.set_ylim([0,1])
#ax1.set_title('Ensemble CRPS Skill Score')
ax1.set_ylabel('CRPS skill score',fontsize='large')
#ax1.text(.95,.75,str(low_pcnt_ecrps[0])+'-'+str(low_pcnt_ecrps[1])+' percentile',fontsize='large',transform=ax1.transAxes,ha='right')
ax1.text(.95,.75,'lower '+str(int(round((low_pcnt_ecrps[1])*100,0)))+'% of flows',transform=ax1.transAxes,ha='right',fontsize='large')
ax1.text(.05,0.95,'i)',transform=ax1.transAxes,ha='left',va='top',fontsize='xx-large',fontweight='bold')

ecrps_ss_hefs = np.empty((len(lds)))
ecrps_ss_v1 = np.empty((len(lds),nsamps))
ecrps_ss_v2 = np.empty((len(lds),nsamps))

for i in range(len(lds)):
    ecrps_ss_hefs[i] = ecrps_ss_hefs_inp[0,lds[i]-1]
    ecrps_ss_v1[i,:] = ecrps_ss_v1_inp[0,lds[i]-1,:]
    ecrps_ss_v2[i,:] = ecrps_ss_v2_inp[0,lds[i]-1,:]

ax2 = fig.add_subplot(gs0[3])
box1 = ax2.boxplot([ecrps_ss_v1[0,:],ecrps_ss_v1[1,:],ecrps_ss_v1[2,:],ecrps_ss_v1[3,:]],positions=[0.9,1.9,2.9,3.9],widths=0.2,whis=(0,100),patch_artist=True)
plt.setp(box1["boxes"], edgecolor=cv1, facecolor='white')
plt.setp(box1["medians"], color=cv1)
plt.setp(box1["whiskers"], color=cv1)
plt.setp(box1["caps"], color=cv1)
box2 = ax2.boxplot([ecrps_ss_v2[0,:],ecrps_ss_v2[1,:],ecrps_ss_v2[2,:],ecrps_ss_v2[3,:]],positions=[1.1,2.1,3.1,4.1],widths=0.2,whis=(0,100),patch_artist=True)
plt.setp(box2["boxes"], edgecolor=cv2, facecolor='white')
plt.setp(box2["medians"], color=cv2)
plt.setp(box2["whiskers"], color=cv2)
plt.setp(box2["caps"], color=cv2)
ax2.set_xticks([1,2,3,4],['Lead %s' %(lds[0]),'Lead %s' %(lds[1]),'Lead %s' %(lds[2]),'Lead %s' %(lds[3])],fontsize='large')
l1 = ax2.scatter((0.75,1.75,2.75,3.75),(ecrps_ss_hefs[0],ecrps_ss_hefs[1],ecrps_ss_hefs[2],ecrps_ss_hefs[3]),s=50,marker='^',color=chefs)
#syn1 = mpatches.Patch(color=cv1, label='syn-HEFS: V1')
#syn2 = mpatches.Patch(color=cv2, label='syn-HEFS: V2')
syn1 = mlines.Line2D([],[],color=cv1, label='syn-M1')
syn2 = mlines.Line2D([],[],color=cv2, label='syn-M2')
ax2.legend([l1,syn1,syn2],['HEFS','syn-M1','syn-M2'],fontsize='x-large',frameon=False)
ax2.set_xlim([0,5])
ax2.set_ylim([0,1])
#ax2.set_title('Ensemble CRPS Skill Score')
#plt.ylabel('eCRPS skill score')
#ax2.text(.95,.75,str(upr_pcnt_ecrps[0])+'-'+str(upr_pcnt_ecrps[1])+' percentile',fontsize='large',transform=ax2.transAxes,ha='right')
ax2.text(.95,.75,'top '+str(int(round((1-upr_pcnt_ecrps[0])*100,0)))+'% of flows',transform=ax2.transAxes,ha='right',fontsize='large')
ax2.text(.05,0.95,'j)',transform=ax2.transAxes,ha='left',va='top',fontsize='xx-large',fontweight='bold')
ax2.yaxis.set_ticklabels([])

fig.savefig('./figs/%s/%s/%s-%s_10panel_cumul-rnk-hist_bse_ecrps_val-lds%s,%s_synvers%s,%s.png' %(loc,site,loc,site,ld,ld2,syn_vers1+vers_out1,syn_vers2+vers_out2),dpi=300,bbox_inches='tight')

#sys.modules[__name__].__dict__.clear()
#---------------------------------------------------------------------end------------------------------------------------------------------