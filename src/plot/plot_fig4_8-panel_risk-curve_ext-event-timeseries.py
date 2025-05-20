# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:26:54 2024

@author: zpb4
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../Synthetic-Forecast_Verification/src'))
import numpy as np
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import model
import syn_util
from time import localtime, strftime
from datetime import datetime
import matplotlib.dates as mdates
import ensemble_verification_functions as verify
import pickle

col_cb = sns.color_palette('colorblind')
sns.palplot(col_cb)  #plot coloblind palette for comparison
colv1 = sns.color_palette('PuRd',10)
colv2 = sns.color_palette('YlOrBr',10)

cv2 = colv1[4]  #version 2 is pink-purplish color
cv1 = colv2[6]  #version 1 is orangey-brown color
chefs = col_cb[0]  #hefs is colorblind blue
cb_grn = col_cb[2]
cb_brwn = col_cb[5]
cb_gry = col_cb[7]

#sns.palplot((col_cb[4],cv1,col_cb[3],cv2)) #base colors pretty close to 'colorblind' palette
kcfs_to_tafd = 2.29568411*10**-5 * 86400
kcfs_to_m3s = 0.0283168466 * 1000
taf_to_m3 = 1233.4818375475 * 1000
taf_to_km3 = 0.0000012335 * 1000

1943*kcfs_to_m3s/1000
#K = 3524 # TAF
#Rmax = 150 * kcfs_to_tafd # estimate - from MBK
#ramping_rate = 30868/1000 * kcfs_to_tafd # cfs to kcfs to tafd
max_lds = 14

sd = '1990-10-01' 
ed = '2019-08-15'

loc = 'NHG'
site = 'NHGC1'
svers = '5fold'
p = 0
evt_no = 2
# ORDC1 declustered top 5 events: 1 2 6 8 12
# ADOC1 declustered top 3 events: 1 2 5 
# LAMC1 declustered top 3 events: 1 2 3 
# NHGC1 declustered top 3 events: 1 2 3 
# NBBC1 declustered top 3 events: 1 2 3 

opt_forecast = 'hefs'
syn_samp = 'hefs'
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

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#plots
#1. 3 x 2 plot of data subsets as defined below
sns.set_theme()
sns.set_style('ticks')
sns.set_context('paper')

R99_base = np.sort(R_base)[int(0.99*len(R_base))]
R99_hefs = np.sort(R_hefs)[int(0.99*len(R_hefs))]

#f, ((a1,a2,a7,a8),(a3,a4,a9,a10),(a5,a6,a11,a12)) = plt.subplots(3,4,figsize=(12,7),sharex=True,layout='constrained')
f = plt.figure(layout='constrained',figsize=(11,7))
#gs0 = fig.add_gridspec(2,2,width_ratios=[1,2])
#ax1 = fig.add_subplot(gs0[0])
gs0 = f.add_gridspec(1,2,width_ratios=(1,1.25))
gs1 = gs0[0].subgridspec(2,1)
gs2 = gs0[1].subgridspec(3,2)
#gs3 = gs0[3].subgridspec(3,2)

a0 = f.add_subplot(gs1[0])

#f, (a0,a1) = plt.subplots(1,2,width_ratios=[1,2])
#plt.subplot(1,2,1)
a0.plot(np.arange(max_lds)+1,risk_curve,linewidth=2,color=chefs)
a0.set_xlabel('Lead time (days)',fontsize='large')
a0.set_ylabel('Risk threshold',fontsize='large')
#a0.set_title('Risk curve',fontsize='x-large',fontweight='bold')
a0.set_ylim([0,1.01])
a0.set_xlim([1,max_lds])
#a0.text(6,0.1,'FIRO pool: '+str(round(pars['firo_pool'],3)) + ' (%s TAF)' %(round((pars['firo_pool']+0.5)*K_scale)),fontsize='x-large')
a0.text(1.5,.9,'a)',fontsize='xx-large',fontweight='bold')
a0.tick_params(axis='both',which='major',labelsize='large')


start_sl = '1993-10-01'
end_sl = '1997-09-30'
evt = '1993-10-15'

dt_idx = df_idx.slice_indexer(start_sl ,end_sl)
evt_idx = df_idx.get_loc(evt)
slice_len = len(df_idx[dt_idx])
xlims = df_idx[df_idx.get_loc(start_sl)],df_idx[df_idx.get_loc(end_sl)]

spill_hefs_idx = np.where(spill_hefs[dt_idx]>0)

a00 = f.add_subplot(gs1[1])
dt_format=mdates.DateFormatter('%m/%y')
a00.xaxis.set_major_formatter(dt_format)

#1b. plot baseline ops vs hefs 
l1, = a00.plot(df_idx[dt_idx], tocs[dt_idx]* taf_to_km3, c=cb_gry,alpha=0.5,linewidth=1)
l2, = a00.plot(df_idx[dt_idx], firo_base[dt_idx]* taf_to_km3, c = cb_brwn,alpha=0.5,linewidth=1)
#leg1 = a00.legend([l1,l2],['TOCS','K'],loc='upper left',fontsize='large')
#a00.add_artist(leg1)
l3, = a00.plot(df_idx[dt_idx], S_base[dt_idx]* taf_to_km3, c='black',alpha=0.6,linewidth=1.5)
l4, = a00.plot(df_idx[dt_idx], S_hefs[dt_idx]* taf_to_km3, c=chefs,linewidth=1.5)
a00.legend([l3,l4,l1],['$S_{base}$','$S_{HEFS}$','TOCS'],loc='upper left',fontsize='large')
a00.set_ylabel('Storage ($km^3$)',fontsize='large')
a00.set_xlabel('Date',fontsize='large')
a00.set_xlim([xlims[0],xlims[1]])
#a00.set_title('Storage timeseries',fontsize='x-large',fontweight='bold')
a00.set_ylim([0.5*K_scale * taf_to_km3, K_scale*1.25 * taf_to_km3])
#a00.text(df_idx[evt_idx],K_scale*1.1,r'$\overline{S}:$'+str(round(np.mean(S_base[dt_idx]),1)) +' TAF',fontsize='small')
#a00.text(df_idx[evt_idx],K_scale*1.05,r'$\overline{S}:$'+str(round(np.mean(S_hefs[dt_idx]),1)) +' TAF',color=chefs,fontsize='small')
#a00.text(df_idx[evt_idx+365],K_scale*1.1,r'$spills:$'+str(np.count_nonzero(spill_base)),fontsize='small')
#a00.text(df_idx[evt_idx+365],K_scale*1.05,r'$spills:$'+str(np.count_nonzero(spill_hefs)),color=chefs,fontsize='small')
#a00.text(df_idx[evt_idx+365+225],K_scale*1.1,r'$R\geq{R_{max}}:$'+str(np.count_nonzero(R_base[R_base>Rmax_scale])),fontsize='small')
#a00.text(df_idx[evt_idx+365+225],K_scale*1.05,r'$R\geq{R_{max}}:$'+str(np.count_nonzero(R_hefs[R_hefs>Rmax_scale])),color=chefs,fontsize='small')
#a00.text(df_idx[evt_idx+365+365+180],K_scale*1.1,r'$R_{99}:$'+str(round(R99_base,1)),fontsize='small')
#a00.text(df_idx[evt_idx+365+365+180],K_scale*1.05,r'$R_{99}:$'+str(round(R99_hefs,1)),color=chefs,fontsize='small')
#a00.text(df_idx[evt_idx+365+365+365+90],K_scale*1.1,r'$R_{max}:$'+str(round(np.max(R_base),1)),fontsize='small')
#a00.text(df_idx[evt_idx+365+365+365+90],K_scale*1.05,r'$R_{max}:$'+str(round(np.max(R_hefs),1)),color=chefs,fontsize='small')
a00.text(df_idx[evt_idx+365+90],K_scale*1.2* taf_to_km3,r'$K:$'+str(round(K_scale* taf_to_km3,3))+' $km^3$',fontweight='bold',fontsize='large')
a00.text(df_idx[evt_idx+365+90],K_scale*1.15* taf_to_km3,r'$R_{max}:$'+str(round(Rmax_scale/ kcfs_to_tafd * kcfs_to_m3s,1)) + ' $m^3/s$',fontweight='bold',fontsize='large')
a00.text(df_idx[evt_idx+365+90],K_scale*1.08* taf_to_km3,r'$\overline{S}:$'+str(round(np.mean(S_base[dt_idx])* taf_to_km3,3)) +' $km^3$',fontsize='large')
a00.text(df_idx[evt_idx+365+90],K_scale*1.03* taf_to_km3,r'$\overline{S}:$'+str(round(np.mean(S_hefs[dt_idx])* taf_to_km3,3)) +' $km^3$',color=chefs,fontsize='large')
a00.text(df_idx[evt_idx+365+365+180],K_scale*1.08* taf_to_km3,r'$R_{99}:$'+str(round(R99_base/ kcfs_to_tafd * kcfs_to_m3s,1))+' $m^3/s$',fontsize='large')
a00.text(df_idx[evt_idx+365+365+180],K_scale*1.03* taf_to_km3,r'$R_{99}:$'+str(round(R99_hefs/ kcfs_to_tafd * kcfs_to_m3s,1))+' $m^3/s$',color=chefs,fontsize='large')
a00.text(df_idx[dt_idx][30],.55*K_scale* taf_to_km3,'b)',fontsize='xx-large',fontweight='bold')
if np.count_nonzero(spill_hefs_idx) > 0:
    for i in range(len(spill_hefs_idx)):
        a00.axvline(df_idx[dt_idx][spill_hefs_idx][i],color='red',linewidth=2,alpha=0.3)
#plt.setp(plt.xticks()[1],rotation=30,ha='right')
a00.tick_params(axis='both',which='major',labelsize='large')

evt_idx = int(np.where(Q == np.sort(Q)[-evt_no])[0])
evt = df_idx[evt_idx].strftime('%Y-%m-%d')
pad = 15
#evt_idx=df_idx.get_loc(evt)
st,ed = df_idx[evt_idx-pad],df_idx[evt_idx+pad]
start,end = str(st)[0:10],str(ed)[0:10]
dt_idx=df_idx.slice_indexer(st,ed)
dt_format=mdates.DateFormatter('%m/%d')

#reliability diagrams for a given thresholds
#Q_cp_idx = np.where((dowy_hefs > 60) & (dowy_hefs < 170) & (Q_hefs > 0))[0]
R_srt = np.sort(R_hefs)
upr_idx = round(0.99 * len(R_srt))
R_thresh = R_srt[upr_idx]

v1_spill = np.apply_along_axis(np.count_nonzero, 1, sim_data1[:,dt_idx,4])
v1_spills = np.count_nonzero(v1_spill)

a1 = f.add_subplot(gs2[0])

ymn = 0.95*min(np.min(sim_data1[:,dt_idx,0]),np.min(sim_data2[:,dt_idx,0]))
ymx = 1.2* (K_scale-ymn)+ymn

l1, = a1.plot(df_idx[dt_idx], firo_base[dt_idx]*taf_to_km3, c = cb_brwn, linewidth=1,alpha=0.5)
#l2, = a1.plot(df_idx[dt_idx], tocs_base[dt_idx], c = cb_gry, linewidth=1,alpha=0.5)
leg1 = a1.legend([l1],['K'],loc='upper left',fontsize='large',frameon=False)
a1.add_artist(leg1)
for i in range(nsamps):
    a1.plot(df_idx[dt_idx], sim_data1[i,dt_idx,0]*taf_to_km3, c=cv1,alpha=0.1)
#l3, = a1.plot(df_idx[dt_idx], S_base[dt_idx], c='black',linewidth=2)
l4, = a1.plot(df_idx[dt_idx], S_hefs[dt_idx]*taf_to_km3, c=chefs,linewidth=2)
l5, = a1.plot(0, 0, c=cv1,alpha=0.5)
a1.legend([l4,l5],['$S_{HEFS}$','$S_{synM1}$'],bbox_transform=a1.transAxes,loc=(.5,.01),fontsize='large',frameon=False)
a1.axvline(df_idx[evt_idx],linewidth=0.5,color='black',linestyle='--')
a1.text(df_idx[evt_idx],(1.05*(K_scale-ymn)+ymn)*taf_to_km3,evt,fontsize='large')
a1.set_ylabel('Storage ($km^3$)',fontsize='large')
a1.text(df_idx[evt_idx-pad+1],(.25*(ymx-ymn)+ymn)*taf_to_km3,'Spill: %s/%s' %(v1_spills,len(v1_spill)),fontsize='large',fontweight='bold')
a1.text(df_idx[evt_idx-pad+1],(.05*(ymx-ymn)+ymn)*taf_to_km3,'c)',fontsize='xx-large',fontweight='bold')
a1.set_ylim([ymn*taf_to_km3, ymx*taf_to_km3])
a1.set_xlim([st, ed])
a1.xaxis.set_major_formatter(dt_format)
#plt.gcf().autofmt_xdate()
a1.xaxis.set_ticklabels([])
a1.tick_params(axis='both',which='major',labelsize='large')

v2_spill = np.apply_along_axis(np.count_nonzero, 1, sim_data2[:,dt_idx,4])
v2_spills = np.count_nonzero(v2_spill)

a2 = f.add_subplot(gs2[1])

l1, = a2.plot(df_idx[dt_idx], firo_base[dt_idx]*taf_to_km3, c = cb_brwn, linewidth=1,alpha=0.5)
#l2, = a2.plot(df_idx[dt_idx], tocs_base[dt_idx], c = cb_gry, linewidth=1,alpha=0.5)
leg1 = a2.legend([l1],['K'],loc='upper left',fontsize='large',frameon=False)
a2.add_artist(leg1)
for i in range(nsamps):
    a2.plot(df_idx[dt_idx], sim_data2[i,dt_idx,0]*taf_to_km3, c=cv2,alpha=0.1)
#l3, = a2.plot(df_idx[dt_idx], S_base[dt_idx], c='black',linewidth=2)
l4, = a2.plot(df_idx[dt_idx], S_hefs[dt_idx]*taf_to_km3, c=chefs,linewidth=2)
l5, = a2.plot(0, 0, c=cv2,alpha=0.5)
a2.legend([l4,l5],['$S_{HEFS}$','$S_{synM2}$'],bbox_transform=a2.transAxes,loc=(.55,.01),fontsize='large',frameon=False)
a2.axvline(df_idx[evt_idx],linewidth=0.5,color='black',linestyle='--')
a2.text(df_idx[evt_idx],(1.05*(K_scale-ymn)+ymn)*taf_to_km3,evt,fontsize='large')
a2.text(df_idx[evt_idx-pad+1],(.25*(ymx-ymn)+ymn)*taf_to_km3,'Spill: %s/%s' %(v2_spills,len(v2_spill)),fontsize='large',fontweight='bold')
a2.text(df_idx[evt_idx-pad+1],(.05*(ymx-ymn)+ymn)*taf_to_km3,'d)',fontsize='xx-large',fontweight='bold')
#a2.set_ylabel('TAF')
a2.set_ylim([ymn*taf_to_km3, ymx*taf_to_km3])
a2.set_xlim([st, ed])
a2.xaxis.set_major_formatter(dt_format)
#plt.gcf().autofmt_xdate()
a2.yaxis.set_ticklabels([])
a2.xaxis.set_ticklabels([])
a2.tick_params(axis='both',which='major',labelsize='large')

a3 = f.add_subplot(gs2[2])

for i in range(nsamps):
    a3.plot(df_idx[dt_idx], sim_data1[i,dt_idx,5] / kcfs_to_tafd * kcfs_to_m3s, c=cv1,alpha=0.1)
l3, = a3.plot(0, 0, c=cv1,alpha=0.5)
#l1, = a3.plot(df_idx[dt_idx], Q_cp_base[dt_idx] / kcfs_to_tafd,c='black',linewidth=2)
l1, = a3.plot(df_idx[dt_idx], Q[dt_idx] / kcfs_to_tafd* kcfs_to_m3s,c='black',linewidth=2,alpha=0.5)
l2, = a3.plot(df_idx[dt_idx], R_hefs[dt_idx] / kcfs_to_tafd* kcfs_to_m3s,c=chefs,linewidth=2)
#a3.legend([l1,l2,l3],['$Q$','$R_{HEFS}$','$R_{synM1}$'],bbox_transform=a3.transAxes,loc=(.65,.45),fontsize='medium',frameon=False)
#a3.legend([l2,l3,l1],['$R_{HEFS}$','$R_{synM1}$','$Q$'],bbox_transform=a3.transAxes,loc=(0,.43),fontsize='large',frameon=False)
a3.legend([l2,l3,l1],['$R_{HEFS}$','$R_{synM1}$','$Q$'],bbox_transform=a3.transAxes,loc=(0.57,.43),fontsize='large',frameon=False)
a3.axhline(Rmax_scale / kcfs_to_tafd* kcfs_to_m3s, color='red')
#a3.axhline(R_thresh, color=chefs,linewidth=0.5,alpha=0.5)
a3.axvline(df_idx[evt_idx],linewidth=0.5,color='black',linestyle='--')
#a3.text(df_idx[evt_idx],1.15 * Rmax_scale / kcfs_to_tafd,evt,fontsize='small')
a3.text(df_idx[evt_idx+int(pad/2)],1.025*Rmax_scale / kcfs_to_tafd* kcfs_to_m3s,'$R_{max}$',color='red',fontsize='large')
a3.set_ylabel('Streamflow ($m^3/s$)',fontsize='large')
a3.text(df_idx[evt_idx-pad+1],1.1 * Rmax_scale / kcfs_to_tafd* kcfs_to_m3s,'e)',fontsize='xx-large',fontweight='bold')
a3.set_ylim([0, 1.25 * Rmax_scale / kcfs_to_tafd* kcfs_to_m3s])
a3.set_xlim([st, ed])
a3.xaxis.set_major_formatter(dt_format)
#plt.gcf().autofmt_xdate()
a3.xaxis.set_ticklabels([])
a3.tick_params(axis='both',which='major',labelsize='large')

a4 = f.add_subplot(gs2[3])

for i in range(nsamps):
    a4.plot(df_idx[dt_idx], sim_data2[i,dt_idx,5] / kcfs_to_tafd* kcfs_to_m3s, c=cv2,alpha=0.1)
l3, = a4.plot(0, 0, c=cv2,alpha=0.5)
#l1, = a4.plot(df_idx[dt_idx], Q_cp_base[dt_idx] / kcfs_to_tafd,c='black',linewidth=2)
l1, = a4.plot(df_idx[dt_idx], Q[dt_idx] / kcfs_to_tafd* kcfs_to_m3s,c='black',linewidth=2,alpha=0.5)
l2, = a4.plot(df_idx[dt_idx], R_hefs[dt_idx] / kcfs_to_tafd* kcfs_to_m3s,c=chefs,linewidth=2)
#a4.legend([l2,l3,l1],['$R_{HEFS}$','$R_{synM2}$','$Q$'],bbox_transform=a4.transAxes,loc=(0,.43),fontsize='large',frameon=False)
a4.legend([l2,l3,l1],['$R_{HEFS}$','$R_{synM2}$','$Q$'],bbox_transform=a4.transAxes,loc=(0.57,.43),fontsize='large',frameon=False)
#a4.legend([l2,l3],['$R_{HEFS}$','$R_{synM2}$'],bbox_transform=a4.transAxes,loc=(.05,.45),fontsize='large',frameon=False)
a4.axhline(Rmax_scale / kcfs_to_tafd* kcfs_to_m3s, color='red')
#a4.axhline(R_thresh, color=chefs,linewidth=0.5,alpha=0.5)
a4.text(df_idx[evt_idx+int(pad/2)],1.025*Rmax_scale / kcfs_to_tafd* kcfs_to_m3s,'$R_{max}$',color='red',fontsize='large')
a4.axvline(df_idx[evt_idx],linewidth=0.5,color='black',linestyle='--')
#a4.text(df_idx[evt_idx],1.15 * Rmax_scale / kcfs_to_tafd,evt,fontsize='small')
a4.text(df_idx[evt_idx-pad+1],1.1 * Rmax_scale / kcfs_to_tafd* kcfs_to_m3s,'f)',fontsize='xx-large',fontweight='bold')
#a4.set_ylabel('kcfs')
a4.set_ylim([0, 1.25 * Rmax_scale / kcfs_to_tafd* kcfs_to_m3s])
a4.set_xlim([st, ed])
a4.xaxis.set_major_formatter(dt_format)
#plt.gcf().autofmt_xdate()
a4.yaxis.set_ticklabels([])
a4.xaxis.set_ticklabels([])
a4.tick_params(axis='both',which='major',labelsize='large')

a5 = f.add_subplot(gs2[4])

pcntiles = ([0.9]) #choose 4
pcntile_idx = ([(1-pcntiles[0])/2,pcntiles[0]+((1-pcntiles[0])/2)])
rel_sset1 = verify.percentile_rel_sset(sim_data1[:,dt_idx,6],rel_ld_hefs[dt_idx],pcntile_idx[0],pcntile_idx[1])
cp1 = round(np.sum(rel_sset1[0][2,:]) / len(S_hefs[dt_idx]) * 100,1)

rel_sset2 = verify.percentile_rel_sset(sim_data2[:,dt_idx,6],rel_ld_hefs[dt_idx],pcntile_idx[0],pcntile_idx[1])
cp2 = round(np.sum(rel_sset2[0][2,:]) / len(S_hefs[dt_idx]) * 100,1)
samp = np.random.choice(np.arange(nsamps,dtype=int),size=1)[0]

l3, = a5.plot(df_idx[dt_idx], np.apply_along_axis(np.median, axis=0,arr= sim_data1[:,dt_idx,6]), c=cv1,alpha=0.2)
a5.fill_between(df_idx[dt_idx],rel_sset1[0][0,:],rel_sset1[0][1,:],color=cv1,alpha=0.2)
#l2, = a5.plot(df_idx[dt_idx], sim_data1[samp,dt_idx,6], c=cv1,alpha=0.75)
l2, = a5.plot(df_idx[dt_idx], np.apply_along_axis(np.median, axis=0,arr= sim_data1[:,dt_idx,6]), c=cv1,alpha=0.75)
l1, = a5.plot(df_idx[dt_idx], rel_ld_hefs[dt_idx], c = chefs,linewidth=2)
a5.legend([l1,l2],['$R^{ld}_{HEFS}$','$R^{ld}_{synM1}$'],loc='upper right',fontsize='large',frameon=False)
a5.axvline(df_idx[evt_idx],linewidth=0.5,color='black',linestyle='--')
#a5.text(df_idx[evt_idx-int(pad*0.7)],14,evt,fontsize='small')
a5.set_ylabel('Release lead time (days)',fontsize='large')
a5.text(df_idx[evt_idx-pad+1],13,'g)',fontsize='xx-large',fontweight='bold')
a5.set_ylim([0, 15.1])
a5.set_xlim([st, ed])
a5.tick_params(axis='both',which='major',labelsize='large')

textstr = '\n'.join((
    r'CI: %s' %(round(pcntiles[0]*100,1)) + ' %',
    r'CP: %s' %(cp1) + ' %'))

#a5.text(.95,.1,textstr,transform=a5.transAxes,fontsize='medium',fontweight='bold',bbox=dict(facecolor='white', alpha=0.75),ha='right',va='center')
#a5.text(df_idx[evt_idx+pad-8],6.5,'CP: %s' %(cp1) + ' %',fontsize='small',fontweight='bold',bbox=dict(facecolor='white', alpha=0.75))
#a5.text(df_idx[evt_idx+pad-8],8,'CI: %s' %(round(pcntiles[0]*100,1)) + ' %',fontsize='small',fontweight='bold',bbox=dict(facecolor='white', alpha=0.75))
a5.xaxis.set_major_formatter(dt_format)
#plt.gcf().autofmt_xdate()
plt.setp(plt.xticks()[1],rotation=45,ha='right')

a6 = f.add_subplot(gs2[5])

l3, = a6.plot(df_idx[dt_idx], np.apply_along_axis(np.median, axis=0,arr= sim_data2[:,dt_idx,6]), c=cv2,alpha=0.2)
a6.fill_between(df_idx[dt_idx],rel_sset2[0][0,:],rel_sset2[0][1,:],color=cv2,alpha=0.2)
#l2, = a6.plot(df_idx[dt_idx], sim_data2[samp,dt_idx,6], c=cv2,alpha=0.75)
l2, = a6.plot(df_idx[dt_idx], np.apply_along_axis(np.median, axis=0,arr= sim_data2[:,dt_idx,6]), c=cv2,alpha=0.75)
l1, = a6.plot(df_idx[dt_idx], rel_ld_hefs[dt_idx], c = chefs,linewidth=2)
a6.legend([l1,l2],['$R^{ld}_{HEFS}$','$R^{ld}_{synM2}$'],loc='upper right',fontsize='large',frameon=False)
a6.axvline(df_idx[evt_idx],linewidth=0.5,color='black',linestyle='--')
#a6.text(df_idx[evt_idx-int(pad*0.7)],14,evt,fontsize='small')
#a6.set_ylabel('Release lead time (days)')
a6.set_ylim([0, 15.1])
a6.set_xlim([st, ed])
textstr = '\n'.join((
    r'CI: %s' %(round(pcntiles[0]*100,1)) + ' %',
    r'CP: %s' %(cp2) + ' %'))
#a6.text(.95,.1,textstr,transform=a6.transAxes,fontsize='medium',fontweight='bold',bbox=dict(facecolor='white', alpha=0.75),ha='right',va='center')
#a6.text(df_idx[evt_idx+pad-8],6.5,'CP: %s' %(cp2) + ' %',fontsize='small',fontweight='bold',bbox=dict(facecolor='white', alpha=0.75))
#a6.text(df_idx[evt_idx+pad-8],8,'CI: %s' %(round(pcntiles[0]*100,1)) + ' %',fontsize='small',fontweight='bold',bbox=dict(facecolor='white', alpha=0.75))
a6.text(df_idx[evt_idx-pad+1],13,'h)',fontsize='xx-large',fontweight='bold')
a6.xaxis.set_major_formatter(dt_format)
plt.setp(plt.xticks()[1],rotation=45,ha='right')
a6.yaxis.set_ticklabels([])
a6.tick_params(axis='both',which='major',labelsize='large')

f.savefig('./figs/%s/%s/4x2_risk-curve-ts-S-Qcp-Rlead_svers-%s_Krat=%s_Rmaxrat=%s_RRrat=%s_seed-%s_evt=%s.png' %(loc,site,svers,round(K_ratios[p],2),round(Rmax_ratios[p],2),round(rr_ratios[p],2),seed,evt_no),dpi=300,bbox_inches='tight')

#sys.modules[__name__].__dict__.clear()

#--------------------------------------end--------------------------------------
