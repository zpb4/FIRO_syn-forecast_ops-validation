# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:01:40 2024

@author: zpb4
"""
import matplotlib.pyplot as plt
import syn_util
import seaborn as sns
import numpy as np
import pandas as pd

col_cb = sns.color_palette('colorblind')

max_lds=14

lo = .1
hi = .8
pwr = 0.25
no_risk = 0
all_risk = 0

risk_curve1 = syn_util.create_param_risk_curve((lo,hi,pwr,no_risk,all_risk),max_lds)

lo = .1
hi = .8
pwr = -0.25
no_risk = 0
all_risk = 0

risk_curve2 = syn_util.create_param_risk_curve((lo,hi,pwr,no_risk,all_risk),max_lds)

lo = .1
hi = .8
pwr = 0.0001
no_risk = 0
all_risk = 0

risk_curve3 = syn_util.create_param_risk_curve((lo,hi,pwr,no_risk,all_risk),max_lds)

lo = 0
hi = 0
pwr = 0.1
no_risk = 3
all_risk = 3

risk_curve4 = syn_util.create_param_risk_curve((lo,hi,pwr,no_risk,all_risk),max_lds)


 
sns.set_theme()
sns.set_style('ticks')
sns.set_context('paper')

fig = plt.figure(layout='constrained',figsize=(6,4))
gs0 = fig.add_gridspec(1,1)

ax1 = fig.add_subplot(gs0[0])

ax1.plot(np.arange(max_lds+1)[1:],risk_curve3,color=col_cb[0])
ax1.plot(np.arange(max_lds+1)[1:],risk_curve1,color=col_cb[4],linestyle='--')
ax1.plot(np.arange(max_lds+1)[1:],risk_curve2,color=col_cb[2],linestyle='--')
ax1.plot(np.arange(max_lds+1)[1:],risk_curve4,color=col_cb[3])
ax1.set_xlim([1,max_lds])
ax1.set_ylim([0,1])
ax1.set_xlabel('Lead time (days)')
ax1.set_ylabel('Risk threshold')
ax1.text(.5,0.1,'$r_1$')
ax1.text(14.25,0.8,'$r_2$')
ax1.text(7,0.75,'$r_3$')
ax1.text(7,0.15,'$r_3$')
ax1.text(2.5,0.025,'$r_4$')
ax1.text(12.5,0.96,'$r_5$')
ax1.legend(['A','B','C','D'],loc='lower right',fontsize='large')

fig.savefig('./figures/risk-curve-example.png',dpi=300,bbox_inches='tight')



lo = .1
hi = .8
pwr = 0.25
no_risk = 3
all_risk = 3

risk_curve1 = syn_util.create_param_risk_curve((lo,hi,pwr,no_risk,all_risk),max_lds)

lo = .1
hi = .8
pwr = -0.25
no_risk = 3
all_risk = 3

risk_curve2 = syn_util.create_param_risk_curve((lo,hi,pwr,no_risk,all_risk),max_lds)

lo = .1
hi = .8
pwr = 0.0001
no_risk = 3
all_risk = 3

risk_curve3 = syn_util.create_param_risk_curve((lo,hi,pwr,no_risk,all_risk),max_lds)
 
sns.set_theme()
sns.set_style('ticks')
sns.set_context('paper')

fig = plt.figure(layout='constrained',figsize=(6,4))
gs0 = fig.add_gridspec(1,1)

ax1 = fig.add_subplot(gs0[0])

ax1.plot(np.arange(max_lds+1)[1:],risk_curve3,color=col_cb[0],linestyle='--')
ax1.plot(np.arange(max_lds+1)[1:],risk_curve1,color=col_cb[4])
ax1.plot(np.arange(max_lds+1)[1:],risk_curve2,color=col_cb[2],linestyle='--')
ax1.set_xlim([1,max_lds])
ax1.set_ylim([0,1])
ax1.set_xlabel('Lead time (days)')
ax1.set_ylabel('Risk %')
#ax1.text(2,0.95,'$K_{ratio}$: '+str(round(K_ratios[p],2)))
ax1.legend(['A','B','C'],loc='lower right',fontsize='large')

fig.savefig('./figures/risk-curve-example2.png',dpi=300,bbox_inches='tight')



#################################END###############################################
