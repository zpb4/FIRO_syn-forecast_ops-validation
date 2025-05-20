# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:32:35 2024

@author: zpb4
"""
import numpy as np
import pandas as pd
import multiprocessing as mp
import calendar
from syn_util import water_day
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import rankdata
from scipy.stats import kendalltau as ktau
#from scipy.stats import cramervonmises_2samp as cvm2

def onesamp_forecast_rearrange(forc_ensemble):
    n,ne,nl = np.shape(forc_ensemble)
    forc_out = np.zeros_like(forc_ensemble)
    for i in range(nl):
        forc_out[(i+1):,:,i] = forc_ensemble[:(n-(i+1)),:,i]
    return forc_out

def onesamp_forecast_rearrange_cumul(forc_ensemble):
    n,ne,nl = np.shape(forc_ensemble)
    forc_out = np.zeros_like(forc_ensemble)
    for i in range(nl):
        forc_out[(i+1):,:,i] = forc_ensemble[:(n-(i+1)),:,:(i+1)].sum(axis=2)
    return forc_out

def ref_forecast_rearrange_cumul(ref_ensemble,nl):
    n,ne = np.shape(ref_ensemble)
    forc_out = np.zeros((n,ne,nl))
    for i in range(ne):
        forc_out[:,i,:] = tgt_cumul(ref_ensemble[:,i],nl)
    return forc_out

def tgt_cumul(tgt,nl):
    n = len(tgt)
    tgt_out = np.zeros((n,nl))
    for i in range(nl):
        for k in range(n):
            tgt_out[k,i] = np.sum(tgt[(k-i):(k+1)])
    return tgt_out

def multisamp_forecast_rearrange(forc_ensemble):
    nsamp,n,ne,nl = np.shape(forc_ensemble)
    forc_out = np.zeros_like(forc_ensemble)
    for i in range(nsamp):
        forc_out[i,:,:,:] = onesamp_forecast_rearrange(forc_ensemble[i,:,:,:])
    return forc_out

def multisamp_forecast_rearrange_cumul(forc_ensemble):
    nsamp,n,ne,nl = np.shape(forc_ensemble)
    forc_out = np.zeros_like(forc_ensemble)
    for i in range(nsamp):
        forc_out[i,:,:,:] = onesamp_forecast_rearrange_cumul(forc_ensemble[i,:,:,:])
    return forc_out

def ens_rank(ensemble,tgt):
    ens_sort = np.sort(ensemble)
    dif = ens_sort - tgt
    rank = np.zeros((1,1))
    if np.all(dif<0):
        rank = np.shape(dif)[0]
    else:
        y = np.min(dif[dif>=0])
        rnk = np.where(dif==y)
        if np.shape(rnk)[1] > 1:
            rank = np.random.choice(rnk[0],size=1)[0]
        else:
            rank=rnk[0][0]
            
    return rank

def ensemble_crps(ensemble,tgt):
    ne = len(ensemble)
    term1 = (1/ne) * np.sum(np.abs(ensemble - tgt))
    
    #for loop to calculate second ecrps term per Wilks 2019
    #term2 = np.zeros((ne,ne))
    #for i in range(ne-1):
        #for j in range(i+1,ne):
            #term2[i,j] = np.abs(ensemble[i]-ensemble[j])
    
    #elementwise matrix calculation for term2
    mat1 = np.reshape(np.repeat(ensemble,ne),(ne,ne))
    mat2 = np.reshape(np.tile(ensemble,ne),(ne,ne))
    idx = np.tril_indices(ne)
    term2 = np.abs(mat1[idx] - mat2[idx])
    
    term2_result = (1 / (ne * (ne-1))) * np.sum(term2)
    out = term1 - term2_result
    
    return out

def ensemble_crps_sample(ensemble,tgt,forc=False):
    if forc == False:
        ne,n = np.shape(ensemble)
        inp_ens = np.copy(ensemble)
    else:
        n,ne = np.shape(ensemble)
        ens = np.copy(ensemble)
        inp_ens = np.transpose(ens)
    ecrps_out = np.empty(n)
    for i in range(n):
        ecrps_out[i] = ensemble_crps(inp_ens[:,i],tgt[i])
    ecrps_out[ecrps_out < 0] = 0
    return ecrps_out

def onesamp_ecrps(ensemble,tgt,pcntile,forc_sort=False):
    n,ne = np.shape(ensemble)
    lwr_idx = int(pcntile[0]*n)
    upr_idx = int(pcntile[1]*n-1)
    if forc_sort == False:
        obs_srt = np.sort(tgt)
        rtn_idx = np.where((tgt >= obs_srt[lwr_idx]) & (tgt <= obs_srt[upr_idx]))[0]
    elif forc_sort == True:
        ensmean = ensemble[:,:].mean(axis=1)
        forc_srt = np.sort(ensmean)
        rtn_idx = np.where((ensmean >= forc_srt[lwr_idx]) & (ensmean <= forc_srt[upr_idx]))[0]
        
    ecrps_out = ensemble_crps_sample(ensemble[rtn_idx,:],tgt[rtn_idx],forc=True)
    ecrps_mean = np.mean(ecrps_out)
    
    return ecrps_out,ecrps_mean

def multisamp_ecrps(ensemble,tgt,pcntile,par=True,forc_sort=False):
    nsamp,n,ne = np.shape(ensemble)
    lwr_idx = int(pcntile[0]*n)
    upr_idx = int(pcntile[1]*n-1)
    rtn_idx = []
    if forc_sort == False:
        obs_srt = np.sort(tgt)
        for k in range(nsamp):
            rtn_idx.append(np.where((tgt >= obs_srt[lwr_idx]) & (tgt <= obs_srt[upr_idx]))[0])
    elif forc_sort == True:
        for k in range(nsamp):
            ensmean = ensemble[k,:,:].mean(axis=1)
            forc_srt = np.sort(ensmean)
            rtn_idx.append(np.where((ensmean >= forc_srt[lwr_idx]) & (ensmean <= forc_srt[upr_idx]))[0])
    if par==False:
        #ecrps_vec = []
        ecrps_out = np.arange(nsamp)
        for i in range(nsamp):
            ecrps = ensemble_crps_sample(ensemble[i,rtn_idx[i],:],tgt[rtn_idx[i]],forc=True)
            #ecrps_vec.append(ecrps)
            ecrps_out[i] = np.mean(ecrps)
            
    elif par==True:
        global ecrps_par_fun
        def ecrps_par_fun(i):
            ecrps = ensemble_crps_sample(ensemble[i,rtn_idx[i],:],tgt[rtn_idx[i]],forc=True)
            ecrps_mean = np.mean(ecrps)
            return ecrps_mean
    
        pool = mp.Pool(mp.cpu_count()-2)
        ecrps_out = pool.map_async(ecrps_par_fun,np.arange(nsamp)).get()
        pool.close()
    
    #if __name__ == '__main__':
        #pool = mp.Pool(mp.cpu_count()-2)
        #ecrps_out = pool.map_async(ecrps_par_fun,np.arange(nsamp)).get()
        #pool.close()
    
    return ecrps_out
    #return ecrps_out,ecrps_mean

def ecrps_ss(ens_ecrps,ref_ecrps):
    ss = np.empty(len(ens_ecrps))
    for i in range(len(ens_ecrps)):
        ss[i] = 1 - (ens_ecrps[i] / ref_ecrps)
    
    return ss

def create_obs_ref_ens(obs_ref,dowy_ref,sd,ed):
    idx = pd.date_range(start = sd, end = ed )
    dowy = np.array([water_day(d,calendar.isleap(d.year)) for d in idx])
    unique, counts = np.unique(dowy_ref, return_counts=True)
    ens_size = np.min(counts)
    obs_ens = np.empty((len(dowy),ens_size))
    for i in range(len(dowy)):
        obs_idx = np.where(dowy_ref == dowy[i])[0]
        obs_ens[i,:] = obs_ref[obs_idx][:ens_size]
    
    return obs_ens

def binned_spread_error(ensemble,tgt,bins,forc=False):
    if forc == False:
        ne,n = np.shape(ensemble)
        inp_ens = np.copy(ensemble)
    else:
        n,ne = np.shape(ensemble)
        ens = np.copy(ensemble)
        inp_ens = np.transpose(ens)
    ens_var = inp_ens.var(axis=0)
    ens_mn = inp_ens.mean(axis=0)
    var_idx = pd.qcut(ens_var,q = bins,retbins=True,duplicates='drop')
    out_ens = np.empty((2,len(var_idx[1])-1))
    for i in range(len(var_idx[1])-1):
        idx = np.where(var_idx[0].codes == i)[0]
        sprd = (ne/(ne-1)) * (1/len(idx)) * np.sum(ens_var[idx])
        out_ens[0,i] = np.sqrt(sprd)
        mse = (ne/(ne+1)) * (1/len(idx)) * np.sum((ens_mn[idx] - tgt[idx])**2)
        out_ens[1,i] = np.sqrt(mse)
        
    return out_ens

def multisamp_bse(ensemble,tgt,bins,pcntile,forc_sort=False):
    nsamp,n,ne = np.shape(ensemble)
    lwr_idx = int(pcntile[0]*n)
    upr_idx = int(pcntile[1]*n-1)
    rtn_idx = []
    if forc_sort == False:
        obs_srt = np.sort(tgt)
        for k in range(nsamp):
            rtn_idx.append(np.where((tgt >= obs_srt[lwr_idx]) & (tgt <= obs_srt[upr_idx]))[0])
    elif forc_sort == True:
        for k in range(nsamp):
            ensmean = ensemble[k,:,:].mean(axis=1)
            forc_srt = np.sort(ensmean)
            rtn_idx.append(np.where((ensmean >= forc_srt[lwr_idx]) & (ensmean <= forc_srt[upr_idx]))[0])
    bse_out = []
    for i in range(nsamp):
        bse = binned_spread_error(ensemble[i,rtn_idx[i],:],tgt[rtn_idx[i]],bins=bins,forc=True)
        bse_out.append(bse)

    return bse_out

def onesamp_bse(ensemble,tgt,bins,pcntile,forc_sort=False):
    n,ne = np.shape(ensemble)
    lwr_idx = int(pcntile[0]*n)
    upr_idx = int(pcntile[1]*n-1)
    if forc_sort == False:
        obs_srt = np.sort(tgt)
        rtn_idx = np.where((tgt >= obs_srt[lwr_idx]) & (tgt <= obs_srt[upr_idx]))[0]
    elif forc_sort == True:
        ensmean = ensemble[:,:].mean(axis=1)
        forc_srt = np.sort(ensmean)
        rtn_idx = np.where((ensmean >= forc_srt[lwr_idx]) & (ensmean <= forc_srt[upr_idx]))[0]
        
    bse_out = binned_spread_error(ensemble[rtn_idx,:],tgt[rtn_idx],bins=bins,forc=True)
    
    return bse_out

def rank_histogram(ensemble,tgt,forc=False):
    if forc == False:
        ne,n = np.shape(ensemble)
        inp_ens = np.copy(ensemble)
    else:
        n,ne = np.shape(ensemble)
        ens = np.copy(ensemble)
        inp_ens = np.transpose(ens)
    rnks = np.empty(n)
    for i in range(n):
        rnks[i] = ens_rank(inp_ens[:,i],tgt[i])
    freq = np.histogram(rnks,bins=(ne+1),density=False)[0]
    cumul_freq = np.empty(len(freq))
    for k in range(len(freq)):
        cumul_freq[k] = freq[:(k+1)].sum(axis=0) / n
    unif_dens = 1/len(freq)
    ent = entropy(freq)
    
    return freq,cumul_freq,unif_dens,ent

def multisamp_rnk_hist(ensemble,tgt,pcntile,forc_sort=False):
    nsamp,n,ne = np.shape(ensemble)
    lwr_idx = int(pcntile[0]*n)
    upr_idx = int(pcntile[1]*n-1)
    rtn_idx = []
    if forc_sort == False:
        obs_srt = np.sort(tgt)
        for k in range(nsamp):
            rtn_idx.append(np.where((tgt >= obs_srt[lwr_idx]) & (tgt <= obs_srt[upr_idx]))[0])
    elif forc_sort == True:
        for k in range(nsamp):
            ensmean = ensemble[k,:,:].mean(axis=1)
            forc_srt = np.sort(ensmean)
            rtn_idx.append(np.where((ensmean >= forc_srt[lwr_idx]) & (ensmean <= forc_srt[upr_idx]))[0])
    rnk_arr = np.empty((nsamp,ne+1))
    cumul_rnk_arr = np.empty((nsamp,ne+1))
    ent_arr = np.empty(nsamp)
    for i in range(nsamp):
        freq,cumul_freq,unif_dens,ent = rank_histogram(ensemble[i,rtn_idx[i],:],tgt[rtn_idx[i]],forc=True)
        rnk_arr[i,:] = freq
        cumul_rnk_arr[i,:] = cumul_freq
        ent_arr[i] = ent
    return rnk_arr,cumul_rnk_arr,ent_arr,unif_dens

def onesamp_rnk_hist(ensemble,tgt,pcntile,forc_sort=False):
    n,ne = np.shape(ensemble)
    lwr_idx = int(pcntile[0]*n)
    upr_idx = int(pcntile[1]*n-1)
    if forc_sort == False:
        obs_srt = np.sort(tgt)
        rtn_idx = np.where((tgt >= obs_srt[lwr_idx]) & (tgt <= obs_srt[upr_idx]))[0]
    elif forc_sort == True:
        ensmean = ensemble[:,:].mean(axis=1)
        forc_srt = np.sort(ensmean)
        rtn_idx = np.where((ensmean >= forc_srt[lwr_idx]) & (ensmean <= forc_srt[upr_idx]))[0]
        
    rnk_hist,cumul_rnk_hist,unif_dens,ent = rank_histogram(ensemble[rtn_idx,:],tgt[rtn_idx],forc=True)
    
    return rnk_hist,cumul_rnk_hist,ent,unif_dens

def pcntile_fun(pcntiles):
    pcnt_out = np.empty((len(pcntiles),2))
    for i in range(len(pcntiles)):
        lwr,upr = (1-pcntiles[i])/2,pcntiles[i]+((1-pcntiles[i])/2)
        pcnt_out[i,:] = lwr,upr
    return pcnt_out

def chisq(freqs):
    m = len(freqs)
    n = np.sum(freqs)
    chi_square = (m/n) * np.sum((freqs - (n/m))**2)
    prob = gamma.cdf(chi_square,a=((m-1)/2),loc=0,scale=2)
    return chi_square,prob

def chisq_comp(freq_ref,freq_tst,nbins):
    freq_denom = np.zeros_like(freq_ref)
    freq_denom[:] = freq_ref[:]
    freq_denom[freq_denom<1] = 1
    m = nbins
    chi_square = np.sum(((freq_tst-freq_ref)**2) / freq_denom)
    prob = gamma.cdf(chi_square,a=((m-1)/2),loc=0,scale=2)
    return chi_square,prob

def ens_chisq_int(ensemble,tgt,ncat):
    ne,n = np.shape(ensemble)
    freq_ref = np.bincount(np.int_(tgt),minlength=ncat)
    freq_tst_ens = np.bincount(np.int_(np.ndarray.flatten(ensemble))) / ne
    chi_sq,prob = chisq_comp(freq_ref[1:],freq_tst_ens[1:]) #don't care about the 0-lead releases
    freq_arr = np.empty((ne,ncat))
    for k in range(ne):
        freq_arr[k,:] = np.bincount(np.int_(ensemble[k,:]))
    freq_max = freq_arr.max(axis=0)
    freq_min = freq_arr.min(axis=0)
    
    return chi_sq,prob,freq_tst_ens[1:],freq_ref[1:],freq_max[1:],freq_min[1:]

def ens_chisq_bin_specify(ensemble,tgt,bins):
    ne,n = np.shape(ensemble)
    idx = np.where((tgt>np.min(bins)) & (tgt<=np.max(bins)))[0]
    inp_ens = ensemble[:,:]
    freq_ref_tst = np.histogram(tgt[idx],bins=bins,density=False)[0]
    flat_ens = np.ndarray.flatten(inp_ens)
    flat_ens_idx = np.where((flat_ens>np.min(bins)) & (flat_ens<=np.max(bins)))[0]
    freq_tst_ens = np.histogram(flat_ens[flat_ens_idx],bins=bins,density=False)[0] / ne
    chi_sq,prob = chisq_comp(freq_ref_tst,freq_tst_ens,len(bins)-1) #don't care about the 0-lead releases
    freq_arr = np.empty((ne,len(bins)-1))
    for k in range(ne):
        ens = inp_ens[k,:]
        ens_idx = np.where((ens>np.min(bins)) & (ens<=np.max(bins)))[0]
        freq_arr[k,:] = np.histogram(ens[ens_idx],bins=bins,density=True)[0]
    freq_ens = np.histogram(flat_ens[flat_ens_idx],bins=bins,density=True)[0]
    freq_ref = np.histogram(tgt[idx],bins=bins,density=True)[0]
    freq_max = freq_arr.max(axis=0)
    freq_min = freq_arr.min(axis=0)
    freq_err = (freq_max-freq_ens,freq_ens-freq_min)
    
    return chi_sq,prob,freq_ens,freq_ref,freq_err

def rel_index(freqs):
    m = len(freqs)
    n = np.sum(freqs)
    ri = (1/n) * np.sum(np.abs(freqs - (n/(m+1))))
    return ri

def entropy(freqs):
    freqs[freqs<1] = 1
    m = len(freqs)
    n = np.sum(freqs)
    ent = ((-1)/np.log(m+1)) * np.sum((freqs/n)*np.log(freqs/n))
    return ent
        
    
def percentile_rel(ensemble,tgt,lwr_pcnt,upr_pcnt):
    ens_sort = np.sort(ensemble)
    lwr_idx = np.max([round(lwr_pcnt*len(ensemble))-1,0])
    upr_idx = round(upr_pcnt*len(ensemble))-1
    lwr_val = ens_sort[lwr_idx]
    upr_val = ens_sort[upr_idx]
    if tgt < lwr_val or tgt > upr_val:
        out = 0
    else:
        out = 1
    return lwr_val,upr_val,out

def percentile_rel_sset(ensemble,tgt,lwr_pcnt,upr_pcnt):
    ne,n = np.shape(ensemble)
    sset_out = np.empty((3,n))
    for i in range(n):
        sset_out[:,i] = percentile_rel(ensemble[:,i],tgt[i],lwr_pcnt,upr_pcnt)
        
    cp = np.sum(sset_out[2,:]) / n
    #riw = np.sum(sset_out[1,:] - sset_out[0,:]) / np.sum(sset_out[2,:]) #zha et al. (2020) 'AR-GARCh with exogenous..'
    return sset_out,cp

#Mcinerney et al. 2017
def precision_sset(ensemble,tgt):
    ne,n = np.shape(ensemble)
    prec = ((1/n) * np.sum(ensemble.std(axis=0))) / ((1/n) * np.sum(tgt))
    return prec

def vol_bias_sset(ensemble,tgt):
    ne,n = np.shape(ensemble)
    vol_bias = np.abs((np.sum(tgt) - np.sum(ensemble.mean(axis=0))) / np.sum(tgt))
    return vol_bias

def ens_cdf_pred(x,ensemble):
    ens_flat = np.ndarray.flatten(ensemble)
    ens_flat_sort = np.sort(ens_flat)
    ens_cdf = np.linspace(1,len(ens_flat_sort),len(ens_flat_sort)) / (len(ens_flat_sort)+1)
    nep = np.zeros_like(x)
    for i in range(len(x)):
        lwr_idx = np.max(np.where(ens_flat_sort<=x[i]))
        upr_idx = np.min([lwr_idx + 1,len(ens_flat_sort)-1])
        if np.shape(np.where(ens_flat_sort==x[i]))[1] > 1:
            idxs = np.where(ens_flat_sort==x[i])[0]
            idx = np.random.choice(idxs,size=1)[0]
            nep[i] = ens_cdf[idx]
        else:
            nep[i] = np.interp(x[i],(ens_flat_sort[lwr_idx],ens_flat_sort[upr_idx]),(ens_cdf[lwr_idx],ens_cdf[upr_idx]))
        
    return nep

def pit_pred(ensemble,tgt):
    n = len(tgt)
    nep = np.zeros_like(tgt)
    for i in range(len(tgt)):
        tgt_inp = np.round(tgt[i],decimals=6)
        ens_flat = ensemble[:,i]
        ens_flat_sort = np.round(np.sort(ens_flat),decimals = 6)
        ens_cdf = np.linspace(1,len(ens_flat_sort),len(ens_flat_sort)) / (len(ens_flat_sort)+1)
        if tgt_inp < ens_flat_sort[0]:
            nep[i] = ens_cdf[0]
        else:
            lwr_idx = np.max(np.where(ens_flat_sort<=tgt_inp))
            upr_idx = np.min([lwr_idx + 1,len(ens_flat_sort)-1])
            if np.shape(np.where(ens_flat_sort==tgt_inp))[1] > 1:
                idxs = np.where(ens_flat_sort==tgt_inp)[0]
                idx = np.random.choice(idxs,size=1)[0]
                nep[i] = ens_cdf[idx]
            else:
                nep[i] = np.interp(tgt_inp,(ens_flat_sort[lwr_idx],ens_flat_sort[upr_idx]),(ens_cdf[lwr_idx],ens_cdf[upr_idx]))
    
    pit_t = nep
    rnk_t = rankdata(pit_t,method='ordinal') / n
    rel = (2/n) * np.sum(np.abs(rnk_t - pit_t)) #McInerney et al., 2017
    return pit_t,rnk_t,rel

def prob_vector_calc(ensemble,threshold):
    ne,n = np.shape(ensemble)
    out_vec = np.zeros(n)
    inp_ens = ensemble[:,:]
    for i in range(n):
        inp = np.empty_like(inp_ens[:,i])
        inp[inp_ens[:,i]<=threshold] = 0
        inp[inp_ens[:,i]>threshold] = 1
        out_vec[i] = np.sum(inp) / ne
        
    return out_vec

def prob_vector_calc_cat(ensemble,cat):
    ne,n = np.shape(ensemble)
    out_vec = np.zeros(n)
    inp_ens = np.int_(ensemble[:,:])
    for i in range(n):
        inp = np.empty_like(inp_ens[:,i])
        inp[inp_ens[:,i]!=cat] = 0
        inp[inp_ens[:,i]==cat] = 1
        out_vec[i] = np.sum(inp) / ne
        
    return out_vec

def prob_vector_calc_bin(ensemble,bins):
    ne,n = np.shape(ensemble)
    out_vec = np.zeros(n)
    inp_ens = ensemble[:,:]
    for i in range(n):
        idx = np.where((inp_ens[:,i]>bins[0]) & (inp_ens[:,i]<bins[1]))[0]
        inp = np.zeros_like(inp_ens[:,i])
        inp[idx] = 1
        out_vec[i] = np.sum(inp) / ne
        
    return out_vec

def ens_uncond_prob(ensemble,threshold):
    ne,n = np.shape(ensemble)
    inp_ens = ensemble[:,:]
    inp_ens[inp_ens<=threshold] = 0
    inp_ens[inp_ens>threshold] = 1
    prob = np.sum(inp_ens) / (ne*n)
    
    return prob

def ens_uncond_prob_cat(ensemble,cat):
    ne,n = np.shape(ensemble)
    inp_ens = ensemble[:,:]
    inp_ens[inp_ens!=cat] = 0
    inp_ens[inp_ens==cat] = 1
    prob = np.sum(inp_ens) / (ne*n)
    
    return prob

def ens_uncond_prob_bin(ensemble,bins):
    ne,n = np.shape(ensemble)
    inp_ens = np.zeros_like(ensemble)
    idx = np.where((inp_ens>bins[0]) & (inp_ens<bins[1]))
    inp_ens[idx] = 1
    prob = np.sum(inp_ens) / (ne*n)
    
    return prob

def brier_score_skill(ensemble,tgt,threshold):
    events = np.empty_like(tgt)
    events[:] = tgt[:]
    events[events<=threshold] = 0
    events[events>threshold] = 1
    uc_prob = np.sum(events) / len(tgt)
    prob_vec = prob_vector_calc(ensemble, threshold)
    ref_vec = np.full(len(events),uc_prob)
    brier_score = (1/len(events)) * np.sum((prob_vec - events)**2)
    brier_score_ref = (1/len(events)) * np.sum((ref_vec - events)**2)
    
    brier_skill = 1 - (brier_score/brier_score_ref)
    
    return brier_score,brier_skill

def brier_score_intmulticat(ensemble,tgt,ncat):
    n = len(tgt)
    events = np.empty_like(tgt)
    events[:] = np.int_(tgt[:])
    #events = events[events>0]
    cats = np.linspace(1,ncat,ncat)
    bscore_arr = np.empty((ncat))
    bscore_ref_arr = np.empty((ncat))
    for k in range(ncat):
        evts = np.zeros_like(events,dtype=int)
        idx = np.where(events==cats[k])[0]
        evts[idx] = 1
        uc_prob = np.sum(events) / n
        ens_prob = prob_vector_calc_cat(ensemble,cats[k])
        bscore_arr[k] = np.sum((ens_prob - events)**2)
        bscore_ref_arr[k] = np.sum((uc_prob - events)**2)
        
    brier_score = (1/n) * np.sum(bscore_arr)
    brier_ref_score = (1/n) * np.sum(bscore_ref_arr)
    
    brier_skill = 1 - (brier_score/brier_ref_score)
    
    return brier_score,brier_skill

def brier_score_binsmulticat(ensemble,tgt,bins):
    n = len(tgt)
    nb = len(bins)-1
    events = np.empty_like(tgt)
    events[:] = np.int_(tgt[:])
    #events = events[events>0]
    bscore_arr = np.empty(nb)
    bscore_ref_arr = np.empty(nb)
    for k in range(nb):
        evts = np.zeros_like(events,dtype=int)
        idx = np.where((events>bins[k]) & (events<bins[k+1]))[0]
        evts[idx] = 1
        uc_prob = np.sum(events) / n
        ens_prob = prob_vector_calc_bin(ensemble,(bins[k],bins[k+1]))
        bscore_arr[k] = np.sum((ens_prob - events)**2)
        bscore_ref_arr[k] = np.sum((uc_prob - events)**2)
        
    brier_score = (1/n) * np.sum(bscore_arr)
    brier_ref_score = (1/n) * np.sum(bscore_ref_arr)
    
    brier_skill = 1 - (brier_score/brier_ref_score)
    
    return brier_score,brier_skill

def rel_plot_calcs(ensemble,tgt,bins,threshold):
    events = np.empty_like(tgt)
    events[:] = tgt[:]
    events[events<=threshold] = 0
    events[events>threshold] = 1
    inp_ens = np.empty_like(ensemble)
    inp_ens[:,:] = ensemble[:,:]

    prob_vec = prob_vector_calc(inp_ens, threshold)
    prob_vec_cut = pd.cut(prob_vec,bins = np.linspace(0,1,bins+1),include_lowest=True,labels=False)

    p_obs_y = np.zeros(bins)
    p_y = np.zeros(bins)

    for i in range(bins):
        idx = np.where(prob_vec_cut == i)
        if len(idx[0]) == 0:
            p_obs_y[i] = 0
            p_y[i] = 0
        else:
            p_obs_y[i] = np.sum(events[idx]) / len(idx[0])
            p_y[i] = np.mean(prob_vec[idx])

    #nzero_idx = np.where(p_y!=0)
    #p_y = p_y[nzero_idx]
    #p_obs_y = p_obs_y[nzero_idx]
    uc_prob = np.sum(events) / len(tgt)
    ens_uc_prob = ens_uncond_prob(inp_ens, threshold)
    
    return p_obs_y,p_y,uc_prob,ens_uc_prob,prob_vec

def brier_scores(ensemble,tgt,bins,threshold):
    events = np.empty_like(tgt)
    events[:] = tgt[:]
    events[events<=threshold] = 0
    events[events>threshold] = 1
    inp_ens = np.empty_like(ensemble)
    inp_ens[:,:] = ensemble[:,:]

    prob_vec = prob_vector_calc(inp_ens, threshold)
    prob_vec_cut = pd.cut(prob_vec,bins = np.linspace(0,1,bins+1),include_lowest=True,labels=False)

    p_obs_y = np.zeros(bins)
    p_y = np.zeros(bins)
    rel_term = []
    res_term = []
    uc_prob = np.sum(events) / len(tgt)

    for i in range(bins):
        idx = np.where(prob_vec_cut == i)
        if len(idx[0]) == 0:
            p_obs_y[i] = 0
            p_y[i] = 0
        else:
            p_obs_y[i] = np.sum(events[idx]) / len(idx[0])
            p_y[i] = np.mean(prob_vec[idx])
        rel_term.append(len(idx) * (p_y[i]-p_obs_y[i])**2)
        res_term.append(len(idx) * (p_obs_y[i]-uc_prob)**2)
    
    bs_rel = (1/len(tgt)) * np.sum(rel_term)
    bs_res = (1/len(tgt)) * np.sum(res_term)
    bs_unc = uc_prob*(1-uc_prob)
    #nzero_idx = np.where(p_y!=0)
    #p_y = p_y[nzero_idx]
    #p_obs_y = p_obs_y[nzero_idx]
    
    #ens_uc_prob = ens_uncond_prob(inp_ens, threshold)
    
    return bs_rel,bs_res,bs_unc

def rel_plot_calcs_cat(ensemble,tgt,bins,cat):
    events = np.empty_like(tgt)
    events[:] = tgt[:]
    events[events!=cat] = 0
    events[events==cat] = 1
    inp_ens = np.empty_like(ensemble)
    inp_ens[:,:] = ensemble[:,:]

    prob_vec = prob_vector_calc(inp_ens,cat)
    prob_vec_cut = pd.cut(prob_vec,bins = np.linspace(0,1,bins+1),include_lowest=True,labels=False)

    p_obs_y = np.zeros(bins)
    p_y = np.zeros(bins)

    for i in range(bins):
        idx = np.where(prob_vec_cut == i)
        if len(idx[0]) == 0:
            p_obs_y[i] = 0
            p_y[i] = 0
        else:
            p_obs_y[i] = np.sum(events[idx]) / len(idx[0])
            p_y[i] = np.mean(prob_vec[idx])

    uc_prob = np.sum(events) / len(tgt)
    ens_uc_prob = ens_uncond_prob_cat(inp_ens, cat)
    
    return p_obs_y,p_y,uc_prob,ens_uc_prob,prob_vec

def rel_plot_calcs_bin(ensemble,tgt,rel_bins,cat_bins):
    events = np.zeros_like(tgt)
    idx = np.where((tgt>cat_bins[0]) & (tgt<cat_bins[1]))
    events[idx] = 1
    inp_ens = np.empty_like(ensemble)
    inp_ens[:,:] = ensemble[:,:]

    prob_vec = prob_vector_calc_bin(inp_ens,cat_bins)
    prob_vec_cut = pd.cut(prob_vec,bins = np.linspace(0,1,rel_bins+1),include_lowest=True,labels=False)

    p_obs_y = np.zeros(rel_bins)
    p_y = np.zeros(rel_bins)

    for i in range(rel_bins):
        idx2 = np.where(prob_vec_cut == i)[0]
        if len(idx2) == 0:
            p_obs_y[i] = 0
            p_y[i] = 0
        else:
            p_obs_y[i] = np.sum(events[idx2]) / len(idx2)
            p_y[i] = np.mean(prob_vec[idx2])

    uc_prob = np.sum(events) / len(tgt)
    ens_uc_prob = ens_uncond_prob_bin(inp_ens, cat_bins)
    
    return p_obs_y,p_y,uc_prob,ens_uc_prob,prob_vec

def C_alpha(alpha,n):
    c_alpha = np.sqrt(-(np.log(alpha/2)/(2*n)))
    return c_alpha

    
def tau_stat(zi):
    n = len(zi)
    rnk_z = rankdata(zi,method='ordinal')
    rnk_z1 = np.zeros_like(rnk_z)
    rnk_z1[:(n-1)] = rnk_z[1:]
    kend_tau = ktau(rnk_z,rnk_z1)[0] 
    tau_stat = kend_tau * np.sqrt((9*n*(n-1))/(2*(2*n+5)))
    return kend_tau,tau_stat

def ktau_ind_test_pval(zi):
    tau = tau_stat(zi)
    p = norm.cdf(tau)
    return p

def unsort_unique_vec(x):
    x_uniq = pd.unique(x)
    x_uniq_idx = np.zeros_like(x_uniq,dtype=int)
    for i in range(len(x_uniq_idx)):
        x_uniq_idx[i] = np.where(x==x_uniq[i])[0][0]
    return x_uniq_idx

def ar1_resid(coef,const,ts):
    resids = np.zeros_like(ts)
    const_arr = np.full(len(ts),const)
    coef_arr = np.full(len(ts),coef)
    pred = (ts - const_arr) * coef_arr 
    resids[1:] = (ts[1:] - const_arr[1:]) - pred[:(len(pred)-1)] 
    return resids

def ar3_resid(coef,ts):
    resids = np.zeros_like(ts)
    #const_arr = np.full(len(ts),const)
    coef_arr1 = np.full(len(ts),coef[0])
    coef_arr2 = np.full(len(ts),coef[1])
    coef_arr3 = np.full(len(ts),coef[2])
    pred = ts[2:(len(ts)-1)] * coef_arr1[3:] + ts[1:(len(ts)-2)] * coef_arr2[3:] + ts[:(len(ts)-3)] * coef_arr3[3:]
    resids[3:] = ts[3:] - pred
    return resids

def ens_ar1_decorrelate(ensemble,coef,const):
    inp_ens = np.empty_like(ensemble)
    inp_ens[:,:] = ensemble[:,:]
    out_ens = np.empty_like(ensemble)
    ne,n = np.shape(inp_ens)
    for i in range(ne):
        resids = ar1_resid(coef,const,inp_ens[i,:])
        out_ens[i,:] = resids
    return out_ens

def ens_ar3_decorrelate(ensemble,coef):
    inp_ens = np.empty_like(ensemble)
    inp_ens[:,:] = ensemble[:,:]
    out_ens = np.empty_like(ensemble)
    ne,n = np.shape(inp_ens)
    for i in range(ne):
        resids = ar3_resid(coef,inp_ens[i,:])
        out_ens[i,:] = resids
    return out_ens

def ar3x_sim(n,ar_coef,x,x_coef,hsked):
    innov = norm.rvs(size = n+3)
    if hsked == True:
        innov[3:] = innov[3:] * x
    arx_out = np.zeros(n+3)
    arx_out[:2] = innov[:2]
    for i in range(n):
        arx_out[i+3] = ar_coef[0]*arx_out[i+2] + ar_coef[1]*arx_out[i+1] + ar_coef[2]*arx_out[i] + x_coef*x[i] + innov[i+3]
        
    return arx_out[3:]
    

#--------------------------------------------end--------------------------------------------------