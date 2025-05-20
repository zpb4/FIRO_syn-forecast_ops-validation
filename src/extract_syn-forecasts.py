import sys
import os
sys.path.insert(0, os.path.abspath('./src'))
import numpy as np
import syn_util
from datetime import datetime

now=datetime.now()
print('extract start',now.strftime("%H:%M:%S"))

sd = '1985-10-15' 
ed = '2019-08-15'

loc = 'YRS'
site = 'NBBC1'
vers_out = '5fold'

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

Q,Qf,dowy,tocs_inp,df_idx = syn_util.extract(sd,ed,forecast_type='syn',syn_sample='Synth1',Rsyn_path=syn_path1,syn_vers=syn_vers1,forecast_param=syn_vers1_param,loc=loc,site=site,opt_pcnt='',obj_pwr='',opt_strat='',gen_setup=syn_vers1_setup)

Qf_v1_array=np.empty((nsamps,np.shape(Qf)[0],np.shape(Qf)[1],np.shape(Qf)[2]))
Qf_MSG_v1_array=np.empty((nsamps,np.shape(Qf)[0],np.shape(Qf)[1],np.shape(Qf)[2]))

Qf_v2_array=np.empty((nsamps,np.shape(Qf)[0],np.shape(Qf)[1],np.shape(Qf)[2]))
Qf_MSG_v2_array=np.empty((nsamps,np.shape(Qf)[0],np.shape(Qf)[1],np.shape(Qf)[2]))

for i in range(nsamps):
    syn_samp = 'Synth'+str(i+1) #if using a 'syn' sample, this specifies which one to use
    Q,Qf,dowy,tocs_inp,df_idx = syn_util.extract(sd,ed,forecast_type='syn',syn_sample=syn_samp,Rsyn_path=syn_path1,syn_vers=syn_vers1,forecast_param=syn_vers1_param,loc=loc,site=site,opt_pcnt='',obj_pwr='',opt_strat='',gen_setup=syn_vers1_setup)
    Qf_v1_array[i,:,:,:] = Qf
    
    Q,Qf,dowy,tocs_inp,df_idx = syn_util.extract(sd,ed,forecast_type='syn',syn_sample=syn_samp,Rsyn_path=syn_path2,syn_vers=syn_vers2,forecast_param='a',loc=loc,site=site,opt_pcnt=syn_vers2_pct,obj_pwr=syn_vers2_objpwr,opt_strat=syn_vers2_optstrat,gen_setup=syn_vers2_setup)
    Qf_v2_array[i,:,:,:] = Qf
    print(i)

np.savez_compressed('data/%s/%s/%s-%s_Qf_syn-forecast%s_nsamp=%s.npz' %(loc,site,loc,site,syn_vers1+vers_out,nsamps), arr=Qf_v1_array)
np.savez_compressed('data/%s/%s/%s-%s_Qf_syn-forecast%s_nsamp=%s.npz' %(loc,site,loc,site,syn_vers2+vers_out,nsamps), arr=Qf_v2_array)


now=datetime.now()
print('extract end',now.strftime("%H:%M:%S"))


#----------------------------------------end--------------------------------------------