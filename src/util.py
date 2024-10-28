import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from numba import jit

cfs_to_afd = 2.29568411*10**-5 * 86400
afd_to_cfs = 1 / cfs_to_afd

def water_day(d, is_leap_year):
    # Convert the date to day of the year
    day_of_year = d.timetuple().tm_yday
    
    # For leap years, adjust the day_of_year for dates after Feb 28
    if is_leap_year and day_of_year > 59:
        day_of_year -= 1  # Correcting the logic by subtracting 1 instead of adding
    
    # Calculate water day
    if day_of_year >= 274:
        # Dates on or after October 1
        dowy = day_of_year - 274
    else:
        # Dates before October 1
        dowy = day_of_year + 91  # Adjusting to ensure correct offset
    
    return dowy

def cdec_build_url(station=None, sensor=None, duration=None, sd=None, ed=None):
  url = 'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?'
  url += 'Stations=%s' % station
  url += '&SensorNums=%d' % sensor
  url += '&dur_code=%s' % duration
  url += '&Start=%s' % sd
  url += '&End=%s' % ed
  return url

# takes df from one (station, sensor) request
# converts to a series indexed by datetime
def cdec_reformat_series(df):
  try:
    # reindex by datetime
    df['DATE TIME'] = pd.to_datetime(df['DATE TIME'])
    df.set_index('DATE TIME', inplace=True)
    df.index.rename('datetime', inplace=True)
    # keep just the "VALUE" column and rename it
    name = '%s_%s_%s' % (df['STATION_ID'][0], df['SENSOR_TYPE'][0], df['UNITS'][0])
    df = df['VALUE']
    df.rename(name, inplace=True)
  except IndexError: #empty data frame causes indexerror
    raise IndexError('Requested data does not exist')
  return df

# gets data from a specific station and sensor type
def cdec_sensor_data(station=None, sensor=None, duration=None, sd=None, ed=None):
  url = cdec_build_url(station, sensor, duration, sd, ed)
  df = pd.read_csv(url)
  series = cdec_reformat_series(df)
  return series

# pull numpy arrays from dataframes
def get_simulation_data(res_keys, pump_keys, df_hydrology, medians, df_demand=None, init_storage=False):
  dowy = df_hydrology.dowy.values
  Q = df_hydrology[[k+'_inflow_cfs' for k in res_keys]].values
  Q_avg = medians[[k+'_inflow_cfs' for k in res_keys]].values
  R_avg = medians[[k+'_outflow_cfs' for k in res_keys]].values
  S_avg = medians[[k+'_storage_af' for k in res_keys]].values 
  Gains_avg = medians['delta_gains_cfs'].values 
  Pump_pct_avg = medians[[k+'_pumping_pct' for k in pump_keys]].values
  Pump_cfs_avg = medians[[k+'_pumping_cfs' for k in pump_keys]].values

  if df_demand is None:
    demand_multiplier = np.ones(dowy.size)
  else:
    demand_multiplier = df_demand.combined_demand.values

  if not init_storage:
    S0 = S_avg[0,:]
  else:
    S0 = df_hydrology[[k+'_storage_af' for k in res_keys]].values[0,:]

  return (dowy, Q, Q_avg, R_avg, S_avg, Gains_avg, Pump_pct_avg, Pump_cfs_avg, demand_multiplier, S0)

# calculate annual objectives from simulation results
def results_to_annual_objectives(df, medians, nodes, rk, df_demand=None):
  
  # 1. water supply reliability north of delta
  NOD_target = pd.Series([np.max((-1*medians.delta_gains_cfs[i], 0.0)) for i in df.dowy], index=df.index)
  NOD_delivery = (-1*df.delta_gains_cfs).clip(lower=0, upper=10**10)
  if df_demand is not None: # account for demand multipliers in future scenarios
    NOD_target *= df_demand.combined_demand
  rel_N = NOD_delivery.resample('AS-OCT').sum() / NOD_target.resample('AS-OCT').sum()
  rel_N[rel_N > 1] = 1
  
  # 2. water supply reliability south of delta
  SOD_target = pd.Series([medians.total_delta_pumping_cfs[i] for i in df.dowy], index=df.index)
  if df_demand is not None:
    SOD_target *= df_demand.combined_demand
  rel_S = df.total_delta_pumping_cfs.resample('AS-OCT').sum() / SOD_target.resample('AS-OCT').sum()
  rel_S[rel_S > 1] = 1

  # 3. flood volume exceeding downstream levee capacity, sum over all reservoirs
  flood_vol_taf = pd.Series(0, index=df.index)
  for k in rk:
    flood_vol_taf += ((df[k+'_outflow_cfs'] - nodes[k]['safe_release_cfs'])
                      .clip(lower=0) * cfs_to_afd / 1000)
  flood_vol_taf = flood_vol_taf.resample('AS-OCT').sum()
  
  # 4. peak flow into the delta
  delta_peak_cfs = df.delta_inflow_cfs.resample('AS-OCT').max()

  objs = pd.concat([rel_N, rel_S, flood_vol_taf, delta_peak_cfs], axis=1)
  objs.columns = ['Rel_NOD_%', 'Rel_SOD_%', 'Upstream_Flood_Volume_taf', 'Delta_Peak_Inflow_cfs']
  return objs


# convert results to dataframe
def results_to_df(results, index, res_keys):
  
  df = pd.DataFrame(index=index)
  df['dowy'] = np.array([water_day(d) for d in df.index.dayofyear])

  R,S,Delta,tocs = results

  for i,r in enumerate(res_keys):
    df[r+'_outflow_cfs'] = R[:,i]
    df[r+'_storage_af'] = S[:,i]
    df[r+'_tocs_fraction'] = tocs[:,i]

  delta_keys = ['delta_gains_cfs', 'delta_inflow_cfs', 'HRO_pumping_cfs', 'TRP_pumping_cfs', 'delta_outflow_cfs']
  for i,k in enumerate(delta_keys):
    df[k] = Delta[:,i]
  df['total_delta_pumping_cfs'] = df.HRO_pumping_cfs + df.TRP_pumping_cfs

  return df




