#!/usr/bin/env python
# coding: utf-8

# In[38]:


import obspy
import glob
import os, sys
import numpy as np
from obspy import read, read_inventory
import obspy.signal
import matplotlib.pyplot as latex

import matplotlib.pyplot as plt
import scipy.signal as signal

import warnings
warnings.filterwarnings('ignore')
# latex parameter
font = {
    'family': 'serif', 
    'serif': ['Computer Modern Roman'],
    'weight' : 'regular',
    'size'   : 14
    }

plt.rc('font', **font)
plt.rc('text', usetex=True)
# plt.style.use('classic')

color_map = 'viridis'


# In[3]:


earthquakes = read('./earthquakes/*.SAC')
explosions = read('./explosions/*.SAC')


# In[4]:


print('Number of earthquakes: {}'.format(len(earthquakes)))
print('Number of Explosions: {}'.format(len(explosions)))


# In[5]:


eqntrk = []
exntrk = []
for i in range(len(earthquakes)):
    eqntrk.append(earthquakes[i].stats.network)

for i in range(len(explosions)):
    exntrk.append(explosions[i].stats.network)
    


# In[34]:


print(np.unique(eqntrk))


# In[35]:


print(np.unique(exntrk))


# In[36]:


net = exntrk + eqntrk
print(np.unique(net))


# In[46]:


eq  = read('./earthquakes/IA.BATI..SHZ.SAC')
plt.title('Sabber')
ex = read('./explosions/IA.TDNI..SHZ.SAC')


# In[48]:


eq.plot(title='Earthquake occured in Mynmer-India border in 2005, recorded in USA')
ex.plot(title='Nuclear explosions occured in North korea in 2017, recorded in USA')


# In[34]:


ex.plot()


# ## Get the statioins and theior location

# In[49]:


eqsta = []
exsta = []
for i in range(len(earthquakes)):
    eqsta.append(earthquakes[i].stats['sac']['kstnm'])

for i in range(len(explosions)):
    exsta.append(explosions[i].stats['sac']['kstnm'])
    
stations = np.unique(eqsta + exsta)


# In[50]:


station_info = []
for sta in stations:
    seq = earthquakes.select(station=str(sta))
    
    if (seq):  
        info  = {'station': sta, 'lat':seq[0].stats['sac']['stla'], 'long': seq[0].stats['sac']['stlo']}
        station_info.append(info)
    else:
        
        sex = explosions.select(station=str(sta))
        info  = {'station': sta, 'lat':sex[0].stats['sac']['stla'], 'long': sex[0].stats['sac']['stlo']}
        station_info.append(info)


# In[70]:


print(station_info[0:10])


# In[53]:


import pickle
output = open('station_info.pkl', 'wb')
pickle.dump(station_info, output)
output.close()


# ## Event Info

# In[57]:


from mpl_toolkits.basemap import Basemap
# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,            llcrnrlon=-180,urcrnrlon=180,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,60.))
m.drawmapboundary(fill_color='aqua')
plt.title("Equidistant Cylindrical Projection")
plt.show()


# In[14]:


# SACPZ.US.OXF.--.BHZ
paz_file = 'SACPZ.'+ tr.stats['sac']['knetwk']+'.'+ tr.stats['sac']['kstnm'] + '.--.'+ tr.stats['sac']['kcmpnm']
root_path = './seismogram_v2/earthquakes/2004-12-26-mw90-sumatra/'
path_name = str(root_path + paz_file)
attach_paz(tr, path_name)
tr.stats


# In[21]:


station_name = 'BATI'
steq = earthquakes.select(station= station_name)
stex = explosions.select(station= station_name)

print(stex[0].stats)

plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(steq[0].data)
plt.xlabel('Number of data points')
plt.ylabel('Counts')
plt.title('Eartqauke')

plt.subplot(122)
plt.plot(stex[0].data)
plt.xlabel('Number of data points')
plt.title('Explosion')
plt.tight_layout()
plt.savefig('raw_seismograms2.png')
plt.show()


# In[5]:


print(explosions[0].stats)


# In[6]:


stations = []
for i in range(len(st)):
    stations.append(st[i].stats['sac']['kstnm'])
stations = np.unique(stations)


# In[7]:


broads = []
for i in stations:
    ncomp = len(st.select(station=i))
    if ncomp == 3:
        broads.append(i)
#         print('Station: {} and no of comp: {}'.format(i, ncomp))


# ### show processed seismograms

# In[13]:


#get_ipython().magic(u'matplotlib tk')
def plot_seismograms(st):
    
    plt.figure(figsize=(6, 4))
    for i in range(len(st)):
        if (st[i].stats['sac']['kcmpnm'] == 'BHZ'):
            plt.subplot(311)
            plt.plot(signal.detrend(st[i].data), 'b-')
            plt.title('Vertical component')
            plt.xticks([])

        elif (st[i].stats['sac']['kcmpnm'] == 'BHE'):
            plt.subplot(312)
            plt.plot(signal.detrend(st[i].data), 'r-')
            plt.title('East-west component')
            plt.xticks([])

        elif (st[i].stats['sac']['kcmpnm'] == 'BHN'):
            plt.subplot(313)
            plt.plot(signal.detrend(st[i].data), 'k-')
            plt.title('North-south component')
            plt.xlabel('Number of data points')
    
    plt.tight_layout()
    plt.show()


# In[19]:


plot_seismograms(earthquakes[0:200])


# In[18]:


plot_seismograms(explosions[0:200])


# In[91]:


import librosa
import scipy as sp
import scipy.signal as signal
aa = librosa.feature.rmse(y=ss[0].data)


# In[95]:


ss = st.select(station=broads[100])
data = ss[0].data
sample_rate = ss[0].stats.sampling_rate
f, Pxx_den = signal.periodogram(data, sample_rate)


# In[96]:


plt.semilogy(f, Pxx_den)
plt.ylim([1e-7, 1e2])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()


# In[99]:


np.sqrt(Pxx_den.max())


# In[100]:


f, Pxx_spec = signal.periodogram(data, sample_rate, 'flattop', scaling='spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.ylim([1e-4, 1e1])
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.show()


# In[101]:


np.sqrt(Pxx_spec.max())


# In[ ]:




