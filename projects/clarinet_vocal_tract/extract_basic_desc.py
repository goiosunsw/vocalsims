import os
import sys
import traceback
from glob import glob
import numpy as np
import pandas as pd
from pypevoc.Heterodyne import HeterodyneHarmonic
from pypevoc.PVAnalysis import PV
import scipy.signal as sig

# times for initial and final analysis
# initial : interval
t_init = [0.01,0.05]
# final time from end (till end)
t_fin = 0.1
nharm=3

datadir = sys.argv[1]

files = glob(os.path.join(datadir, '*.pickle'))

def nextpow2(ii):
    return 2**(np.ceil(np.log2(ii)))


allres = []

def descriptors(data):
    sr = data['js']['simulation']['sample rate']

    pb = data['p_b']
    t = np.arange(len(pb))/sr

    # check for perturbation
    if data['pert_time']>0:
        res['pert_time']=data['pert_time']
    else:
        res['pert_time'] = 0
        
    # fundamental frequency from bore signal
    pvpb = PV(pb,sr=sr,progress=False)
    pvpb.run_pv()
    f0v = pvpb.fundamental_frequency
    f0 = np.median(f0v[pvpb.t>max(t)-0.2])
    if f0<20:
        # in case the oscillation stops
        f0 = np.median(f0v[f0v>0])
    res['f0'] = f0

    # frequency from max power
    fw,pbw = sig.welch(pb,nperseg=2**10,nfft=2**12,fs=sr)
    res['f0welch'] = fw[np.argmax(pbw)]

    for p_lab in ('p_b', 'p_vt'):
        p = data[p_lab]
        # amplitude from times signal
        pinit = p[int(t_init[0]*sr):int(t_init[1]*sr)]
        res['{} initial amplitude'.format(p_lab)] = np.max(pinit)-np.min(pinit)
        pfin = p[-int(t_fin*sr):]
        res['{} final amplitude'.format(p_lab)] = np.max(pfin)-np.min(pfin)
        
        # harmonics
        try:
            nwind = min(int(sr/f0*3),1024)
        except ValueError:
            nwind = 1024

        hh = HeterodyneHarmonic(p,sr=sr,tf=[0,max(pvpb.t)],f=[f0,f0],nwind=nwind,nhop=128)

        if res['pert_time']:
            ipert = np.argmin(np.abs(hh.t-data['pert_time']))
        else:
            ipert = -1

        for i in range(nharm):
            imin = np.argmin(np.abs(hh.camp[:,i]))
            imax = np.argmax(np.abs(hh.camp[:,i]))
            res['{} h{} t_min'.format(p_lab, i+1)]=hh.t[imin]
            res['{} h{} t_max'.format(p_lab, i+1)]=hh.t[imax]
            res['{} h{} abs_min'.format(p_lab, i+1)]=np.abs(hh.camp[imin,i])
            res['{} h{} abs_max'.format(p_lab, i+1)]=np.abs(hh.camp[imax,i])
            res['{} h{} abs_fin'.format(p_lab, i+1)]=np.abs(hh.camp[-1,i])
            if ipert>-1:
                res['{} h{}_abs_pert'.format(p_lab, i+1)]=np.abs(hh.camp[ipert,i])
            else:
                res['{} h{}_abs_pert'.format(p_lab, i+1)]=np.nan
    return res

for f in files:
    
    print("Processing {}".format(f))
    res = {}
    try:
        data = np.load(f,allow_pickle=True)
        res = descriptors(data)
        res['error'] = False
    except Exception as e:
        res = {'error': True}
        traceback.print_exc()

    res['filename']=f
    allres.append(res)

df = pd.DataFrame(allres)
df.to_pickle(os.path.join(datadir,'basic_desc.pickle'))