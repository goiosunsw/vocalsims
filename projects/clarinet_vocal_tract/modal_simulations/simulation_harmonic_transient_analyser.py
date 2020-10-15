import os
import numpy as np
from glob import glob
from pypevoc.Heterodyne import HeterodyneHarmonic
from pypevoc.PVAnalysis import PV
import timeseries as ts
import scipy.signal as sig

def db(x):
    return 20*np.log10(x)

def do_analysis(data, t_init=[0.01,0.05], t_fin=0.1, tsust=0.1,impedance=True):
    # extract main parameters
    res = {}
    sr = data['js']['simulation']['sample rate']
    pb = data['p_b']
    pvt = data['p_vt']
    t = np.arange(len(pb))/sr
    if data['pert_time']>0:
        res['pert_time']=data['pert_time']
    else:
        res['pert_time'] = 0
        

    pbinit = pb[int(t_init[0]*sr):int(t_init[1]*sr)]
    res['initial amplitude'] = np.max(pbinit)-np.min(pbinit)
    pbfin = pb[-int(t_fin*sr):]
    res['final amplitude'] = np.max(pbfin)-np.min(pbfin) 

    # fundamental frequency for harmonic analysis
    pvpb = PV(pb,sr=sr,progress=False)
    pvpb.run_pv()
    fts = ts.SampledTimeSeries(pvpb.fundamental_frequency, pvpb.t)

    f0 = fts.percentile(50,from_time=fts.t[-1]-0.2)
    if f0<20:
        f0 = np.median(fts.v[fts.v>0])
    res['f0'] = f0
    fw,pbw = sig.welch(pb,nperseg=2**10,nfft=2**12,fs=sr)

    # harmonic analysis
    try:
        nwind = (int(sr/f0*3))
    except ValueError:
        nwind = 1024


    ha = {}
    for lab in ('b', 'vt', 'm'):
        try:
            p = data['p_{}'.format(lab)]
        except KeyError:
            continue
        
        h = HeterodyneHarmonic(p,sr=sr,tf=[0,max(pvpb.t)],f=[f0,f0],nwind=nwind,nhop=128)
        h_array = [ts.SampledTimeSeries(np.abs(h.camp[:,ii]),h.t,label='h{}{}'.format(lab,ii+1)) for ii in range(h.camp.shape[1])]
        ha[lab] = h_array
        for ii, hts in enumerate(h_array):
            res['{}_t_min'.format(hts.label)]=hts.min_time()
            res['{}_t_max'.format(hts.label)]=hts.max_time()
            res['{}_abs_min'.format(hts.label)]=hts.min()
            res['{}_abs_max'.format(hts.label)]=hts.max()
            res['{}_abs_fin'.format(hts.label)]=hts.v[-1]
            if data['pert_time']>1e-4:
                res['{}_abs_pert'.format(hts.label)]=hts[data['pert_time']]
            else:
                res['{}_abs_pert'.format(hts.label)]=np.nan    

    hts_array = ha['b']
    vts_array = ha['vt']

    # transient detection using one envelope
    ats = hts_array[0]

    amax = ats.max()

    a_trans_end = amax*.7
    a_trans_start = amax*.001
    try:
        t_trans_end = ats.crossing_times(a_trans_end)[0][0]
    except IndexError:
        t_trans_end = ats.v[-1]
    try:
        t_trans_start = ats.crossing_times(a_trans_start,to_time=t_trans_end)[0][0]
    except IndexError:
        t_trans_start = t_trans_end - ats.dt
        
    atrans, ttrans = ats.apply(db).times_values_in_range(from_time=t_trans_start,to_time=t_trans_end)

    # frequency comparison
    fsusav = fts.mean(from_time=fts.t[-1]-tsust)
    res['t_trans_start']=t_trans_start
    res['t_trans_end']=t_trans_end
    res['sus_f0_avg']=fts.mean(from_time=fts.t[-1]-tsust)
    res['sus_f0_std']=fts.std(from_time=fts.t[-1]-tsust)
    res['trans_f0_avg']=fts.mean(from_time=t_trans_start, to_time=t_trans_end)
    res['trans_f0_std']=fts.std(from_time=t_trans_start, to_time=t_trans_end)

    # harmonic descriptor extraction
    for h_array in (hts_array,vts_array):
        for ii, hts in enumerate(h_array):
            res['{}_abs_sus'.format(hts.label)]=hts.percentile(50,from_time=fts.t[-1]-tsust)
            res['{}_abs_sus_var'.format(hts.label)]=np.diff(hts.percentile([75,25],from_time=fts.t[-1]-tsust))[0]
            res['{}_abs_trans'.format(hts.label)]=hts.percentile(50,from_time=t_trans_start, to_time=t_trans_end)
            res['{}_abs_trans_var'.format(hts.label)]=np.diff(hts.percentile([75,25],from_time=t_trans_start, to_time=t_trans_end))[0]

        # growth rate via derivative
        dts = hts.apply(db).diff()
        dts.label = hts.label
        for pct in [25,50,75]:
            res['{}_trans_rate_pct{}'.format(dts.label,pct)]=dts.percentile(pct,from_time=t_trans_start, to_time=t_trans_end)

        res['{}_abs_fin'.format(hts.label)]=hts.v[-1]
        if data['pert_time']>1e-4:
            res['{}_abs_pert'.format(hts.label)]=hts[data['pert_time']]
        else:
            res['{}_abs_pert'.format(hts.label)]=np.nan    

    # harmonic ratio descriptors
    for ii, hts in enumerate(hts_array):
        vts = vts_array[ii]
        rts = vts.apply(db)-hts.apply(db)
        rts.label = 'hrat{}'.format(ii+1)
        res['{}_abs_sus'.format(rts.label)]=rts.percentile(50,from_time=fts.t[-1]-tsust)
        res['{}_abs_sus_var'.format(rts.label)]=np.diff(rts.percentile([75,25],from_time=fts.t[-1]-tsust))[0]
        res['{}_abs_trans'.format(rts.label)]=rts.percentile(50,from_time=t_trans_start, to_time=t_trans_end)
        res['{}_abs_trans_var'.format(rts.label)]=np.diff(rts.percentile([75,25],from_time=t_trans_start, to_time=t_trans_end))[0]
        res['{}_abs_fin'.format(rts.label)]=rts.v[-1]
        if data['pert_time']>1e-4:
            res['{}_abs_pert'.format(rts.label)]=rts[data['pert_time']]
        else:
            res['{}_abs_pert'.format(rts.label)]=np.nan    
    
    # impedance peaks
    if impedance:
    
        for lab, iresp in (('zv', data['impresp_vt']), ('zb', data['impresp_b'])):
            rb = np.fft.fft(iresp)
            rb=rb[:len(rb)//2]
            zb = (1+rb)/(1-rb)
            zs = ts.SampledTimeSeries(np.abs(zb),dt=data['js']['simulation']['sample rate']/len(zb))
            lzs = zs.apply(db)
            pk = lzs.peaks(twind=50)[0]
            for ii in range(5):
                res[lab+'_'+str(ii)+'_f'] = pk[ii]
                res[lab+'_'+str(ii)+'_z'] = lzs[pk[ii]]


    return res

