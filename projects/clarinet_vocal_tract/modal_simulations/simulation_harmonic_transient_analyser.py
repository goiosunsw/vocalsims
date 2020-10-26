import os
import numpy as np
from glob import glob
from pypevoc.Heterodyne import heterodyne 
from pypevoc.PVAnalysis import PV
import timeseries as ts
import scipy.signal as sig
from sklearn.linear_model import RANSACRegressor
import traceback

# number of periods in heterodyne analysis
nper=3

def heterodyne_corr(x,sr,f,maxwind=2**14,nhop=2**10,nper=3,dc_cut=50):
    xx = x.copy()
    t = np.arange(len(x))/sr
    nharm = len(f)
    ret = []
    part = np.zeros((len(x),nharm))
    for ii,ff in enumerate(f):
        #nwind = int(min(maxwind,sr/ff*nper))
        if ff==0.:
            nwind=maxwind
        foth = np.delete(f,ii)
        nwind = (sr/np.min(np.abs(foth-ff))*nper)
        print(nwind)
        hetsig = np.exp(1j*2*np.pi*ff*t)
        cc,ih = heterodyne(xx,hetsig,wind=np.hanning(nwind),hop=nhop)
        if ff==0.:
            cc/=2
        th=ih/sr
        thists = ts.SampledTimeSeries(cc,th,label='%.2f'%ff)
        thists.f = ff
        ret.append(thists)
        hf = np.interp(t,th,cc)
        xp = np.real(np.conjugate(hf)*hetsig)
        xx-=xp
        part[:,ii]=xp
    return ret,xx,part

def nearestpow2(x):
    return 2**(np.round(np.log2(x)))



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
   
    # pick frequencies from periodogram in order to extract non-harmonic components 
    freq_resolution = 25
    pk_prominence = 20
    fw,w = sig.welch(pb,fs=sr,nperseg=nearestpow2(sr/freq_resolution))
    pks,_ = sig.find_peaks(10*np.log10(w),prominence=pk_prominence)
    ff = fw[pks]
    ff = ff[(ff<2000)&(ff>100)]
    nw = int(sr/np.min(ff)*6)
    ff = np.insert(ff,0,0)

    # identify the frequency of interest
    mainidx = np.argmin(np.abs(np.array(ff)-f0))
    subharmonics = ff[(ff<ff[mainidx])&(ff>0)]
    
    # harmonic analysis
    try:
        nwind = (int(sr/f0*3))
    except ValueError:
        nwind = 1024


    ha = {}
    fa = {}
    for lab in ('b', 'vt', 'm'):
        try:
            p = data['p_{}'.format(lab)]
        except KeyError:
            continue
        
        h_array_cplx, resid, partials = heterodyne_corr(p,sr,ff,maxwind=nw,nhop=128,nper=nper)
        h_array = []
        f_array = []
        for h in h_array_cplx:
            habs = h.apply(np.abs)
            habs.f = h.f
            h_array.append(habs)
            #partial frequencies 
            fc=(h.apply(np.angle).apply(np.unwrap).diff()/2/np.pi).apply(lambda x: h.f-x)
            f_array.append(fc)

        ha[lab] = h_array
        fa[lab] = f_array
        n_non = 1
        for ii, (hts,pts) in enumerate(zip(h_array,f_array)):
            f = hts.f
            frat = f/f0
            hno = np.round(frat)
            fdev = np.abs(frat-hno)
            if fdev<0.05:   
                label = lab+str(int(hno))
                res['f_'+str(int(hno))] = f
            else:
                label = lab+'n%d'%n_non
                res['f_'+'n%d'%n_non] = f
                n_non += 1

            hts.label = label
            pts.label = label

            res['{}_t_min'.format(label)]=hts.min_time()
            res['{}_t_max'.format(label)]=hts.max_time()
            res['{}_abs_min'.format(label)]=hts.min()
            res['{}_abs_max'.format(label)]=hts.max()
            res['{}_abs_fin'.format(label)]=hts.v[-1]
            if data['pert_time']>1e-4:
                res['{}_abs_pert'.format(label)]=hts[data['pert_time']]
            else:
                res['{}_abs_pert'.format(label)]=np.nan    

    hts_array = ha['b']
    vts_array = ha['vt']

    # transient detection using one envelope
    ats = hts_array[mainidx]

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
            res['{}_abs_sus_var'.format(hts.label)]=np.diff(hts.percentile([25,75],from_time=fts.t[-1]-tsust))[0]
            res['{}_abs_trans'.format(hts.label)]=hts.percentile(50,from_time=t_trans_start, to_time=t_trans_end)
            res['{}_abs_trans_var'.format(hts.label)]=np.diff(hts.percentile([25,75],from_time=t_trans_start, to_time=t_trans_end))[0]

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

    # harmonic descriptor extraction
    for lab, f_array in fa.items():
        for ii, pts in enumerate(f_array):
            res['f{}_abs_sus'.format(pts.label)]=pts.percentile(50,from_time=fts.t[-1]-tsust)
            res['f{}_abs_sus_var'.format(pts.label)]=np.diff(pts.percentile([25,75],from_time=fts.t[-1]-tsust))[0]
            res['f{}_abs_trans'.format(pts.label)]=pts.percentile(50,from_time=t_trans_start, to_time=t_trans_end)
            res['f{}_abs_trans_var'.format(pts.label)]=np.diff(pts.percentile([25,75],from_time=t_trans_start, to_time=t_trans_end))[0]

    # harmonic ratio descriptors
    for ii, hts in enumerate(hts_array):
        vts = vts_array[ii]
        rts = vts.apply(db)-hts.apply(db)
        rts.label = 'hrat{}'.format(hts.label[1:])
        res['{}_abs_sus'.format(rts.label)]=rts.percentile(50,from_time=fts.t[-1]-tsust)
        res['{}_abs_sus_var'.format(rts.label)]=np.diff(rts.percentile([25,75],from_time=fts.t[-1]-tsust))[0]
        res['{}_abs_trans'.format(rts.label)]=rts.percentile(50,from_time=t_trans_start, to_time=t_trans_end)
        res['{}_abs_trans_var'.format(rts.label)]=np.diff(rts.percentile([25,75],from_time=t_trans_start, to_time=t_trans_end))[0]
        res['{}_abs_fin'.format(rts.label)]=rts.v[-1]
        if data['pert_time']>1e-4:
            res['{}_abs_pert'.format(rts.label)]=rts[data['pert_time']]
        else:
            res['{}_abs_pert'.format(rts.label)]=np.nan    

    # transient rate and jump
    for h_array in (hts_array,vts_array):
        for ii, hts in enumerate(h_array):
            label = hts.label
            hbl1=hts.apply(db)
            vals=hbl1.percentile([5,99])
            try:
                ted=hbl1.crossing_times(vals[1]-6)[0][0]
            except IndexError:
                print(f'No transient found in {label}')
                continue
            try:
                tst=hbl1.crossing_times(vals[1]-46,to_time=ted)[0][-1]
            except IndexError:
                tst = hbl1.min_time(to_time=ted)
            x,y=hbl1.times_values_in_range(tst,ted)
            X = np.array([x]).T
            try:
                rm=RANSACRegressor().fit(X,y)
            except ValueError:
                print(f'Error in exponential estimation in {label}')
                traceback.print_exc()
                continue
            res[f'{label}_rate'] = rm.estimator_.coef_[0]
            xpred,ytrue = hbl1.times_values_in_range(res['pert_time']-.05,ted)
            Xpred = np.array([xpred]).T
            try:
                ypred = rm.predict(Xpred)
            except ValueError:
                print(f'Error in jump prediction in {label}')
                print(res['pert_time'], ted, xpred)
                traceback.print_exc()
                continue
            res[f'{label}_jump'] = np.max(ypred-ytrue)
            res[f'{label}_t_before_jump'] = xpred[np.argmax(ypred-ytrue)]
    
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

