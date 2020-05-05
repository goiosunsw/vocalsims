from glob import glob
import os
import sys
import pandas
import json_object as jso
import pickle
import re
import scipy.signal as sig
import pandas as pd
import seaborn as sns
import numpy as np

from pypevoc.Heterodyne import HeterodyneHarmonic
from pypevoc.PVAnalysis import PV

#datadir = '/Users/goios/Data/clarinet_simulations/20200502-2seg_vt_high_q_reed/'
datadir = sys.argv[1]
filelist = glob(os.path.join(datadir,'*.pickle'))

basedir = os.path.split(datadir)[-1]
outfile = 'simulation_analysis_'+basedir+'.csv'

params = [[float(y) for y in re.findall('([\d.]+)(?:_|\.pickle)',x)] for x in filelist]

df=pandas.DataFrame(params, columns=['tongue_rad','vt_len','p_blow'])
df['filename']=filelist

print('{} simulations found'.format(len(df)))
print()
#df.sample(3)

print('Unique blowing pressure values:')
print(df.p_blow.unique().tolist())
print('Unique blowing pressure values:')
print(df.tongue_rad.unique().tolist())
print()

# get one sample rate 
fl=df.filename

with open(fl.iloc[0],'rb') as fo:
    pckl = pickle.load(fo)
js = pckl['js']
sr = js['simulation/sample rate']

allres=[]

nzpks = 3
nharm = 3

tsust = .1
nsust_smpl = int(tsust*sr)

nzfft = 2**16

for thisfile in fl:
    with open(thisfile,'rb') as fo:
        pckl = pickle.load(fo)
    results = {'filename':thisfile}
    
    js = pckl['js']
    tongue_rad = js['tracts/vocal/elements/0/radius']
    results['tongue radius'] = tongue_rad
    
    vt_len = sum([x['length'] for x in js['tracts/vocal/elements'].to_python() if x['type']=='cylinder'])
    results['vocal tract length']=vt_len
    results['blowing pressure']=js['perturbation/blowing pressure']
    
    sr = js['simulation/sample rate']
    impresp_b = pckl['impresp_b']
    impresp_vt = pckl['impresp_vt']
    f = np.arange(nzfft)/nzfft*sr

    rf=np.fft.fft(impresp_vt,n=nzfft)
    z_vt=(1+rf)/(1-rf)*pckl['zch_vt']


    rf=np.fft.fft(impresp_b,n=nzfft)
    z_b=(1+rf)/(1-rf)*pckl['zch_b']
    
    z_b_pks,_ = sig.find_peaks(np.abs(z_b))
    for ii,zz in enumerate(z_b_pks[:nzpks]):
        results['z_bore_pk_f_{}'.format(ii)] = f[zz]
        results['z_bore_pk_z_{}'.format(ii)] = z_b[zz]
    z_v_pks,_ = sig.find_peaks(np.abs(z_vt))
    for ii,zz in enumerate(z_v_pks[:nzpks]):
        results['z_vocal_pk_f_{}'.format(ii)] = f[zz]
        results['z_vocal_pk_z_{}'.format(ii)] = z_vt[zz]
        

    #ax[0].semilogy(f,np.abs(z_vt),label='{:.3f}'.format(vt_len))
    #ax[1].plot(f,np.angle(z_vt),label='{:.3f}'.format(vt_len))

    pb = pckl['p_b']
    pv = pckl['p_vt']

    bore_len = sum([x['length'] for x in js['tracts/bore/elements'].to_python() if x['type']=='cylinder'])
    c = js['environment/acoustic/speed of sound']
    f0=(c/bore_len/4)
    results['lf0']=f0
    results['bore length']=bore_len


    pvoc = PV(pb,sr=sr,progress=False)
    pvoc.run_pv()
    amax = max(pvoc.fundamental_magnitude)

    a_trans_end = amax*.9
    a_trans_start = amax*.01
    i_trans_end = np.flatnonzero(pvoc.fundamental_magnitude>a_trans_end)[0]-1
    i_trans_start = np.flatnonzero(pvoc.fundamental_magnitude[:i_trans_end]<a_trans_start)[-1]
    i_trans = slice(i_trans_start,i_trans_end)
    atrans = 20*np.log10(pvoc.fundamental_magnitude[i_trans])
    ttrans = pvoc.t[i_trans]
    polytrans = np.polyfit(ttrans,atrans,1) 
#    print(polytrans)
#     fig,ax = subplots(1)
#     ax.plot(pvoc.t,20*np.log10(pvoc.fundamental_magnitude))
#     ax.plot(pvoc.t[i_trans],np.polyval(polytrans,pvoc.t[i_trans]))
#     ax.axvspan(pvoc.t[i_trans_start],pvoc.t[i_trans_end],color='red',alpha=.2)
    
    isus = pvoc.t>(pvoc.t[-1]-tsust)
    fsus = pvoc.fundamental_frequency[isus]
    #print(np.mean(fsus),np.std(fsus))
    asus = pvoc.fundamental_magnitude[isus]
    #print(np.mean(asus),np.std(asus),np.polyfit(pvoc.t[isus],asus,1))
    fsusav = np.mean(fsus)
    #print('sust','f0',fsusav,np.std(fsus))
    #print('trns','f0',np.mean(pvoc.fundamental_frequency[i_trans]),np.std(pvoc.fundamental_frequency[i_trans]))
    results['sus_f0_avg']=fsusav
    results['sus_f0_std']=np.std(fsus)
    results['trans_f0_avg']=np.mean(pvoc.fundamental_frequency[i_trans])
    results['trans_f0_std']=np.std(pvoc.fundamental_frequency[i_trans])


    hhb = HeterodyneHarmonic(pb,sr=sr,nwind=1024*2,nhop=256*2,f=fsusav)
    hhv = HeterodyneHarmonic(pv,sr=sr,nwind=1024*2,nhop=256*2,f=fsusav)

#     for ii in range(3):
#         ax.plot(hhb.t,20*np.log10(np.abs(hhb.camp[:,ii])))
#     for ii in range(3):
#         ax.plot(hhv.t,20*np.log10(np.abs(hhv.camp[:,ii])),ls='--')

    t_trans_start = pvoc.t[i_trans_start]
    t_trans_end = pvoc.t[i_trans_end]


    hidxsus = hhb.t>(hhb.t[-1]-tsust)
    hidxtrans = (hhb.t>=t_trans_start) & (hhb.t<=t_trans_end)
    hi_trans_start = np.flatnonzero(hhb.t>=t_trans_start)[0]
    hi_trans_end = np.flatnonzero(hhb.t<=t_trans_end)[-1]

    for ii in range(3):
        #print('harm ',ii+1)
        for hlab,hh in [('bore',hhb),('vt',hhv)]:
            #print(' '*2,hlab)
            v = np.abs(hh.camp[:,ii])
            amax = max(v)
            a_trans_end = amax*.9
            a_trans_start = amax*.01
            try:
                i_trans_end = np.flatnonzero(v>a_trans_end)[0]-1
            except IndexError:
                i_trans_end = hi_trans_end
            try:
                i_trans_start = np.flatnonzero(v[:i_trans_end]<a_trans_start)[-1]
            except IndexError:
                i_trans_start = hi_trans_start
            #print(' '*4,'trans times',hh.t[i_trans_start],hh.t[i_trans_end])
            results['{}_{}_trans_start'.format(hlab,ii)]=hh.t[i_trans_start]
            results['{}_{}_trans_end'.format(hlab,ii)]=hh.t[i_trans_end]

            hidxtrans = slice(i_trans_start,i_trans_end)

            for rlab, reg in [('sus',hidxsus),('trans',hidxtrans)]:

                v = hh.camp[reg,ii]
                #print(' '*4,rlab,'abs',np.mean(np.abs(v)),np.std(np.abs(v)))
                #print(' '*4,rlab,'arg',np.mean(np.unwrap(np.angle(v))),np.std(np.unwrap(np.angle(v))))
                results['{}_{}_{}_abs_avg'.format(hlab,ii+1,rlab)] = np.mean(np.abs(v))
                results['{}_{}_{}_abs_std'.format(hlab,ii+1,rlab)] = np.std(np.abs(v))
                results['{}_{}_{}_arg_avg'.format(hlab,ii+1,rlab)] = np.mean(np.unwrap(np.angle(v)))
                results['{}_{}_{}_arg_std'.format(hlab,ii+1,rlab)] = np.std(np.unwrap(np.angle(v)))

                v = hhv.camp[reg,ii]/hhb.camp[reg,ii]
                #print(' '*4,rlab,'rat','abs',np.mean(np.abs(v)),np.std(np.abs(v)))
                #print(' '*4,rlab,'rat','arg',np.mean(np.unwrap(np.angle(v))),np.std(np.unwrap(np.angle(v))))
                results['{}_{}_{}_arg_avg'.format('hrat',ii+1,rlab)] = np.mean(np.unwrap(np.angle(v)))
                results['{}_{}_{}_arg_std'.format('hrat',ii+1,rlab)] = np.std(np.unwrap(np.angle(v)))
                results['{}_{}_{}_abs_avg'.format('hrat',ii+1,rlab)] = np.mean(np.abs(v))
                results['{}_{}_{}_abs_std'.format('hrat',ii+1,rlab)] = np.std(np.abs(v))

            v = hh.camp[reg,ii]
            try:
                trpol = np.polyfit(hh.t[reg],20*np.log10(np.abs(v)),1)
                #print(' '*4,rlab,'exr',trpol[0])
                results['{}_{}_{}'.format('rate',ii+1,rlab)] = trpol[0]
                results['{}_{}_{}_resid'.format('rate',ii+1,rlab)] = np.std(20*np.log10(np.abs(v))-np.polyval(trpol,hh.t[reg]))

            except TypeError:
                trpol = np.nan

    allres.append(results)
    
resdf = pd.DataFrame(allres)
resdf.to_csv(outfile)
