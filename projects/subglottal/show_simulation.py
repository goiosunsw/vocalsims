import os
import sys
import numpy as np
import matplotlib.pyplot as pl
import pandas
import sounddevice as sd
from pypevoc import PV
import pympedance.Synthesiser as psyn
import TimeDomainTubes as tdt


import h5py
import json

filename = sys.argv[1]

def dump_attr(name,obj):
    def print_attr(xx):
        for k,v in xx.attrs.items():
            print("{:>30} : {}".format(k,v))
    print(name)
    try:
        print_attr(obj)
    except AttributeError:
        pass
    print()

def play_array(x,sr):
    sd.play(x/np.max(np.abs(x)), sr)

def press(event):
    if event.key == ' ':
        play_array(u_mth,sr)

fig,ax=pl.subplots(4,sharex=True)

with h5py.File(filename, 'r') as f:

    f.visititems(dump_attr)


    for name,g in f.items():
        try:
            gg = g['simulation']
        except KeyError:
            print('{}: no data found'.format(name))
            continue
        js = json.loads(g.attrs['param_json'])
        sr = js['simulation']['sample rate']
        p_lung = js['environment']['lung pressure']
        rho = js['environment']['acoustic']['density']
        c = js['environment']['acoustic']['speed of sound']
        aglot = g['glottal_area']['sample']
        p_sg_in = gg['p_sg_in'][:]
        p_sg_out = gg['p_sg_out'][:]
        p_vt_in = gg['p_vt_in'][:]
        p_vt_out = gg['p_vt_out'][:]
        p_mth_in = gg['p_mth_in'][:]
        p_mth_out = gg['p_mth_out'][:]

        p_sg = p_sg_in + p_sg_out
        p_vt = p_vt_in + p_vt_out

        t = ((np.arange(len(p_sg))/sr));



        u_mth = (p_mth_out - p_mth_in)#/zc_mth
        u = p_vt_out - p_vt_in
        u_sg = p_sg_out - p_sg_in


        ax[0].plot(aglot)
        ax[1].plot(p_vt)
        #ax[1].plot(p_lung+p_sg)
        #ax[2].plot(aglot*np.sqrt(2*p_lung/rho))
        ax[2].plot(u)
        #ax[2].plot(u_sg)
        #ax[2].plot(np.sqrt(np.abs(2*(p_lung+p_sg-p_vt)/rho))*aglot*np.sign(p_lung+p_sg-p_vt))
        ax[3].plot(u_mth)
fig.canvas.mpl_connect('key_press_event', press)
#ax[2].plot(u_mth)
pl.show()