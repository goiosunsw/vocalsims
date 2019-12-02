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

with h5py.File(filename, 'r') as f:
    for k in f.keys():
        timestamp = k
    g = f[timestamp]
    c = g.attrs['speed_of_sound']
    try:
        rho = g.attrs['air_density']
    except KeyError:
        rho=1.2
    sr = g.attrs['sampling_rate']
    gg = g['simulation']

    p_sg_out = gg['p_sg_out'][:]
    p_sg_in  = gg['p_sg_in'][:]
    p_vt_out = gg['p_vt_out'][:]
    p_vt_in  = gg['p_vt_in'][:]
    p_mth_out= gg['p_mth_out'][:]
    p_mth_in = gg['p_mth_in'][:]

    p_lung = gg.attrs['lung_pressure']
    
    gg = g['vt']
    vt_rad = gg['radii']
    vt_len = gg['lengths']
    vt_ir = gg['impulse_response'][:]
    zc_vt = rho*c/vt_rad[0]**2/np.pi
    zc_mth = rho*c/vt_rad[-1]**2/np.pi
    
    gg = g['sg']
    sg_rad = gg['radii']
    sg_len = gg['lengths']
    sg_ir = gg['impulse_response'][:]
    zc_sg = rho*c/sg_rad[0]**2/np.pi
    
    gg = g['glottal_area']
    aglot = gg['sample'][:]

    f.visititems(dump_attr)

rf_vt = np.fft.fft(vt_ir)
z_vt = (1+rf_vt)/(1-rf_vt)
ff_vt = np.arange(len(z_vt))/len(z_vt)*sr

rf_sg = np.fft.fft(sg_ir)
z_sg = (1+rf_sg)/(1-rf_sg)
ff_sg = np.arange(len(z_sg))/len(z_sg)*sr

fig,ax = pl.subplots(2,sharex=True)
ax[0].plot(ff_vt,20*np.log10(np.abs(z_vt)))
ax[1].plot(ff_vt,np.angle(z_vt))
ax[0].plot(ff_sg,20*np.log10(np.abs(z_sg)))
ax[1].plot(ff_sg,np.angle(z_sg))

p_sg = p_sg_in + p_sg_out;
p_vt = p_vt_in + p_vt_out;

u = (p_vt_out - p_vt_in)/zc_vt;
u_sg = -(p_sg_out - p_sg_in)/zc_sg

t = ((np.arange(len(p_sg))/sr));



u_mth = (p_mth_out - p_mth_in)/zc_mth

def play_array(x,sr):
    sd.play(x/np.max(np.abs(x)), sr)

def press(event):
    if event.key == ' ':
        play_array(u_mth,sr)


fig,ax=pl.subplots(4,sharex=True)
ax[0].plot(aglot)
ax[1].plot(p_vt)
ax[1].plot(p_lung+p_sg)
ax[2].plot(aglot*np.sqrt(2*p_lung/rho))
ax[2].plot(u)
ax[2].plot(u_sg)
ax[2].plot(np.sqrt(np.abs(2*(p_lung+p_sg-p_vt)/rho))*aglot*np.sign(p_lung+p_sg-p_vt))
ax[3].plot(u_mth)
fig.canvas.mpl_connect('key_press_event', press)
#ax[2].plot(u_mth)
pl.show()