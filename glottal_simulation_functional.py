import os
import numpy as np
import h5py
from datetime import datetime
import json 
import argparse

from scipy.optimize import fsolve

from pypevoc import PV
import pympedance.Synthesiser as psyn
import TimeDomainTubes as tdt

from tqdm import trange as trange

__simulation_type__ = "time-domain filter-losses 1st-order reflection filter (object version)"
__simulation_version__ = "20191128"
__parameter_version__ = "20191128"



def td_to_tf_tract(ta):
    r,l=ta.to_physical_dimensions()
    taz = psyn.Duct()
    for rr, ll in zip(r,l):
        taz.append_element(psyn.StraightDuct(length=ll,radius=rr,loss_multiplier=10))
    taz.set_termination(psyn.FlangedPiston(radius=rr))

    return taz


class VocalSimulation(object):
    """
    Class for a single simulation
    """   
    def __init__(self):
        self.tracts={}
        self.char_impedances = {}
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.aglot = np.zeros(0)
        self.last_callback = 0 
        self.samp_no = 0
        self.callback_every = -1
        self.sol_eps = 1e-10;
        self.init_time = datetime.now().strftime('%Y%m%d-%H%M%S')

    def reset(self):
        self.samp_no = 0
        self.last_callback = 0
        for tract in self.tracts.values():
            tract.reset() 

    def set_tract(self, id, tract):
        radii = tract.radii
        zc = self.rho*self.c/radii[0]**2/np.pi
        self.tracts[id] = tract
        self.char_impedances[id] = zc
        self.tract_data_to_hdf5(id)

    def tract_from_json(self, jd):
        lengths = []
        radii = []
        if self.freq_dep_losses:
            ta = tdt.RealTimeDuct(speed_of_sound=self.c,sr=self.sr)
        else:
            ta = tdt.ConstFreqLossDuct(speed_of_sound=self.c,sr=self.sr)
        for el in jd['elements']:
            if el['type'] == 'cylinder':
                lengths.append(el['length'])
                radii.append(el['radius'])
                ta.append_tube(length=el['length'],radius=el['radius'])
            elif el['type'] == 'termination':
                pass
        ta.adjust_termination()
        return ta, radii

    def from_json(self, jstr):
        jj = jstr
        self.json = jj
        self.desc = jj['description']

        jenv = jj['environment']
        print(json.dumps(jenv,indent=2))

        self.p_lung = jenv['lung pressure']

        jac = jenv['acoustic']
        self.c = jac['speed of sound']
        self.rho = jac['density']
        self.mu = jac['viscosity']

        jsim = jj['simulation']
        self.sr = jsim['sample rate']
        self.t_max = jsim['duration']
        self.callback_every = jsim['callback every']

        jf0 = jj['glottis']['f0']
        f0bits = []
        try:
            jfb = jf0[0]
            prev_f = jfb['f']
            prev_t = jfb['t']
            for jfb in jf0[1:]:
                f0bits.append(np.linspace(prev_f,jfb['f'],int((jfb['t']-prev_t)*sr)))
            
            self.fvec = np.concatenate(f0bits)
        except TypeError:
            self.fvec = np.ones(int(self.t_max*self.sr))*jf0
            
        # turn on/off subglottal tract
        self.sg_on=jenv['subglottal tract enabled']

        # turbulence
        self.add_noise=jenv['noise']['enabled']
        self.turb_scale = jenv['noise']['scale']

        self.freq_dep_losses = jenv['frequency dependent losses']

        ## vocal tract
        jvt = jj['supraglottal']
        vt, radii = self.tract_from_json(jvt)
        self.set_tract('vocal',vt)

        # Subglottal tract
        jsg = jj['subglottal']
        sg, sg_rad = self.tract_from_json(jsg)
        self.set_tract('subglottal',sg)

        # Glottal area
        jj = self.json
        jg = jj['glottis']
        self.set_glottis_lf(jg)

        self.n_samp = int(self.t_max*self.sr)
        self.init_solution_vectors()


    def read_json_conifg(self, json_file):
        with open(json_file) as f:
            jj = json.load(f)
        self.from_json(jj)

    def impulse_response(self, id, n_ir=16384):
        tract = self.tracts[id]
        tforesp, tfiresp, impresp = tract.transfer_impulse_response(n_ir)
        print("{} impulse response sum:".format(id),np.sum(np.array(impresp)**2))
        return tforesp, tfiresp, impresp

    def set_hdf5_path(self, hdf5_file, group_path):
        self.hdf5_file = hdf5_file
        self.hdf5_path = group_path
        with h5py.File(hdf5_file, 'a') as f:
            g = f.create_group(self.hdf5_path)
       

    def tract_data_to_hdf5(self, id):
        tforesp, tfiresp, impresp = self.impulse_response(id)
        with h5py.File(self.hdf5_file, 'a') as f:
            g = f[self.hdf5_path]
            gg = g.create_group('tract'+id)
            gg.create_dataset('impulse_response',data=impresp)
            gg.create_dataset('transfer_response_outgoing',data=tforesp)
            gg.create_dataset('transfer_response_ingoing',data=tfiresp)


    def init_hdf5(self):
        hdf5_file = self.hdf5_file
        ## Write preliminary data
        with h5py.File(hdf5_file, 'a') as f:
            f.attrs['last_simulation_created'] = self.init_time
            #g = f.create_group(self.hdf5_path)
            g = f[self.hdf5_path]
            g.attrs['start_time'] = self.timestamp
            g.attrs['param_json'] = json.dumps(self.json)
            g.attrs['code_version'] = __simulation_version__
            g.attrs['code type'] = __simulation_type__

            # allocate output vectors
            # g = f[timestamp]
            gg = g.create_group('simulation')
            print('Initialising output HDF5 file')
            gg.attrs['lung_pressure']=self.p_lung
            gg.attrs['subglottal_on']=self.sg_on

            gg.attrs['turbulence_on']=self.add_noise
            gg.attrs['turbulence_scale']=self.turb_scale
            gg.attrs['solution_epsilon']=self.sol_eps

            # simulation variables
            gg.create_dataset('p_sg_out', (self.n_samp,))
            gg.create_dataset('p_sg_in', (self.n_samp,))
            gg.create_dataset('p_vt_out', (self.n_samp,))
            gg.create_dataset('p_vt_in', (self.n_samp,))
            gg.create_dataset('p_mth_out', (self.n_samp,))
            gg.create_dataset('p_mth_in', (self.n_samp,))
            print('Init''d HDF5 vectors with {} samples'.format(self.n_samp))


    def set_glottis_lf(self, jg):
        from lf_model import LFModel

        lf = LFModel()
        f0 = jg['f0']
        t0 = 1.

        jlf = jg['lf coefficients']
        rg = jlf['rg']/100
        tp = t0/2/rg
        re = jlf['re']/100
        te = re*t0
        oq = jlf['oq']/100
        tc = oq*t0
        ra = jlf['ra']/100
        ta_glot = ra*t0
        print(tp,te,tc,ta_glot)
        lf.set_timing_parameters(ta=ta_glot,te=te,tc=1.,tp=tp)

        lf_max = lf(lf.tp)

        lrect = jg['rectangular slot']
        a_max = jg['maximum aperture']

        glot_area = lambda x: lf(x)/lf_max*a_max 
        phvec = np.cumsum(self.fvec/self.sr)
        self.aglot = glot_area(phvec)

        # Hanson and Stevens 1995
        self.lg = lrect['length']
        self.dg = np.mean(self.aglot)/self.lg
        self.hg = lrect['depth']
        self.Krho = lrect['krho']

    def glottis_data_to_hdf5(self):
        ## Write glottis data
        with h5py.File(self.hdf5_file, 'a') as f:
            g = f[self.hdf5_path]
            gg = g.create_group('glottal_area')
            gg.create_dataset('sample',data=self.aglot)

    def init_solution_vectors(self):
        self.p_sg_out = (np.zeros(self.n_samp));
        self.p_sg_in = (np.zeros(self.n_samp));
        self.p_vt_out = (np.zeros(self.n_samp));
        self.p_vt_in = (np.zeros(self.n_samp));
        self.p_mth_out = (np.zeros(self.n_samp));
        self.p_mth_in = (np.zeros(self.n_samp));
        self.u_mth = (np.zeros(self.n_samp));
        try:
            self.init_hdf5()
        except AttributeError:
            pass


    def simulation_init(self):
        self.samp_no = 0
        vocal_tract = self.tracts['vocal']
        self.zc_vt = self.char_impedances['vocal']
        self.zeta_mul = self.rho/2/self.zc_vt**2;
        vocal_tract.reset()

        try:
            subglottal_tract = self.tracts['subglottal']
            subglottal_tract.reset()
            self.zc_sg = subglottal_tract.radii[0]**2*np.pi
            self.zc_sg_eff = self.zc_sg
            zc = self.zc_vt + self.zc_sg_eff
        except KeyError:
            self.sg_on = False
            self.zc_sg_eff = 0
            zc = self.zc_vt

        aglot = self.aglot
        self.zc = zc

        # parameters for numerical resolution
        sol = np.ones((1,2));

        self.u_prev = np.sqrt(np.abs(self.p_lung*2*aglot[0]**2/self.rho))

    def simulate(self):
        self.simulation_init()
        while self.samp_no < self.n_samp:
            self.simulation_tick()
            if self.callback_every > 0:
                if (self.samp_no >= self.last_callback + self.callback_every):
                    self.hdf5_callback(self.last_callback, self.samp_no)
                    self.last_callback = self.samp_no

        self.hdf5_finish()
        
    def hdf5_finish(self):
        self.hdf5_callback(self.last_callback, self.samp_no)
        with h5py.File(self.hdf5_file, 'a') as f:
            g = f[self.hdf5_path]
            g.attrs['end_time'] = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.glottis_data_to_hdf5()

    def func(self, x, args):
        p_lung, p_in, a = args
        return p_lung - 2 * p_in - (self.zc) * x - self.rho / 2 / a**2 * x**2

    def jac(self, x,args):
        p_lung, p_in, a = args
        return - (self.zc) - self.rho / 2 / a**2 * x * 2

    def simulation_tick(self):
        samp_no = self.samp_no
        a = self.aglot[samp_no];
        ta = self.tracts['vocal']

        # pressures are read one sample before because new sample has not been inserted
        p_vt_in_cur = ta.read_next_in_without_tick()
        if self.sg_on:
            sga = self.tracts['subglottal']
            p_sg_in_cur = sga.read_next_in_without_tick()
        else:
            p_sg_in_cur = 0
        
        p_in = - p_sg_in_cur + p_vt_in_cur
        
        if a > self.sol_eps:
            zeta = self.zeta_mul / a**2;
            #u_guess = self.u_prev
            u_guess = np.sqrt(np.abs(self.p_lung*2*a**2/self.rho))
            u = fsolve(self.func, u_guess, args = [self.p_lung, p_in, a],
                       fprime = self.jac)
            
            if self.add_noise:
                turb = np.random.randn()*u*self.turb_scale
            else:
                turb = 0
            
            po_vt = self.zc_vt*(u+turb) + p_vt_in_cur;
            po_sg = -self.zc_sg_eff*(u-turb) + p_sg_in_cur;
            
            
        else:
            # closed glottis, flow=0
            po_vt = 0 + p_vt_in_cur;
            po_sg = 0 + p_sg_in_cur;
            u=0
        
        self.u_prev = u
        _=ta.tick(po_vt)
        pmth_out = ta.end_out_last
        pmth_in = ta.end_in_last
        pllost=sga.tick(po_sg)
        
        self.p_sg_out[samp_no] = po_sg
        self.p_vt_out[samp_no] = po_vt
        self.p_sg_in[samp_no] = p_sg_in_cur
        self.p_vt_in[samp_no] = p_vt_in_cur
        
        self.p_mth_in[samp_no] = ta.end_in_last
        self.p_mth_out[samp_no] = ta.end_out_last
        #u_mth[samp_no] = p_mth_in[samp_no] - p_mth_out[samp_no]


        self.samp_no += 1

    def hdf5_callback(self,from_idx,to_idx):
        print("Callback at {}".format(to_idx))
        with h5py.File(self.hdf5_file, 'a') as f:
            g = f[self.hdf5_path]
            gg = g['simulation']
            idx = slice(from_idx,to_idx)
            gg['p_sg_out'][idx] = self.p_sg_out[idx]
            gg['p_sg_in'][idx] = self.p_sg_in[idx]
            gg['p_vt_out'][idx] = self.p_vt_out[idx]
            gg['p_vt_in'][idx] = self.p_vt_in[idx]
            gg['p_mth_out'][idx] = self.p_mth_out[idx]
            gg['p_mth_in'][idx] = self.p_mth_in[idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('param_file',help='input file with parameters in json format')
    parser.add_argument('-p','--plot',action='store_true', help='produce plot when finished')
    parser.add_argument('-o','--output',help='output HDF5 file')
    args = parser.parse_args()

    infile = args.param_file

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    if not args.output:
        args.output = timestamp+'.hdf5'

    output = args.output

    sim = VocalSimulation()
    sim.set_hdf5_path(output,timestamp)
    sim.read_json_conifg(infile)
    sim.simulate()


    if args.plot:
        from matplotlib.pyplot import plot, show
        p_sg = sim.p_sg_in + sim.p_sg_out;
        p_vt = sim.p_vt_in + sim.p_vt_out;

        u = (sim.p_vt_out - sim.p_vt_in)/sim.zc_vt;
        u_sg = -(sim.p_sg_out - sim.p_sg_in)/sim.zc_sg

        plot(u)
        show()

