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
__simulation_version__ = "20200304"
__parameter_version__ = "20200304"



def td_to_tf_tract(ta):
    r,l=ta.to_physical_dimensions()
    taz = psyn.Duct()
    for rr, ll in zip(r,l):
        taz.append_element(psyn.StraightDuct(length=ll,radius=rr,loss_multiplier=10))
    taz.set_termination(psyn.FlangedPiston(radius=rr))

    return taz


class ReedSimulation(object):
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

    def from_json(self, json):
        self.json = json
        self.desc = json['description']

        jenv = json['environment']

        self.p_blow = jenv['blowing pressure']
        self.k = jenv['reed']['stiffness']
        self.a0 = jenv['reed']['rest opening']

        jac = jenv['acoustic']
        self.c = jac['speed of sound']
        self.rho = jac['density']
        self.mu = jac['viscosity']
        self.sqrt_two_on_rho = np.sqrt(2/self.rho)

        jsim = json['simulation']
        self.sr = jsim['sample rate']
        self.t_max = jsim['duration']
        self.callback_every = jsim['callback every']

        # turn on/off subglottal tract
        self.vt_on=jenv['vocal tract enabled']

        # turbulence
        self.add_noise=jenv['noise']['enabled']
        self.turb_scale = jenv['noise']['scale']

        self.freq_dep_losses = jenv['frequency dependent losses']

        ## vocal tract
        jvt = json['tracts']['vocal']
        vt, radii = self.tract_from_json(jvt)
        self.set_tract('vocal',vt)

        # Subglottal tract
        jsb = json['tracts']['bore']
        bore, bore_rad = self.tract_from_json(jsb)
        self.set_tract('bore',bore)

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
            g.attrs['param_json'] = (json.dumps(self.json))
            g.attrs['code_version'] = __simulation_version__
            g.attrs['code type'] = __simulation_type__

            # allocate output vectors
            # g = f[timestamp]
            gg = g.create_group('simulation')
            print('Initialising output HDF5 file')
            gg.attrs['blowing_pressure']=self.p_blow
            gg.attrs['vocal_tract_on']=self.vt_on

            gg.attrs['turbulence_on']=self.add_noise
            gg.attrs['turbulence_scale']=self.turb_scale
            gg.attrs['solution_epsilon']=self.sol_eps

            # simulation variables
            gg.create_dataset('p_out', (self.n_samp,))
            gg.create_dataset('p_in', (self.n_samp,))
            gg.create_dataset('p_vt_out', (self.n_samp,))
            gg.create_dataset('p_vt_in', (self.n_samp,))
            gg.create_dataset('p_rad_out', (self.n_samp,))
            gg.create_dataset('p_rad_in', (self.n_samp,))
            print('Init''d HDF5 vectors with {} samples'.format(self.n_samp))

    def init_solution_vectors(self):
        self.p_vt_out = (np.zeros(self.n_samp));
        self.p_vt_in = (np.zeros(self.n_samp));
        self.p_out = (np.zeros(self.n_samp));
        self.p_in = (np.zeros(self.n_samp));
        self.p_rad_out = (np.zeros(self.n_samp));
        self.p_rad_in = (np.zeros(self.n_samp));
        self.u_rad = (np.zeros(self.n_samp));
        try:
            self.init_hdf5()
        except AttributeError:
            pass

    def simulation_init(self):
        self.samp_no = 0
        bore_tract = self.tracts['bore']
        self.zc_b = self.char_impedances['bore']
        bore_tract.reset()

        try:
            vocal_tract = self.tracts['vocal']
            vocal_tract.reset()
            self.zc_vt = self.char_impedances['vocal']
            self.zc_vt_eff = self.zc_vt
            zc = self.zc_vt + self.zc_vt_eff
        except KeyError:
            self.vt_on = False
            self.zc_vt = 0
            zc = self.zc_b
        self.zeta_mul = self.rho/2/self.zc_vt**2;

        aglot = self.aglot
        self.zc = zc

        # parameters for numerical resolution
        sol = np.ones((1,2));

        self.u_prev = 0.
        #self.init_hdf5()

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

    def func(self, u, args):
        p_blow, p_in = args
        p = self.zc_b*u + 2*p_in
        a = -(p_blow - p)/self.k
        deltap = p_blow - p
        return u - (self.a0 - a) * self.sqrt_two_on_rho * np.sqrt(deltap) * np.sign(deltap)

    def jac(self, u,args):
        p_blow, p_in = args
        p = self.zc_b*u + 2*p_in
        dp = self.zc_b
        a = -(p_blow - p)/self.k
        da = 1./self.k
        deltap = p_blow-p
        ddeltap = -self.zc_b
        return 1 + da * self.sqrt_two_on_rho * np.sqrt(deltap) * np.sign(deltap) + \
                (self.a0-a) * self.sqrt_two_on_rho * np.sign(deltap) * .5*deltap**(-.5)

    def simulation_tick(self):
        samp_no = self.samp_no
        ta = self.tracts['bore']

        # pressures are read one sample before because new sample has not been inserted
        p_b_in_cur = ta.read_next_in_without_tick()
        if self.vt_on:
            vta = self.tracts['vocal']
            p_vt_in_cur = vta.read_next_in_without_tick()
        else:
            p_vt_in_cur = 0
        
        p_in = - p_vt_in_cur + p_b_in_cur
        
        #zeta = self.zeta_mul / a**2;
        #u_guess = self.u_prev
        deltap_guess = self.p_blow - 2*p_in 
        u_guess = np.sqrt(np.abs(2*deltap_guess/self.rho))*(self.a0-deltap_guess/self.k)
        u = fsolve(self.func, u_guess, args = [self.p_blow, p_in])
                    #fprime = self.jac)
        
        if self.add_noise:
            turb = np.random.randn()*u*self.turb_scale
        else:
            turb = 0
        
        po = self.zc_b*(u+turb) + p_b_in_cur;
        po_vt = -self.zc_vt_eff*(u-turb) + p_b_in_cur;
            
            
        
        self.u_prev = u
        _=ta.tick(po)
        pllost=vta.tick(po_vt)
        
        self.p_out[samp_no] = po
        self.p_vt_out[samp_no] = po_vt
        self.p_in[samp_no] = p_b_in_cur
        self.p_vt_in[samp_no] = p_vt_in_cur
        
        self.p_rad_in[samp_no] = ta.end_in_last
        self.p_rad_out[samp_no] = ta.end_out_last
        #u_mth[samp_no] = p_mth_in[samp_no] - p_mth_out[samp_no]


        self.samp_no += 1

    def hdf5_callback(self,from_idx,to_idx):
        print("Callback at {}".format(to_idx))
        with h5py.File(self.hdf5_file, 'a') as f:
            g = f[self.hdf5_path]
            gg = g['simulation']
            idx = slice(from_idx,to_idx)
            gg['p_vt_out'][idx] = self.p_vt_out[idx]
            gg['p_vt_in'][idx] = self.p_vt_in[idx]
            gg['p_out'][idx] = self.p_out[idx]
            gg['p_in'][idx] = self.p_in[idx]
            gg['p_rad_out'][idx] = self.p_rad_out[idx]
            gg['p_rad_in'][idx] = self.p_rad_in[idx]

            
class NoInteractionReedSimulation(ReedSimulation):
    def simulation_tick(self):
        samp_no = self.samp_no
        u0 = np.sqrt(2*self.p_lung/self.rho)
        a = self.aglot[samp_no];
        u = a*u0
        ta = self.tracts['vocal']

        # pressures are read one sample before because new sample has not been inserted
        p_vt_in_cur = ta.read_next_in_without_tick()

        
        
        if self.add_noise:
            turb = np.random.randn()*u*self.turb_scale
        else:
            turb = 0
        
        po_vt = self.zc_vt*(u+turb) + p_vt_in_cur;
            
        
        self.u_prev = u
        _=ta.tick(po_vt)
        pmth_out = ta.end_out_last
        pmth_in = ta.end_in_last
        
        self.p_vt_out[samp_no] = po_vt
        self.p_vt_in[samp_no] = p_vt_in_cur
        
        self.p_mth_in[samp_no] = ta.end_in_last
        self.p_mth_out[samp_no] = ta.end_out_last
        #u_mth[samp_no] = p_mth_in[samp_no] - p_mth_out[samp_no]
        
        self.samp_no += 1

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

    sim = ReedSimulation()
    sim.set_hdf5_path(output,timestamp)
    sim.read_json_conifg(infile)
    sim.simulate()


    if args.plot:
        from matplotlib.pyplot import plot, show
        p_b = sim.p_in + sim.p_out;
        p_vt = sim.p_vt_in + sim.p_vt_out;

        u = (sim.p_out - sim.p_in)/sim.zc_b;
        u_sg = -(sim.p_vt_out - sim.p_vt_in)/sim.zc_vt

        plot(u)
        show()

