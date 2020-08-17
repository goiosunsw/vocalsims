from copy import deepcopy
from reed_up_downstream_dyn import ReedSimulation, calc_fixed_point
from json_object import JSONObject
from pypevoc.PVAnalysis import PV
from pypevoc.Heterodyne import HeterodyneHarmonic
from pypevoc.SoundUtils import FuncWind
from scipy.optimize import fsolve
import numpy as np
import scipy.signal as sig
import pickle

def imp_resp(js, nfft=1024):
    sim = ReedSimulation()
    sim.from_json(js)
    impresp = sim.tracts['bore'].impulse_response(n=nfft)
    impresp_vt = sim.tracts['vocal'].impulse_response(n=nfft)
    return impresp, impresp_vt


with open('tongue_opening_simulation_with_tuning.json') as f:
    js = JSONObject(f)

dx = js['environment/acoustic/speed of sound']/js['simulation/sample rate']
step_len = js['tracts/vocal/elements/2/length'] 
n_steps = len(js['tracts/vocal/elements'] )

tongue_rad_list=[0.02,0.015,0.01,0.008,0.006,0.004,0.003,0.0015]
#tongue_rad_list=[0.02,0.0015]
len_range = 0.02
nfft_ir = 2**14

base_out_name = 'tongue_tuning_'


def run_smooth_pert(js):
    sim = ReedSimulation()
    sim.from_json(js)

    sr=sim.sr
    dp_per_samp = 1.0
    n_samp=int(js['simulation/duration']*sr)
    frac = dp_per_samp/js['environment/blowing pressure/value']

    pert_pressure = js['perturbation/blowing pressure']
    pert_ampl = js['perturbation/factor']

    sim.simulation_init(pert=False)
    sim.p_blow=0.0
    sim.pert=False
    p_blow = []

    f0=1/(sim.tracts['bore'].total_delay/sim.sr*2)

    pert_duration = 1/f0/2

    pert_started = False
    pert_finished = False

    while sim.samp_no < n_samp:
        sim.simulation_tick(reverse=False)
        cur_t = sim.samp_no/sr
        if sim.p_blow<pert_pressure:
            sim.p_blow+=(js['environment/blowing pressure/value']-sim.p_blow)*frac
        else:
            if pert_ampl and not pert_started:
                pert_started=True
                pert_samp_on = sim.samp_no
                pert_delta_a = sim.a0*(pert_ampl-1)
                pert_a_start = sim.a0
                pert_t_on = cur_t
        if pert_started and not pert_finished:
            delta_t = cur_t - pert_t_on
            sim.a0 = pert_a_start+(1-np.cos(np.pi*delta_t/pert_duration))/2*pert_delta_a
            if delta_t>pert_duration:
                pert_finished=True
                pert_samp_off = sim.samp_no
                pert_t_off = cur_t
                
        p_blow.append(sim.p_blow)
        
    sim.p_blow_vec = np.array(p_blow)
    sim.pert_t_on = pert_t_on
    sim.pert_time = pert_t_on
    sim.pert_t_off = pert_t_off
    sim.finalize()
    return sim

n_main = len(js['tracts/vocal/elements'])-3
print(n_main)

for tongue_rad in tongue_rad_list:
    with open('tongue_opening_simulation_with_tuning.json') as f:
        js = JSONObject(f)
    base_main_len = js['tracts/vocal/elements/{}/length'.format(n_main)] 
    #sim = ReedSimulation()
    #tongue_rad = js['tracts/vocal/elements/0/radius']
    main_vt_rad = js['tracts/vocal/elements/{}/radius'.format(n_main)]
    log_main_vt_rad = np.log(main_vt_rad)
    log_tongue_rad = np.log(tongue_rad)
    #cone_len = 0    

    for ii in range(0,n_steps):
        js['tracts/vocal/elements/%d/radius'%(ii)] = np.exp(log_tongue_rad + (log_main_vt_rad-log_tongue_rad)*ii/n_steps)
        #js['tracts/vocal/elements/%d/length'%(ii)] = step_len

    #js['tracts/vocal/elements/%d/length'%(ii+1)] -=cone_len
    
    for main_len in np.arange(base_main_len-len_range,base_main_len+len_range,dx):
        js['tracts/vocal/elements/{}/length'.format(n_main)]=main_len 
        impresp,impresp_vt = imp_resp(js,nfft=nfft_ir)

        sim = run_smooth_pert(js)
        
        p_b = sim.p_in + sim.p_out;
        p_vt = sim.p_vt_in + sim.p_vt_out;

        u = (sim.p_out - sim.p_in)/sim.zc_b;
        u_sg = -(sim.p_vt_out - sim.p_vt_in)/sim.zc_vt

        a = sim.a
        f0=1/(sim.tracts['bore'].total_delay/sim.sr*2)
        hhb = HeterodyneHarmonic(p_b,sr=sim.sr,nwind=1024,nhop=256,f=f0)
        hhv = HeterodyneHarmonic(p_vt,sr=sim.sr,nwind=1024,nhop=256,f=f0)
        zch_b = sim.char_impedances['bore']
        zch_vt = sim.char_impedances['vocal']

        this_dict = {'p_b':p_b,'p_vt':p_vt,'hhb':hhb,'hhv':hhv,'pert_time':sim.pert_time,'p_blow':sim.p_blow_vec,
                    'impresp_b':impresp, 'impresp_vt':impresp_vt, 'zch_b':zch_b, 'zch_vt':zch_vt,'js':js,
                     'u':u, 'a':a}
        

        outfile = base_out_name + '_{}_{}_.pickle'.format(tongue_rad, main_len)

        with open(outfile,'wb') as f:
            pickle.dump(this_dict,f)