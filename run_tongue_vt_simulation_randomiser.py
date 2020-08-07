import sys
import random
import types
import collections
import six
from datetime import datetime

from copy import deepcopy
from reed_up_downstream_dyn import ReedSimulation
from json_object import JSONObject
from scipy.optimize import fsolve
import numpy as np
import scipy.signal as sig
import pickle
import multiprocessing

def json_path_get(js, addr):
    ret = js
    for kk in addr.split('/'):
        try:
            kk=int(kk)
        except ValueError:
            pass
        ret = ret[kk]
    return ret

def is_iterable_but_not_string(arg):
    return (
        isinstance(arg, collections.Iterable) 
        and not isinstance(arg, six.string_types)
    )

def iternodes(js):
    for k,v in js.items():
        yield k,v
        try:
            yield from iternodes(v)
        except AttributeError:
            pass
   
# def iterable_but_not_string(obj):
#     try:
#         obj.__iter__
#         if not isinstance(obj, (str, bytes, unicode)):
#             return True
#         print('Here')
#     finally:
#         return False

def list_dict_items(obj):
    try:
        for k, v in obj.items():
            yield k, v
    except AttributeError:
        for k, v in enumerate(obj):
            yield k, v
            
def iternodes(js, root=""):
    for k,v in list_dict_items(js):
        if root:
            fullv = root+'/'+str(k)
        else:
            fullv = str(k)
        yield fullv,v
        try:
            if is_iterable_but_not_string(v):
                yield from iternodes(v,root=fullv)
        except TypeError:
            pass

def iternodes_replacer(js, root=""):
    if root=="":
        iternodes_replacer.js = js
    for k,v in list_dict_items(js):
        if root:
            fullv = root+'/'+str(k)
        else:
            fullv = str(k)
            
        yield fullv,v

        if is_iterable_but_not_string(v):
            try:
                v.keys()
            except AttributeError:
                pass
            else:    
                if '_range' in v.keys():
                    vr = v['_range']
                    v['_value'] = random.uniform(min(vr),max(vr))
                elif '_choice' in v.keys():
                    vl = v['_choice']
                    v['_value'] = random.choice(vl)
                elif '_link' in v.keys():
                    v['_value'] = json_path_get(iternodes_replacer.js,v['_link'])
            finally:
                yield from iternodes_replacer(v,root=fullv)
                

def apply_underscores(js, root=""):
    #print("!!"+str(js)[:20])
    if root=="":
        apply_underscores.js = js
        
    if root.split('/')[-1] == '_link':
        vv = json_path_get(apply_underscores.js,js)
        #print(vv,root)
        yield from apply_underscores(vv, root=root[:root.find('/_link')])
        #print("!H")
    else:
        try:
            if '_link' in js.keys():
                pass
            else:
                yield root,js

        except AttributeError:
            yield root,js
        
    if is_iterable_but_not_string(js):
        for k,v in list_dict_items(js):
            if root:
                fullv = root+'/'+str(k)
            else:
                fullv = str(k)

            if is_iterable_but_not_string(v):
                try:
                    v.keys()
                except AttributeError:
                    pass
                else:    
                    if '_range' in v.keys():
                        vr = v['_range']
                        v = random.uniform(min(vr),max(vr))
                        
                    elif '_choice' in v.keys():
                        vl = v['_choice']
                        v = random.choice(vl)
            yield from apply_underscores(v,root=fullv)

def imp_resp(js, nfft=1024):
    sim = ReedSimulation()
    sim.from_json(js)
    impresp = sim.tracts['bore'].impulse_response(n=nfft)
    impresp_vt = sim.tracts['vocal'].impulse_response(n=nfft)
    return impresp, impresp_vt


#jsfile = 'tongue_2seg_vt_open_simulation_with_tuning.json'
jsfile = sys.argv[1]

with open(jsfile) as f:
    js = JSONObject(f)

dx = js['environment/acoustic/speed of sound']/js['simulation/sample rate']
n_main = 1

#tongue_rad_list=[0.02,0.015,0.01,0.008,0.006,0.004,0.003,0.0015]
#tongue_rad_list=[0.015,0.008,0.005,0.003]
tongue_rad_vt_len_dict = {0.02: 0.09705464909865272,
 0.015: 0.1326027397260274,
 0.01: 0.16329209263348388,
 0.008: 0.17943091668228756,
 0.006: 0.18477824876547813,
 0.004: 0.1893602987979393,
 0.003: 0.19988585823287583,
 0.0015: 0.19178082191780824}

pblow_mult_list = [.9,1.0,1.05,1.1,1.2]
pblow_traget_mul = 1.1

len_range = 0.03
nfft_ir = 2**14

base_out_name = 'tongue_vt_open_tuning'


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

def js_generator():
    
    while True:
        with open(jsfile) as f:
            js0 = JSONObject(f)

        jsn = JSONObject()
        for k,v in apply_underscores(js0.to_python()):
            if not is_iterable_but_not_string(v):
                jsn[k]=v

        impresp,impresp_vt = imp_resp(jsn,nfft=nfft_ir)
        yield jsn, impresp, impresp_vt

def md_worker(args):

    js, impresp, impresp_vt = args

    tongue_rad = js['tracts/vocal/elements/0/radius']
    main_len = js['tracts/vocal/elements/{}/length'.format(n_main)] 
    pblow = js['perturbation/blowing pressure'] 
    print("tongue radius {}, vt length = {}, pblow = {}".format(tongue_rad,main_len,pblow))


    sim = run_smooth_pert(js)
    
    p_b = sim.p_in + sim.p_out;
    p_vt = sim.p_vt_in + sim.p_vt_out;

    u = (sim.p_out - sim.p_in)/sim.zc_b;
    u_sg = -(sim.p_vt_out - sim.p_vt_in)/sim.zc_vt

    a = sim.a
    f0=1/(sim.tracts['bore'].total_delay/sim.sr*2)
    #hhb = HeterodyneHarmonic(p_b,sr=sim.sr,nwind=1024,nhop=256,f=f0)
    #hhv = HeterodyneHarmonic(p_vt,sr=sim.sr,nwind=1024,nhop=256,f=f0)
    zch_b = sim.char_impedances['bore']
    zch_vt = sim.char_impedances['vocal']

    this_dict = {'p_b':p_b,'p_vt':p_vt,#'hhb':hhb,'hhv':hhv,
                 'pert_time':sim.pert_time,'p_blow':sim.p_blow_vec,
                 'impresp_b':impresp, 'impresp_vt':impresp_vt, 'zch_b':zch_b, 'zch_vt':zch_vt,'js':js,
                    'u':u, 'a':a}
    
    outfile = base_out_name + '_{}.pickle'.format(datetime.strftime(datetime.now(),'%Y%M%d_%H%M'))

    with open(outfile,'wb') as f:
        pickle.dump(this_dict,f)

        
p = multiprocessing.Pool(8)
jsg = js_generator()
p.map(md_worker,jsg)
