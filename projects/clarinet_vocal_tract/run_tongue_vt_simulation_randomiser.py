import sys
import random
import types
import collections
import six
import traceback
import json
import pickle
from datetime import datetime

from copy import deepcopy
from vocalsims.reed_up_downstream_dyn import ReedSimulation
from simulation_harmonic_transient_analyser import do_analysis
from vocalsims.interfacing.json_object import JSONObject
from scipy.optimize import fsolve
import numpy as np
import scipy.signal as sig
import pickle
import multiprocessing

nfft_ir = 2**14

base_out_name = 'tongue_vt_open_tuning'
output_data=False
sidx=0

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



def run_smooth_pert(js):
    sim = ReedSimulation()
    sim.from_json(js)
    sim.set_probe('vocal', -1, 0, label='m')

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
    pert_t_on = -1.
    pert_t_off = -1.

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

        for probe in sim.probes:
            sim.update_probe(probe)
                
        p_blow.append(sim.p_blow)
        
    sim.p_blow_vec = np.array(p_blow)
    sim.pert_t_on = pert_t_on
    sim.pert_time = pert_t_on
    sim.pert_t_off = pert_t_off
    sim.finalize()
    return sim

def js_generator(js0):
    
    while True:

        jsn = JSONObject()
        for k,v in apply_underscores(js0.to_python()):
            if not is_iterable_but_not_string(v):
                jsn[k]=v

        yield jsn

from time import sleep

def collect_time_domain(sim):
    p_b = sim.p_in + sim.p_out;
    p_vt = sim.p_vt_in + sim.p_vt_out;

    u = (sim.p_out - sim.p_in)/sim.zc_b;
    u_sg = -(sim.p_vt_out - sim.p_vt_in)/sim.zc_vt

    a = sim.a

    zch_b = sim.char_impedances['bore']
    zch_vt = sim.char_impedances['vocal']

    this_dict = {'p_b':p_b,'p_vt':p_vt,#'hhb':hhb,'hhv':hhv,
                'pert_time':sim.pert_time,'p_blow':sim.p_blow_vec,
                'zch_b':zch_b, 'zch_vt':zch_vt,
                    'u':u, 'a':a}

    for probe in sim.probes:
        lab = probe['label']
        this_dict['p_{}'.format(lab)] = np.array(probe['in']) + np.array(probe['out'])

    return this_dict
        
def simulate_js(js):
    impresp, impresp_vt = imp_resp(js,nfft=nfft_ir)
    
    jsfile = base_out_name+'_'+js['startstamp']+'_init.json'
    with open(jsfile,'w') as f:
        json.dump(js.to_python(),f)

    sim = run_smooth_pert(js)
    
    this_dict = collect_time_domain(sim)
    
    this_dict.update({'impresp_b':impresp, 'impresp_vt':impresp_vt, 'js':js})

    return this_dict

def work_on_js(js):
    res = {'simulation':
            {'start':datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S'),
             'params':js.to_python()},
           'startstamp':js['startstamp']
          }
    try:
        data = simulate_js(js)
        if output_data:
            with open(base_out_name+'_'+js['startstamp']+'.pickle','wb') as f:
                pickle.dump(data, f)
        res['simulation']['end'] = datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S')
    except Exception as e:
        err = traceback.format_exc()
        print(err)
        res['simulation']['error'] = err
        return res
        
    res['analysis'] = {}
    try:
        res['analysis'] = do_analysis(data)
    except Exception as e:
        err = traceback.format_exc()
        print(err)
        res['analysis']['error'] = err
        return res
    return res
    

def md_worker(q,iolock):
    while True:
        js = q.get()

        if js is None:
            break

        startstamp = datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S')
        js['startstamp'] = "{:9d}".format(sidx)
        sidx+=1

        with iolock:
            print("Processing "+js['startstamp'])    
        res = work_on_js(js)
        
        # outfile = base_out_name + '_{}.json'.format(datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S'))
        outfile = base_out_name+'_'+res['startstamp']+'.json'

        with open(outfile,'w') as f:
            json.dump(res,f)

if  __name__ == '__main__': 
    #jsfile = 'tongue_2seg_vt_open_simulation_with_tuning.json'

    jsfile = sys.argv[1]

    run_one = False
    try:
        if sys.argv[2] == '-1':
            run_one = True
    except IndexError:
        pass
            

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

    jsg = js_generator(js)

    if run_one:
        js1 = jsg.__next__()
        js1['startstamp']=str(sidx)
        output_data = True
        res = work_on_js(js1)
        print(res)
    else:
            

        mpq = multiprocessing.Queue(maxsize=4)
        iolock = multiprocessing.Lock()
        p = multiprocessing.Pool(8, initializer=md_worker, initargs=(mpq,iolock))

        for js in jsg:
            mpq.put(js)
            with iolock:
                print("Queued")
                
