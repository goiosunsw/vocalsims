import sys
import random
import types
import collections
import six
import traceback
import json
from datetime import datetime
from sshtunnel import SSHTunnelForwarder
import pymongo
import os
import paramiko


from copy import deepcopy

from simulation_harmonic_transient_analyser import do_analysis
from json_object import JSONObject
from scipy.optimize import fsolve
import numpy as np
import scipy.signal as sig
import pickle
import multiprocessing

import instr_vt_poleres as ivp

nfft_ir = 2**14

pkey=os.path.join(os.environ['HOME'],'.ssh/id_rsa')
SSH_KEY=paramiko.RSAKey.from_private_key_file(pkey)

MONGO_HOST = "129.94.162.112"
MONGO_USER = "goios"
MONGO_DB = "modal-2duct-simulations"
MONGO_COLLECTION = "random-runs-var-gamma"
local_port = 25017

collection = None
server = None

def gamma_t(t,tau=.05,gamma_lim=.5,gamma_pert=0.45,gamma_fact=1.01):
    g = (1-np.exp(-t/tau))*gamma_lim
    if g>gamma_pert:
        g = gamma_pert*gamma_fact
    return g

def perturbation(t):
    if t<0:
        return 0
    if t<1:
        #return (1-cos(pi*t))/2
        return t
    return 1

def nl_from_js(js):
    if js is None:
        return None
    else:
        if js['type'] == 'parabolic':
            x_st = js['nl_start']
            x_ev = js['nl_ev']
            

def ivp_simulate(js):

    
    jsg = js['gamma']
    gamma = jsg['sustain_val']
    tau = jsg['time_const']
    gamma_lim_mult= jsg['exp_limit_multiplier']
    
    zeta = js['zeta']
    
    jsr = js['reed']
    fr = jsr['frequency']
    qr = jsr['q']
    ar = jsr['amplitude']

    jsb = js['bore']             
    fac = np.array(jsb['frequencies'].to_python())
    qac = np.array(jsb['qs'].to_python())
    aac = np.array(jsb['amplitudes'].to_python())

    jsv = js['vocal']
    fvt = np.array(jsv['frequencies'].to_python())
    qvt = np.array(jsv['qs'].to_python())
    avt = np.array(jsv['amplitudes'].to_python())

    nlfunc = nl_from_js(js['nlfunc'])
    tmax = js['tmax']
    
    jsp = js['perturbation']
    pert_mult = jsp['multiplier'] 
    pert_duration = jsp['duration']
    
    gamma_lim = gamma*gamma_lim_mult
    gamma_t = lambda t: (1-np.exp(-t/tau))*gamma_lim

    
    cs=ivp.croak_poleres(gamma=gamma_t,zeta=zeta,
                         fr=fr,qr=1,ar=1,
                         f1=fac,q1=qac,a1=aac,
                         fv=fvt,qv=qvt,av=avt,
                         nlfunc=nlfunc)
    cs.gamma_vec=[]
    cs.zeta_vec=[]
    cs.perturbation_on=False

    cs.set_tmax(tmax)
    cs.set_initial_state(x0=0.,v0=0.,p0=0.,dp0=0.)
    cs.setup_integrator()
    
    js['simulation']={}
    js['simulation']['sample rate'] = 1./cs.dt

    while cs.odesol.t <= cs.tmax and not cs.error:
        newt = cs.cur_t+cs.dt
        if cs.gamma(newt)>gamma and not cs.perturbation_on:
            cs.perturbation_on = True
            cs.pert_time = newt
        if cs.perturbation_on:  
            pert_mult_t = 1+(perturbation((newt-cs.pert_time)/pert_duration))*(pert_mult-1)
            cs.gamma = lambda t: gamma/pert_mult_t
            cs.zeta = zeta*pert_mult_t
        cs.calculate_at_time(newt)
        cs.gamma_vec.append(cs.gamma(newt))
        cs.zeta_vec.append(cs.zeta)

    cs.finish_simulation()
    cs.gamma_vec=np.array(cs.gamma_vec)
    cs.zeta_vec=np.array(cs.zeta_vec)
    
    data = {'p_b':cs.y[:,1],'p_vt':cs.y[:,2],#'hhb':hhb,'hhv':hhv,
                'pert_time':cs.pert_time,'p_blow':cs.gamma_vec,
                'js':js.to_python(),
                'a':cs.y[:,0]}
    return data

def ivp_plot(cs,ax=None,label=None):
    t = cs.t
        
    if ax is None:
        fig,ax = subplots(2,sharex=True)
        ax[0].set_ylabel('Reed displacement')
        ax[0].grid(True)
        ax[1].set_ylabel('Acoust. pressure')
        ax[1].set_xlabel('Time')
        ax[1].grid(True)

    ax1 = ax[0]
    ax1.plot(cs.t,cs.y[:,0], label=label)

    ax2 = ax[1]
    lns = ax2.plot(cs.t,cs.y[:,1], label=label)
    
    if cs.nvtmodes > 0:
        pvt = np.asarray(cs.y[:,2])+cs.gamma_vec
        ax2.plot(cs.t,pvt,ls='--',color=lns[0].get_color())
    return ax


def json_path_get(js, addr):
    ret = js
    for kk in addr.split('/'):
        try:
            kk=int(kk)
        except ValueError:
            pass
        ret = ret[kk]
    return ret

def json_path_set(js, addr, v):
    ret = js
    for kk in addr.split('/'):
        try:
            kk=int(kk)
        except ValueError:
            pass
        oldret = ret
        ret = ret[kk]
    oldret[kk] = v

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
        apply_underscores.js = js.copy()
        
    if root.split('/')[-1] == '_link':
        vv = json_path_get(apply_underscores.js,js)
        #print(vv,root)
        yield from apply_underscores(vv, root=root[:root.find('/_link')])
        #print("!L"+str(vv))
    # elif root.split('/')[-1] == '_range':
    #     vr = js
    #     js = random.uniform(min(vr),max(vr))
    #     print("!R")
    #     yield ('/').join(root.split('/')[:-1]), js
    # elif root.split('/')[-1] == '_choice':
    #     vl = js
    #     js = random.choice(vl)
    #     print("!C")
    #     yield ('/').join(root.split('/')[:-1]), js
    # elif root.split('/')[-1] == '_value':
    #     print("!V")
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
                        json_path_set(apply_underscores.js,fullv,v)
                        
                        #print(fullv,v)
                        #print(apply_underscores.js)
                    elif '_choice' in v.keys():
                        vl = v['_choice']
                        v = random.choice(vl)
            yield from apply_underscores(v,root=fullv)

# def imp_resp(js, nfft=1024):
#     sim = ReedSimulation()
#     sim.from_json(js)
#     impresp = sim.tracts['bore'].impulse_response(n=nfft)
#     impresp_vt = sim.tracts['vocal'].impulse_response(n=nfft)
#     return impresp, impresp_vt



def js_generator(js0):
    
    while True:

        jsn = JSONObject()
        for k,v in apply_underscores(js0.copy().to_python()):
            if not is_iterable_but_not_string(v):
                jsn[k]=v

        yield jsn

from time import sleep

def work_on_js(js):
    res = {'simulation':
            {'start':datetime.now(),
             'params':js.to_python()}
          }
    try:
        data = ivp_simulate(js)
        res['simulation']['end'] = datetime.now()
    except Exception as e:
        err = traceback.format_exc()
        print(err)
        res['simulation']['error'] = err
        return res
        
    res['analysis'] = {}
    try:
        res['analysis'] = do_analysis(data, impedance=False)
    except Exception as e:
        err = traceback.format_exc()
        print(err)
        res['analysis']['error'] = err
        return res
    try:    
        write_to_mongo(res)
    except Exception as e:
        err = traceback.format_exc()
        print(err)
 
    return res
    

def md_worker(q,iolock):
    while True:
        js = q.get()

        if js is None:
            break


        with iolock:
            print("Processing")    
        res = work_on_js(js)
        

        #with open(outfile,'w') as f:
        #    json.dump(res,f)

def write_to_mongo(js):
        
    with pymongo.MongoClient('localhost', local_port) as connection:
        db = connection[MONGO_DB]
        # mongo_collection = js['simulation']['params']['db']
        # print(mongo_collection)
        mongo_collection = MONGO_COLLECTION
        collection = db[mongo_collection]

        collection.insert_one(js)

def get_server():
    return SSHTunnelForwarder(
        MONGO_HOST,
        ssh_username=MONGO_USER,
        ssh_pkey=SSH_KEY,
        remote_bind_address=('localhost', 27017),
        local_bind_address=('localhost', local_port)
    )

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

    # MONGO_COLLECTION = js['db']
    n_main = 1


    len_range = 0.03
    nfft_ir = 2**14

    jsg = js_generator(js)

    # define ssh tunnel
    server = get_server()

    if run_one:
        js1 = jsg.__next__()
        res = work_on_js(js1)
        print(res)
    else:
        base_out_name = 'tongue_vt_open_tuning'

            

        mpq = multiprocessing.Queue(maxsize=4)
        iolock = multiprocessing.Lock()
        p = multiprocessing.Pool(8, initializer=md_worker, initargs=(mpq,iolock))

        for js in jsg:
            # check here whether the SSH server is still active
            if not (server.is_active):
                print("reconnecting SSH tunnel... " + datetime.ctime(datetime.now()))
                server.stop()
                server = get_server()
                server.start()
            mpq.put(js)
            with iolock:
                print("Queued")
                
