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

from glottal_simulation_functional import NoInteractionVocalSimulation
from json_object import JSONObject 
from scipy.optimize import minimize
import json_iterators as jsi
from tqdm import trange as trange

__simulation_type__ = "time-domain filter-losses 1st-order reflection filter (object version)"
__simulation_version__ = "20191128"
__parameter_version__ = "20191128"


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

    sim = NoInteractionVocalSimulation()
    with open(infile) as f:
        js = JSONObject(f)
    try:
        jseq = js['sequence']
        print('Sequence found')
    except KeyError:
        jseq = None
    
    if jseq:
        for jj in jsi.json_expander(js, jseq):
            print(jj)
            sim.reset()
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            sim.set_hdf5_path(output,timestamp)
            sim.from_json(jj)
            sim.simulate()


    if args.plot:
        from matplotlib.pyplot import plot, show
        p_sg = sim.p_sg_in + sim.p_sg_out;
        p_vt = sim.p_vt_in + sim.p_vt_out;

        u = (sim.p_vt_out - sim.p_vt_in)/sim.zc_vt;
        u_sg = -(sim.p_sg_out - sim.p_sg_in)/sim.zc_sg

        plot(u)
        show()

