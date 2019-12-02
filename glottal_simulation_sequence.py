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

from .glottal_simulation_functional import VocalSimulation
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

    sim = VocalSimulation()
    sim.set_output_hdf5(output)
    js = json.read(infile)
    try:
        jseq = js['sequence']
        print('Sequence found')
    else:
        jseq = None
    
    if jseq:
        
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

