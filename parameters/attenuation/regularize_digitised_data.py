import argparse
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input',help='input file with arbitrary data in CSV format')
    parser.add_argument('-n','--num-points', help='number of points in regularised table')
    parser.add_argument('-x','--xmin', type=float, help='minimum bound of x values')
    parser.add_argument('-X','--xmax', type=float, help='maximum bound of x values')
    parser.add_argument('-l','--log', action='store_true', help='log-spaced output')
    parser.add_argument('-R','--row-title', action='store_true', help='input includes extra line with row title')
    parser.add_argument('-m','--multiplier', default=1.0, type=float, help='multiplier for all cells')
    parser.add_argument('-i','--interpolation-method', default='cubic')
    args = parser.parse_args() 
    return args

if __name__ == '__main__':
    args = parse_arguments()
    
    jat = pd.read_csv(args.input,header=0,index_col=0)
    if args.row_title:
        jat2 = pd.read_csv(args.input,header=1,index_col=0)

        jat2.columns = jat.columns
        jat2.columns.name = jat.index.name
        jat=jat2

    try:
        numpoints = int(args.num_points)
    except TypeError:
        numpoints = int(jat.apply(lambda x: np.sum(~ np.isnan(x)),axis=0).median())
    
    if args.xmin:
        xmin = args.xmin
    else:
        xmin = np.min(jat.index.values)
    
    if args.xmax:
        xmax = args.xmax
    else:
        xmax = np.max(jat.index.values)

    if args.log:
        newx = np.logspace(np.log10(xmin), np.log10(xmax), numpoints)
    else:
        newx = np.linspace(xmin, xmax, numpoints)

    jat *= args.multiplier

    newjat = pd.DataFrame(columns=jat.columns, index=newx)

    for icol, col in jat.iteritems():
        idx = ~ np.isnan(col)
        fun = interp1d(col.index[idx].values, col[idx], 
                       kind=args.interpolation_method,
                       fill_value='extrapolate')
        newjat[icol] = fun(newx)


    newjat.to_csv(sys.stdout)