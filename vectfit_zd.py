"""
Duplication of the vector fitting algorithm in python (http://www.sintef.no/Projectweb/VECTFIT/)

All credit goes to Bjorn Gustavsen for his MATLAB implementation, and the following papers


 [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
     domain responses by Vector Fitting", IEEE Trans. Power Delivery,
     vol. 14, no. 3, pp. 1052-1061, July 1999.

 [2] B. Gustavsen, "Improving the pole relocating properties of vector
     fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
     July 2006.

 [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
     "Macromodeling of Multiport Systems Using a Fast Implementation of
     the Vector Fitting Method", IEEE Microwave and Wireless Components
     Letters, vol. 18, no. 6, pp. 383-385, June 2008.
"""
__author__ = 'Phil Reinhold'
from pylab import *
from numpy.linalg import eigvals, lstsq

def cc(z):
    return z.conjugate()

def model(s, poles, residues, d, h):
    return sum(r/(1-p/s) for p, r in zip(poles, residues)) + d + h/s

def vectfit_step(f, s, poles, weights=None, relocate_pi=False):
    """
    f = complex data to fit
    s = j*frequency
    poles = initial poles guess
        note: All complex poles must come in sequential complex conjugate pairs
    returns adjusted poles
    """
    N = len(poles)
    Ns = len(s)

    cindex = zeros(N)
    # cindex is:
    #   - 0 for real poles
    #   - 1 for the first of a complex-conjugate pair
    #   - 2 for the second of a cc pair
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % (poles[i], poles[i+1]))
                cindex[i] = 1
            else:
                cindex[i] = 2

    # First linear equation to solve. See Appendix A
    A = zeros((Ns, 2*N+2), dtype=np.complex64)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

        A [:, N+2+i] = -A[:, i] * f/s

    A[:, N] = 1/s
    A[:, N+1] = 1/s**2

    A = A * np.outer(weights, np.ones(2*N+2))
    # Solve Ax == b using pseudo-inverse
    b = f/s * weights
    A = vstack((real(A), imag(A)))
    b = concatenate((real(b), imag(b)))
    x, residuals, rnk, s = lstsq(A, b, rcond=-1)

    residues = x[:N]
    d = x[N]
    h = x[N+1]

    # We only want the "tilde" part in (A.4)
    x = x[-N:]

    # Calculation of zeros: Appendix B
    A = diag(poles)
    b = ones(N)
    c = x
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = real(p), imag(p)
            A[i, i] = A[i+1, i+1] = x
            A[i, i+1] = -y
            A[i+1, i] = y
            b[i] = 2
            b[i+1] = 0
            #cv = c[i]
            #c[i,i+1] = real(cv), imag(cv)

    H = A - outer(b, c)
    H = real(H)
    poles_unsorted = eigvals(H)
    real_poles = poles_unsorted[poles_unsorted.imag==0]
    real_poles = sort(real_poles)
    cplx_poles = poles_unsorted[poles_unsorted.imag!=0]
    cplx_poles = flipud(sort(cplx_poles))

    new_poles = np.concatenate((real_poles,cplx_poles))
    unstable = np.abs(new_poles) > 1.
    new_poles[unstable] = 1./new_poles[unstable]
    if relocate_pi:
        pi_poles = np.logical_and(new_poles.imag==0, new_poles.real<0)
        new_poles[pi_poles] = -new_poles[pi_poles]

    return new_poles

# Dear gods of coding style, I sincerely apologize for the following copy/paste
def calculate_residues(f, s, poles, rcond=-1, weights=None):
    Ns = len(s)
    N = len(poles)

    cindex = zeros(N)
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % poles[i:i+1])
                cindex[i] = 1
            else:
                cindex[i] = 2

    # use the new poles to extract the residues
    A = zeros((Ns, N+2), dtype=np.complex128)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

    A[:, N] = 1/s
    A[:, N+1] = 1/s**2

    A = A * np.outer(weights, np.ones(N+2))
    # Solve Ax == b using pseudo-inverse
    b = f/s * weights
    A = vstack((real(A), imag(A)))
    b = concatenate((real(b), imag(b)))
    cA = np.linalg.cond(A)
    if cA > 1e13:
        print ('Warning!: Ill Conditioned Matrix. Consider scaling the problem down')
        print ('Cond(A)', cA)
    x, residuals, rnk, s = lstsq(A, b, rcond=rcond)

    # Recover complex values
    x = np.complex64(x)
    for i, ci in enumerate(cindex):
       if ci == 1:
           r1, r2 = x[i:i+2]
           x[i] = r1 - 1j*r2
           x[i+1] = r1 + 1j*r2

    residues = x[:N]
    d = x[N].real
    h = x[N+1].real
    return residues, d, h

def print_params(poles, residues, d, h):
    cfmt = "{0.real:g} + {0.imag:g}j"
    print ("poles: " + ", ".join(cfmt.format(p) for p in poles))
    print ("residues: " + ", ".join(cfmt.format(r) for r in residues))
    print ("offset: {:g}".format(d))
    print ("slope: {:g}".format(h))

def vectfit_auto(f, s, n_poles=10, n_iter=10, show=False,
                 inc_real=False, loss_ratio=1e-2, rcond=-1, 
                 track_poles=False, init_resonances=None,
                 weights=None,relocate_pi=False):
    w = imag(s)
    if init_resonances is None: 
        pole_angles = linspace(np.angle(s[0]), np.angle(s[-1]), n_poles+2)[1:-1]
        lr = loss_ratio
        init_poles = poles = concatenate([[(1-lr)*np.exp(1j*p), (1-lr)*np.exp(-1j*p)] for p in pole_angles])
    else:
        init_poles = init_resonances

    if weights is None:
        weights = np.ones(len(s))
       

    if inc_real:
        poles = concatenate((poles, [1]))

    poles_list = []
    for _ in range(n_iter):
        poles = vectfit_step(f, s, poles, weights=weights, relocate_pi=relocate_pi)
        poles_list.append(poles)

    residues, d, h = calculate_residues(f, s, poles, rcond=rcond, weights=weights)

    if track_poles:
        return poles, residues, d, h, np.array(poles_list)

    print_params(poles, residues, d, h)
    return poles, residues, d, h

def poleres2pz(poles,residues,offset,slope, maxsteps=500):
    """
    Convert a pole-resiude representation of a TF -- 
     
        f(s) = Sum [r/(1-p/z)] + k + h/z 

    to a pole zero representation

        f(s) = Sum[1-r/z] / Sum[1-p/z]
    """
    from sympy import Symbol, together, expand, numer, denom, nroots

    x = Symbol('x')
    f = sum([r/(1-p*x) for p,r in zip(poles, residues)])
    f += offset + slope*x

    ft = together(f)
    nf = expand(numer(ft))

    zeros = nroots(nf,maxsteps=maxsteps)
    return np.array([complex(xx) for xx in poles]), np.array([complex(xx) for
                                                              xx in zeros])
    
def vectfit_auto_rescale(f, s, **kwargs):
    s_scale = abs(s[-1])
    f_scale = abs(f[-1])
    print ('SCALED')
    poles_s, residues_s, d_s, h_s = vectfit_auto(f / f_scale, s / s_scale, **kwargs)
    poles = poles_s * s_scale
    residues = residues_s * f_scale * s_scale
    d = d_s * f_scale
    h = h_s * f_scale / s_scale
    print ('UNSCALED')
    print_params(poles, residues, d, h)
    return poles, residues, d, h

if __name__ == '__main__':
    test_s = 1j*np.linspace(1, 1e5, 800)
    test_poles = [
        -4500,
        -41000,
        -100+5000j, -100-5000j,
        -120+15000j, -120-15000j,
        -3000+35000j, -3000-35000j,
        -200+45000j, -200-45000j,
        -1500+45000j, -1500-45000j,
        -500+70000j, -500-70000j,
        -1000+73000j, -1000-73000j,
        -2000+90000j, -2000-90000j,
    ]
    test_residues = [
        -3000,
        -83000,
        -5+7000j, -5-7000j,
        -20+18000j, -20-18000j,
        6000+45000j, 6000-45000j,
        40+60000j, 40-60000j,
        90+10000j, 90-10000j,
        50000+80000j, 50000-80000j,
        1000+45000j, 1000-45000j,
        -5000+92000j, -5000-92000j
    ]
    test_d = .2
    test_h = 2e-5

    test_f = sum(c/(test_s - a) for c, a in zip(test_residues, test_poles))
    test_f += test_d + test_h*test_s
    vectfit_auto(test_f, test_s)

    poles, residues, d, h = vectfit_auto_rescale(test_f, test_s)
    fitted = model(test_s, poles, residues, d, h)
    figure()
    plot(test_s.imag, test_f.real)
    plot(test_s.imag, test_f.imag)
    plot(test_s.imag, fitted.real)
    plot(test_s.imag, fitted.imag)
    show()