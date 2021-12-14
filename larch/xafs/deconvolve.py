#!/usr/bin/env python
# XAS spectral decovolution
#

import numpy as np
from scipy.signal import deconvolve
from scipy.ndimage import gaussian_filter
from larch import parse_group_args

from larch.math import (gaussian, lorentzian, interp,
                        index_of, index_nearest, remove_dups,
                        savitzky_golay)

from .xafsutils import set_xafsGroup

import matplotlib.pyplot as plt


def xas_deconvolve(energy, norm=None, group=None, form='lorentzian',
                   esigma=1.0, eshift=0.0, smooth=True,
                   sgwindow=None, sgorder=3, _larch=None):
    """XAS spectral deconvolution

    de-convolve a normalized mu(E) spectra with a peak shape, enhancing the
    intensity and separation of peaks of a XANES spectrum.

    This uses numpy's "deconvolve" function, using inverse filtering.

    The results can be unstable, and noisy, and should be used
    with caution!

    Arguments
    ----------
    energy:   array of x-ray energies (in eV) or XAFS data group
    norm:     array of normalized mu(E)
    group:    output group
    form:     functional form of deconvolution function. One of
              'gaussian' or 'lorentzian' [default]
    esigma    energy sigma to pass to gaussian() or lorentzian()
              [in eV, default=1.0]
    eshift    energy shift to apply to result. [in eV, default=0]
    smooth    whether to smooth result with savitzky_golay method [True]
    sgwindow  window size for savitzky_golay [found from data step and esigma]
    sgorder   order for savitzky_golay [3]

    Returns
    -------
    None
       The array 'deconv' will be written to the output group.

    Notes
    -----
       Support See First Argument Group convention, requiring group
       members 'energy' and 'norm'

       Smoothing with savitzky_golay() requires a window and order.  By
       default, window = int(esigma / estep) where estep is step size for
       the gridded data, approximately the finest energy step in the data.
    """
    energy, mu, group = parse_group_args(energy, members=('energy', 'norm'),
                                         defaults=(norm,), group=group,
                                         fcn_name='xas_deconvolve')
    eshift = eshift + 0.5 * esigma

    en  = remove_dups(energy)
    estep1 = int(0.1*en[0]) * 2.e-5
    en  = en - en[0]
    estep = max(estep1, 0.01*int(min(en[1:]-en[:-1])*100.0))

    npts = 1  + int(max(en) / estep)
    if npts > 25000:
        npts = 25001
        estep = max(en)/25000.0

    x = np.arange(npts)*estep
    y = interp(en, mu, x, kind='cubic')

    kernel = lorentzian
    if form.lower().startswith('g'):
        kernel = gaussian

    yext = np.concatenate((y, np.arange(len(y))*y[-1]))

    ret, err = deconvolve(yext, kernel(x, center=0, sigma=esigma))
    nret = min(len(x), len(ret))

    ret = ret[:nret]*yext[nret-1]/ret[nret-1]
    if smooth:
        if sgwindow is None:
            sgwindow = int(1.0*esigma/estep)

        sqwindow = int(sgwindow)
        if sgwindow < (sgorder+1):
            sgwindow = sgorder + 2
        if sgwindow % 2 == 0:
            sgwindow += 1
        ret = savitzky_golay(ret, sgwindow, sgorder)

    out = interp(x+eshift, ret, en, kind='cubic')
    group = set_xafsGroup(group, _larch=_larch)
    group.deconv = out





def warp_array(input_function, energies, warp_function):
    incremental_energies = (energies[-1]/len(energies))*np.ones_like(energies) # first assume linear sampling
    warped_incremental_energies = incremental_energies / (warp_function/np.min(warp_function))
    warped_energies = np.cumsum(warped_incremental_energies) - warped_incremental_energies[0]

    warped_new_uniform_grid = np.linspace(0,warped_energies[-1], samples)
    re_sampled_warped_convolved = interpolate.interp1d(warped_energies, input_function, kind="cubic")(warped_new_uniform_grid)

    return warped_new_uniform_grid, re_sampled_warped_convolved


# Arguments for and against separate function: 
# Safer, less scary
# deconvolve already takes a bunch of args,
# sg smoothing is not needed
# breaks DRY, lots of identical setup / teardown code

            # try:
            #     import skimage
            #     from silx.image.marchingsquares._skimage import MarchingSquaresSciKitImage
            #     self._ms = MarchingSquaresSciKitImage(image,
            #                                           mask=mask)
            # except ImportError:
            #     self._logger.error('skimage not found')
            #     self._ms = None

# Tests: 0 array, array longer than 25000 (might need a tweak of the length algo)

import skimage

def xas_iterative_deconvolve(energy, norm=None, group=None, form='lorentzian',
                   esigma=1.0, regularization_filter_width=0.5, grid_spacing=None, eshift=0.0, max_iterations=1000, _larch=None):
    """XAS spectral deconvolution

    de-convolve a normalized mu(E) spectra with a peak shape, enhancing the
    intensity and separation of peaks of a XANES spectrum.

    If the plain deconvolution needs attention,
    this is not tested sufficiently for publication-quality data. 
    Please validate against other techniques if possible


    WARNING: there seems to be some kind of shift somehow.

    Arguments
    ----------
    energy:   array of x-ray energies (in eV) or XAFS data group
    norm:     array of normalized mu(E)
    group:    output group
    form:     functional form of deconvolution function. One of
              'gaussian' or 'lorentzian' [default]
    esigma    energy sigma to pass to gaussian() or lorentzian()
              [in eV, default=1.0]
    eshift    energy shift to apply to result. [in eV, default=0]
    smooth    whether to smooth result with savitzky_golay method [True]
    sgwindow  window size for savitzky_golay [found from data step and esigma]
    sgorder   order for savitzky_golay [3]

    Returns
    -------
    None
       The array 'deconv' will be written to the output group.

    Notes
    -----
       Support See First Argument Group convention, requiring group
       members 'energy' and 'norm'

    """

    #
    # Setup 
    #
    energy, mu, group = parse_group_args(energy, members=('energy', 'norm'),
                                         defaults=(norm,), group=group,
                                         fcn_name='xas_deconvolve')
    eshift = eshift + 0.5 * esigma

    en  = remove_dups(energy)
    en  = en - en[0]

    print("energy:",en)
    if(not grid_spacing):
        estep1 = int(0.1*en[0]) * 2.e-5
        print(0.01*int(min(en[1:]-en[:-1])*100.0))
        print(en[1:],en[:-1])
        estep = max(estep1, 0.01*int(min(en[1:]-en[:-1])*100.0))
        npts = 1  + int(max(en) / estep)
        if npts > 25000:
            npts = 25001
            estep = max(en)/25000.0

        x = np.arange(npts)*estep

    else:
        estep = grid_spacing
        x = np.arange(0, en[-1], estep)
        npts = len(x)

    y = interp(en, mu, x, kind='cubic')

    #
    # ??!? is this a window function or supposed to be padding
    #
    # yext = np.concatenate((y, np.arange(len(y))*y[-1]))
    yext = y

    # plt.plot(np.arange(len(P_conv_On)), P_conv_On)
    # plt.plot(np.arange(len(I_over_P_conv_On)), I_over_P_conv_On)
    # plt.plot(np.arange(len(error_estimate)), error_estimate)
    # plt.plot(np.arange(len(guess)), guess)

    # plt.show()
    #
    # Process PSF / experimental distribution kernel
    #
    kernel = lorentzian
    if form.lower().startswith('g'):
        kernel = gaussian

    point_spread_function = kernel(x, center=x[-1]/2, sigma=esigma)
    # point_spread_function /= np.sum(point_spread_function)
    # because r-l flips the psf, it's important that this be rotationally symmetric
    # plt.plot(np.arange(len(point_spread_function)), point_spread_function)

    # plt.show()

    # "I" in Fister et al
    # could use a better name

    #
    # Perform deconvolution proper
    # 


    ret = 0.5*np.ones_like(yext) # may want the initial guess to be variable?
    # "On" in Fister et al

    # scikit-image has a richardson_lucy function. However, 
    # we want to smooth on each iteration.

    # constructed referring to 
    # https://github.com/chrrrisw/RL_deconv
    # and 
    # https://github.com/scikit-image/scikit-image/blob/602d94d35d3a04e6b66583c3a1a355bfbe381224/skimage/restoration/deconvolution.py

    # how does this really work? I should try to describe it intuitively.

    flipped_point_spread_function = point_spread_function[::-1] #P* in Fister et al
    convergence = []

    # ret = skimage.restoration.richardson_lucy(yext, point_spread_function, num_iter=50, filter_epsilon=False, clip=False)
    # ret /= 20.0
    # ret,_ = deconvolve(yext,point_spread_function) 

    # eps = 1e-12

    # for _ in range(50):
    #     conv = np.convolve(ret, point_spread_function, mode='same') + eps
    #     relative_blur = yext / conv
    #     ret *= np.convolve(relative_blur, flipped_point_spread_function, mode='same')

    for i in range(max_iterations):
        # convolution is commutative
        P_conv_On = np.convolve(ret,point_spread_function, mode='same') + 1e-12
        # plt.plot(np.arange(len(P_conv_On)), P_conv_On)
        # plt.plot(np.arange(len(yext)), yext)
        # plt.show()
        I_over_P_conv_On = yext/P_conv_On # zeros nan here 
        error_estimate = np.convolve(I_over_P_conv_On,flipped_point_spread_function, mode='same')


        # re-smooth each iteration - the "regularizing filter"
        # ret = ret * error_estimate
        ret = gaussian_filter(ret * error_estimate, regularization_filter_width)

        chi_squared = np.sum(((P_conv_On - yext)**2) / yext)
        chi_squared *= 1.0 / len(yext) 
        convergence.append(chi_squared)

    convergence = np.array(convergence)


    #
    # Trim, scale, and return output
    #
    nret = min(len(x), len(ret))
    ret = ret[:nret]

    out = interp(x+eshift, ret, en, kind='cubic')

    group = set_xafsGroup(group, _larch=_larch)
    group.deconv = out
    group.convergence = convergence


# def xas_iterative_deconvolve_robustness(energy, norm=None, group=None, form='lorentzian',





def xas_convolve(energy, norm=None, group=None, form='lorentzian',
                   esigma=1.0, eshift=0.0, _larch=None):
    """
    convolve a normalized mu(E) spectra with a Lorentzian or Gaussian peak
    shape, degrading separation of XANES features.

    This is provided as a complement to xas_deconvolve, and to deliberately
    broaden spectra to compare with spectra measured at lower resolution.

    Arguments
    ----------
    energy:   array of x-ray energies (in eV) or XAFS data group
    norm:     array of normalized mu(E)
    group:    output group
    form:     form of deconvolution function. One of
              'lorentzian' or  'gaussian' ['lorentzian']
    esigma    energy sigma (in eV) to pass to gaussian() or lorentzian() [1.0]
    eshift    energy shift (in eV) to apply to result [0]

    Returns
    -------
    None
       The array 'conv' will be written to the output group.

    Notes
    -----
       Follows the First Argument Group convention, using group members named
       'energy' and 'norm'
    """

    energy, mu, group = parse_group_args(energy, members=('energy', 'norm'),
                                         defaults=(norm,), group=group,
                                         fcn_name='xas_convolve')
    eshift = eshift + 0.5 * esigma

    en  = remove_dups(energy)
    en  = en - en[0]
    estep = max(0.001, 0.001*int(min(en[1:]-en[:-1])*1000.0))

    npad = 1 + int(max(estep*2.01, 50*esigma)/estep)

    npts = npad  + int(max(en) / estep)

    x = np.arange(npts)*estep
    y = interp(en, mu, x, kind='cubic')

    kernel = lorentzian
    if form.lower().startswith('g'):
        kernel = gaussian

    k = kernel(x, center=0, sigma=esigma)
    ret = np.convolve(y, k, mode='full')

    out = interp(x-eshift, ret[:len(x)], en, kind='cubic')

    group = set_xafsGroup(group, _larch=_larch)
    group.conv = out / k.sum()
