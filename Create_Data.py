#!/usr/bin/python
#----------------------------------------------------------------------
# analyze_vibmachine.py
#
# perform desired analysis on a certain vibmachine
#
# Created: September 11, 2018 - Daniel M Newman -- danielnewman@gatech.edu
#
# Modified:
#   * Septermber 11, 2018 - DMN
#            - Added documentation for this script
#----------------------------------------------------------------------


import os
import sys
import argparse
import json

from scipy import signal
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import describe

import numpy as np

def create_noisy_signal(
    duration, samplingRate, frequencies, amplitudes,
    noiseStDev, phase, 
    frequencyError=0.05, harmonics=1,
    saveSignal=False,):
    '''
    create_noisy_signal

    Create a signal with desired randomness and spectral qualities.

    Inputs:
        - duration: time (in seconds) captured by the signal
        - samplingRate: rate (in Hz) of the signal
        - frequencies: list of frequencies in the signal
        - amplitudes: amplitudes of the corresponding frequencies
        - (float) noiseStDev: standard deviation squared) of
                the gaussian noise added to the signal
        - (float) frequencyStDev: standard deviation
                of the gaussian noise added to the frequency
        - (float) amplitudeStDev: standard deviation
                of the gaussian noise added to the amplitudes
        - (float) phaseStDev: StDev (standard deviation squared) of
                the gaussian noise added to the phase of the signal

    '''

    # determine the required number of datapoints to cover the duration
    # at the required sampling rate
    numPoints = int(duration * samplingRate)

    # Create a time array with the correct start and endpoint, sampled at
    # the required sampling rates
    time = np.atleast_2d(np.linspace(0,duration,numPoints))

    # Ensure that all of the inputs are cast as numpy arrays
    freqs = np.atleast_2d(np.asarray(frequencies).flatten()).T
    amps = np.atleast_2d(np.asarray(amplitudes).flatten()).T
    noiseStDev = np.asarray(noiseStDev)

    # Modify the signal slightly
    m, n = freqs.shape
#     phase = np.atleast_2d(phaseStDev * np.random.random((m, n)))

    # Create randomly distributed noise with a given standard deviation
    noise = noiseStDev * np.random.random(numPoints)

    # The number of input frequencies must be identical to the number
    # of input amplitudes
    if len(freqs) != len(amps):
        raise ValueError('Length of input frequencies must be identical to\
                          that of input amplitudes')

    signal = np.sum(amps * np.sin(2 * np.pi * freqs * time + phase), axis=0) + noise


    return signal