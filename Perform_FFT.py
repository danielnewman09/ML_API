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

def parse_data(data,fftPoints,samplingInterval):
    _,minmax,mean,variance,skewness,kurtosis = describe(data)

    NyquistFrequency = 0.5 / samplingInterval

    freqs,amps = signal.welch(data, fs=1 / samplingInterval, nperseg=fftPoints, scaling='density')

    frequencyInterval = freqs[1] - freqs[0]
    amps = lin_log_interp(amps)

    sampleRMS = np.sqrt(1 / data.shape[0] * np.sum((data - mean)**2))

    output = {'frequencyInterval':frequencyInterval,
              'fftAmps':amps[1:].tolist(),
              'Vibration':data.tolist(),
              'RMS':sampleRMS,
              'Kurtosis':kurtosis,
              'Mean':mean,
              'Skewness':skewness,
              'Variance':variance}
    return output


def lin_log_interp(fft_features):
    '''
    Scale the fft features from the logarithmic axis to be approximately on 
    the interval from 0 to 1
    '''

    # Minimum exponent we expect to see in the data
    minimum = -12

    # Maximum exponent we expect to see
    maximum = 0

    # Number of points to use for interpolation
    numpoints = 1000

    # Map the logarithmic x-axis to a linear y-axis
    x = np.logspace(minimum,maximum,numpoints)
    y = np.linspace(0,1,numpoints)

    # Return the interpolated valuess
    return np.interp(np.log10(fft_features),np.log10(x),y)

if __name__ == '__main__':

    # Create an argParser to parse through the user-given arugments
    argParser = argparse.ArgumentParser()

    # Add an arugment to parse through the raw vibration data.
    argParser.add_argument(
        '-data', '--rawdata', type=str,
        help='comma-separated string of raw accelerometer data'
        )

    # Add an argument to parse the type of machine
    argParser.add_argument(
        '-n', '--fftPoints', type=int,
        help='',
        )

    # Add an arugment to parse the spindle rpm
    argParser.add_argument(
        '-dt', '--samplingInterval', type=float,
        help='',
        )

    # pack the args into a nice list
    args = vars(argParser.parse_args())

    data = np.array([float(i) for i in args['rawdata'].split(',')])
    fftPoints = args['fftPoints']
    samplingInterval = args['samplingInterval']

    _,minmax,mean,variance,skewness,kurtosis = describe(data)

    NyquistFrequency = 0.5 / samplingInterval

    freqs,amps = signal.welch(data, fs=1 / samplingInterval, nperseg=fftPoints, scaling='spectrum')

    frequencyInterval = freqs[1] - freqs[0]

    sampleRMS = np.sqrt(1 / data.shape[0] * np.sum((data - mean)**2))
    # sampleKurtosis = kurtosis(data)

    # _,minmax,mean,variance,skewness,kurtosis = describe(data)


    print('{{ "frequencyInterval":{}, "fftAmps":{}, "Vibration":{}, "RMS":{}, "Kurtosis":{}, "Mean":{}, "Skewness":{}, "Variance": {}  }}'\
        .format(
            np.round(frequencyInterval,5),
            np.array2string(amps,precision=5,separator=','),
            np.array2string(data,precision=5,separator=','),
            np.round(sampleRMS,5),
            np.round(kurtosis,5),
            np.round(mean,5),
            np.round(skewness,5),
            np.round(variance,5)))
