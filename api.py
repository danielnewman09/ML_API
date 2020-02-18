#!flask/bin/python
from flask import Flask, jsonify
from flask import request
import os
import binascii

import datetime

import json

from Perform_FFT import parse_data
from Create_Data import create_noisy_signal

app = Flask(__name__)

import numpy as np
import tflite_runtime.interpreter as tflite
import tensorflow as tf
import tensorflow.keras as keras
from Custom_Layers import Dropout_Live


class TF_Model(object):

    def __init__(self,path):


@app.route('/models/save',methods=['POST'])
def save_model():

    hex_vals = request.json['values']
    file_path = request.json['path']
    file_name = request.json['filename']

    if not os.path.exists(file_path):
        os.makedirs(file_path)


    with open(file_path + file_name,'wb') as fout:
        fout.write(binascii.unhexlify(hex_vals))

    return jsonify({'output':True}),201

@app.route('/simulate/vibration',methods=['POST'])
def simulate_vibration():

    duration = request.json['duration']
    samplingRate = request.json['samplingRate']
    amplitudes = np.array([request.json['amplitudes']])
    frequencies = np.array([request.json['frequencies']])
    noiseStd = request.json['noiseStd']

    NyquistFreq = 0.5 * samplingRate

    amplitudes = np.linspace(1.0,2.5,50)
    frequencies = np.linspace(0.3 * NyquistFreq, 0.3 * NyquistFreq,1)
    noiseStdDev = np.linspace(0,0,1)
    phase = np.linspace(-np.pi/2,np.pi/2,25)

    amplitude = np.random.choice(amplitudes)
    phase = np.random.choice(phase)
    print(amplitude)
    print(phase)

    signal = create_noisy_signal(duration,samplingRate,frequencies,amplitude,noiseStd,phase)
    signal = signal.tolist()
    output = {'values':signal}

    return jsonify(output), 201


@app.route('/features/parse/vibration',methods=['POST'])
def parse_vibration():
    values = np.array(request.json['values']).astype(float)
    fftPoints = request.json['fftPoints']
    samplingInterval = request.json['samplingInterval']

    output = parse_data(values,fftPoints,samplingInterval)

    return jsonify(output), 201

@app.route('/models/inference/full',methods=['POST'])
def model_inference_full():
    assetId = request.json['assetId']
    dataItemId = request.json['dataItemId']
    isWarmUp = request.json['isWarmUp']
    spindleSpeed = request.json['spindleSpeed']
    xInference = np.array(request.json['xInference']).astype(float)
    basePath = request.json['basePath']
    modelId = request.json['modelId']
    feature = request.json['feature']


    model_path = basePath + 'Models/' + assetId + '/' + dataItemId + '/' + str(isWarmUp).lower() + '/' + str(spindleSpeed) + '/'

    if not os.path.exists(model_path):
        return jsonify({'output':False}),201

    with open(model_path + 'control_params_{}_{}_full.json'.format(modelId,feature), 'r') as fp:
        param_dict = json.load(fp)
    
    start = datetime.datetime.now()

    new_model = tf.keras.models.load_model(model_path + "model_{}_{}_full.h5".format(modelId,feature),custom_objects={'Dropout_Live': Dropout_Live})

    end = datetime.datetime.now()

    print((end-start).microseconds)

    num_samples = 1

    xInference = xInference.reshape(1,512,1)
    X_predict = np.repeat(xInference,num_samples,axis=0)

    start = datetime.datetime.now()
    predict = new_model.predict(X_predict)
    end = datetime.datetime.now()
    print((end - start).microseconds)
    mse = 1 / num_samples * np.sum((X_predict - predict)**2,axis=1)
    mse = mse.reshape(int(mse.shape[0] / num_samples),num_samples)

    means = np.mean(mse,axis=1)
    variances = np.var(mse,axis=1)

    zMeans = (means - float(param_dict['avgMean'])) / float(param_dict['avgStd'])
    zStds = (variances - float(param_dict['varMean'])) / float(param_dict['varStd'])

    output = {
        'valueMean':zMeans.tolist(),
        'valueStd':zStds.tolist(),
        'dataItemId':dataItemId,
        'state':spindleSpeed,
        'modelId':modelId,
        'feature':feature
    }

    return jsonify(output), 201

@app.route('/models/inference/lite',methods=['POST'])
def model_inference_lite():

    assetId = request.json['assetId']
    dataItemId = request.json['dataItemId']
    isWarmUp = request.json['isWarmUp']
    spindleSpeed = request.json['spindleSpeed']
    xInference = np.array(request.json['xInference']).astype(float)
    basePath = request.json['basePath']
    modelId = request.json['modelId']
    feature = request.json['feature']


    model_path = basePath + 'Models/' + assetId + '/' + dataItemId + '/' + str(isWarmUp).lower() + '/' + str(spindleSpeed) + '/'
    print(model_path)
    if not os.path.exists(model_path):
        return jsonify({'output':False}),201

    with open(model_path + 'control_params_{}_{}_lite.json'.format(modelId,feature), 'r') as fp:
        param_dict = json.load(fp)


    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path + "model_{}_{}_lite.tflite".format(modelId,feature))
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = xInference.reshape(input_shape).astype(np.float32)

    num_samples = 1

    all_outputs = np.zeros((num_samples,input_shape[1],input_shape[2]))

    for i in range(num_samples):

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index']).reshape(input_shape)

        all_outputs[i,:,:] = output_data

    print(input_data.shape)

    input_data = np.repeat(input_data,num_samples,axis=0)

    #print(all_outputs.shape)

    #mse = 1 / num_samples * np.sum((all_outputs - input_data)**2,axis=1)

    #print(mse.shape)

    #mse = mse.reshape(int(mse.shape[0] / num_samples),num_samples)

    #print(mse.shape)

    mse = keras.metrics.mean_squared_error(all_outputs,input_data)
    means = np.mean(mse,axis=1)
    means = np.mean(means)

    #means = np.mean(mse,axis=1).flatten()
    variances = np.var(mse,axis=1).flatten()

    #print(means)

    #zMeans = means
    zStds = variances
    zMeans = (means - float(param_dict['avgMean'])) / float(param_dict['avgStd'])
    #zStds = (variances - float(param_dict['varMean'])) / float(param_dict['varStd'])


    output = {
        'valueMean':zMeans.tolist(),
        'valueStd':zStds.tolist(),
        'dataItemId':dataItemId,
        'state':spindleSpeed,
        'modelId':modelId,
        'feature':feature
    }


    return jsonify(output), 201


if __name__ == '__main__':
    app.run(debug=True)
