#!flask/bin/python
from flask import Flask, jsonify
from flask import request
import os
import binascii

import json

from Perform_FFT import parse_data
from Create_Data import create_noisy_signal

app = Flask(__name__)

import numpy as np
import tflite_runtime.interpreter as tflite
import tensorflow as tf

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
    phase = 0

    signal = create_noisy_signal(duration,samplingRate,frequencies,amplitudes,noiseStd,phase)
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

    with open(model_path + 'control_params_{}_{}.json'.format(modelId,feature), 'r') as fp:
        param_dict = json.load(fp)


    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path + "model_{}_{}.tflite".format(modelId,feature))
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = xInference.reshape(input_shape).astype(np.float32)

    num_samples = 100

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

    print(all_outputs.shape)

    mse = 1 / num_samples * np.sum((all_outputs - input_data)**2,axis=1)

    print(mse.shape)

    mse = mse.reshape(int(mse.shape[0] / num_samples),num_samples)

    print(mse.shape)

    means = np.mean(mse,axis=1).flatten()
    variances = np.var(mse,axis=1).flatten()

    #zMeans = means
    #zStds = variances
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


if __name__ == '__main__':
    app.run(debug=True)
