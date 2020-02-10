#!flask/bin/python
from flask import Flask, jsonify
from flask import request
import os

from Perform_FFT import parse_data

app = Flask(__name__)

import numpy as np
import tflite_runtime.interpreter as tflite

@app.route('/features/parse/vibration',methods=['POST'])
def parse_vibration():
    values = np.array(request.json['values']).astype(float)
    fftPoints = request.json['fftPoints']
    samplingInterval = request.json['samplingInterval']

    output = parse_data(values,fftPoints,samplingInterval)

    return jsonify(output), 201

@app.route('/models/inference',methods=['POST'])
def model_inference():

    assetId = request.json['assetId']
    dataItemId = request.json['dataItemId']
    isWarmUp = request.json['isWarmUp']
    spindleSpeed = request.json['spindleSpeed']
    xInference = np.array(request.json['xInference'])
    basePath = request.json['basePath']

    model_path = basePath + 'Models/' + assetId + '/' + dataItemId + '/' + str(isWarmUp).lower() + '/' + str(spindleSpeed) + '/'

    if not os.path.exists(model_path):
        return jsonify({'output':False}),201

    with open(model_path + 'control_params.json', 'r') as fp:
        param_dict = json.load(fp)


    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path + "model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = xInference.reshape(input_shape)

    num_samples = 100

    for i in range(num_samples):

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index']).reshape(512,1)

        if i == 0:
            all_outputs = output_data
        else:
            all_outputs = np.vstack((all_outputs,output_data))

    mse = 1 / num_samples * np.sum((all_outputs - input_data)**2,axis=1)
    mse = mse.reshape(int(mse.shape[0] / num_samples),num_samples)

    means = np.mean(mse,axis=1).flatten().tolist()
    variances = np.var(mse,axis=1).flatten().tolist()

    zMeans = (means - param_dict['avgMean']) / param_dict['avgStd']
    zStds = (variances - param_dict['varMean']) / param_dict['varStd']

    output = {
        'means':zMeans,
        'variances':zStds
    }


    return jsonify(output), 201


if __name__ == '__main__':
    app.run(debug=False)
