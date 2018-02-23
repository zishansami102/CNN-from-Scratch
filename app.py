from flask import Flask, jsonify, render_template, request, flash, logging, session, redirect, url_for
import numpy as np
import sys
import pickle
sys.path.insert(0, 'MNIST')

from convnet import predict
from preprocessing import *

app = Flask(__name__)
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/digit_process', methods=['POST'])
def digit_process():
	if(request.method == "POST"):
		img = request.get_json()
		img = preprocess(img)

		PICKLE_FILE = 'MNIST/trained.pickle'
		pickle_in = open(PICKLE_FILE, 'rb')
		out = pickle.load(pickle_in)

		[filt1, filt2, bias1, bias2, theta3, bias3, _, _] = out
		digit, probability = predict(img, filt1, filt2, bias1, bias2, theta3, bias3)
		
		data = { "digit":digit, "probability":np.round(probability,3) }
		print data
		return jsonify(data)

if __name__ == "__main__":
	app.run(debug=True)
