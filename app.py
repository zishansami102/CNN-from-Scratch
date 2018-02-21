from flask import Flask, jsonify, render_template, request, flash, logging, session, redirect, url_for
import numpy as np

app = Flask(__name__)
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/digit_process', methods=['POST'])
def digit_process():
	if(request.method == "POST"):
		img = request.get_json()
		img = np.array(img).reshape(1,28,28)
		print img.shape
		return "Take that"

if __name__ == "__main__":
	app.run(debug=True)
