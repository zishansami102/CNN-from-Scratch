from flask import Flask, jsonify
import numpy as np

app = Flask(__name__)
@app.route('/satya')
def index():
	sendreco={'class':5,'probability':0.82}
	return jsonify(sendreco)

if __name__ == "__main__":
	app.run()