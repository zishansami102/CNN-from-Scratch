from flask import Flask, jsonify, render_template, request, flash, logging, session, redirect, url_for
import numpy as np

app = Flask(__name__)
@app.route('/')
def index():

	return render_template('index.html')

if __name__ == "__main__":
	app.run(debug=True)
