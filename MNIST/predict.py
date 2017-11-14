import numpy as np
import pickle
from convnet import *



## Hyperparameters
IMG_WIDTH = 28
IMG_DEPTH = 1
PICKLE_FILE = 'trained.pickle'

## Data extracting
m =10000
X = extract_data('t10k-images-idx3-ubyte.gz', m, IMG_WIDTH)
y = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
X-= int(np.mean(X))
X/= int(np.std(X))


## Opening the saved model parameter
pickle_in = open(PICKLE_FILE, 'rb')
out = pickle.load(pickle_in)
[filt1, filt2, bias1, bias2, theta3, bias3, cost, acc] = out


for i in range(20,50):
	image = X[i].reshape(IMG_DEPTH, IMG_WIDTH, IMG_WIDTH)
	digit, prob = predict(image, filt1, filt2, bias1, bias2, theta3, bias3)
	print digit, prob, y[i][0]
