import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
import time
import random
from convnet import *

#####################################################################################################################################
################################################## ------ START HERE --------  ######################################################
##################################### ---------- CONVOLUTIONAL NEURAL NETWORK ---------------  ######################################
################ ----ARCHITECTURE PROPOSED : [INPUT - CONV1 - RELU - CONV2 - RELU- MAXPOOL - FC1 - OUT]---- #########################
#####################################################################################################################################


## Hyperparameters
NUM_OUTPUT = 10
LEARNING_RATE = 0.01	#learning rate
IMG_WIDTH = 32
IMG_DEPTH = 3
FILTER_SIZE=5
NUM_FILT1 = 16
NUM_FILT2 = 16
BATCH_SIZE = 8
NUM_EPOCHS = 5	 # number of iterations
MU = 0.95
PICKLE_FILE = 'output.pickle'
# PICKLE_FILE = 'trained.pickle'


data_dash = unpickle('data_batch_1')
X= data_dash['data']	# m * n
## Normalization
X-= int(np.mean(X))
X/= int(np.std(X))
(m,n) = X.shape
y_dash = np.array(data_dash['labels']).reshape((m,1))	# m * 1
## Data preprocessing
data = np.hstack((X,y_dash))
np.random.shuffle(data)
train_data = data[0:int(len(data)*0.9),:]
test_data = data[-int(len(data)*0.1):,:]



NUM_IMAGES = train_data.shape[0]

## Initializing all the parameters
filt1 = {}
filt2 = {}
bias1 = {}
bias2 = {}
for i in range(0,NUM_FILT1):
	filt1[i] = initialise_param_lecun_normal(FILTER_SIZE,IMG_DEPTH)
	bias1[i] = 0
	# v1[i] = 0
for i in range(0,NUM_FILT2):
	filt2[i] = initialise_param_lecun_normal(FILTER_SIZE,NUM_FILT1)
	bias2[i] = 0
	# v2[i] = 0
w1 = IMG_WIDTH-FILTER_SIZE+1
w2 = w1-FILTER_SIZE+1
theta3 = initialize_theta(NUM_OUTPUT, (w2/2)*(w2/2)*NUM_FILT2)

bias3 = np.zeros((NUM_OUTPUT,1))
cost = []
acc = []
# pickle_in = open(PICKLE_FILE, 'rb')
# out = pickle.load(pickle_in)

# [filt1, filt2, bias1, bias2, theta3, bias3, cost, acc] = out


print("Learning Rate:"+str(LEARNING_RATE)+", Batch Size:"+str(BATCH_SIZE))
# Training start here

for epoch in range(0,NUM_EPOCHS):
	np.random.shuffle(train_data)
	batches = [train_data[k:k + BATCH_SIZE] for k in xrange(0, NUM_IMAGES, BATCH_SIZE)]
	print(batches[0].shape)
	x=0
	for batch in batches:
		stime = time.time()
		# LEARNING_RATE =  LEARNING_RATE/(1+epoch/10.0)
		out = momentumGradDescent(batch, LEARNING_RATE, IMG_WIDTH, IMG_DEPTH, MU, filt1, filt2, bias1, bias2, theta3, bias3, cost, acc)
		[filt1, filt2, bias1, bias2, theta3, bias3, cost, acc] = out
		epoch_acc = round(np.sum(acc[epoch*NUM_IMAGES/BATCH_SIZE:])/(x+1),2)
		per = float(x+1)/len(batches)*100
		print("Epoch:"+str(round(per,2))+"% Of "+str(epoch+1)+"/"+str(NUM_EPOCHS)+", Cost:"+str(cost[-1])+", B.Acc:"+str(acc[-1]*100)+", E.Acc:"+str(epoch_acc))
		ftime = time.time()
		deltime = ftime-stime
		remtime = (len(batches)-x-1)*deltime+deltime*len(batches)*(NUM_EPOCHS-epoch-1)
		hrs = int(remtime)/3600
		mins = int((remtime/60-hrs*60))
		secs = int(remtime-mins*60-hrs*3600)
		print(str(int(deltime))+"secs/batch : ########  "+str(hrs)+"Hrs "+str(mins)+"Mins "+str(secs)+"Secs remaining  ########")
		x+=1

	
	
## saving the trained model parameters
with open(PICKLE_FILE, 'wb') as file:
	pickle.dump(out, file)

# Opening the saved model parameter
pickle_in = open(PICKLE_FILE, 'rb')
out = pickle.load(pickle_in)

[filt1, filt2, bias1, bias2, theta3, bias3, cost, acc] = out

plt.figure(0)
plt.plot(cost)
plt.ylabel('Cost')
plt.xlabel('iteration')
plt.figure(1)
plt.plot(acc, color='r')
plt.ylabel('Accuracy')
plt.xlabel('iteration')
plt.show()

X = test_data[:,0:-1]
X = X.reshape(len(test_data), IMG_DEPTH, IMG_WIDTH, IMG_WIDTH)
y = test_data[:,-1]
corr = 0
print("Computing accuracy over test set:")
for i in range(0,len(test_data)):
	image = X[i]
	label = np.zeros((theta3.shape[0],1))
	label[int(y[i]),0] = 1
	if predict(image, label, filt1, filt2, bias1, bias2, theta3, bias3)==y[i]:
		corr+=1
	if (i+1)%int(0.01*len(test_data))==0:
		print(str(float(i+1)/len(test_data)*100)+"% Completed")
test_acc = float(corr)/len(test_data)*100
print("Test Set Accuracy:"+str(test_acc))