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
n_output = 10
learning_rate = 0.01	#learning rate
img_width = 20
img_depth = 1
filter_size=5
n_filt1 = 8
n_filt2 = 8
batch_size = 20
n_epochs = 5	 # number of iterations
mu = 0.95


data_dash = sio.loadmat('data.mat')
X= data_dash['X']	# m * np
(m,n) = X.shape
y_dash = data_dash['y']	# m * 1
## Data preprocessing

y_dash = y_dash*(y_dash!=10)	# changing value with 10 back to 0.(Digit '0' was represented by '10'. Coverting back to '0' to have simple calculations)

data = np.hstack((X,y_dash))
np.random.shuffle(data)
train_data = data[0:int(len(data)*0.8),:]
test_data = data[-int(len(data)*0.2):,:]
m = train_data.shape[0]
## Initializing all the parameters



filt1 = {}
filt2 = {}
bias1 = {}
bias2 = {}
for i in range(0,n_filt1):
	filt1[i] = initialize_param(filter_size,img_depth)
	bias1[i] = 0
	# v1[i] = 0
for i in range(0,n_filt2):
	filt2[i] = initialize_param(filter_size,n_filt1)
	bias2[i] = 0
	# v2[i] = 0
w1 = img_width-filter_size+1
w2 = w1-filter_size+1
theta3 = initialize_theta(n_output, (w2/2)*(w2/2)*n_filt2)

bias3 = np.zeros((n_output,1))
cost = []
acc = []
# pickle_in = open('output.pickle', 'rb')
# out = pickle.load(pickle_in)

# [filt1, filt2, bias1, bias2, theta3, bias3, cost, acc] = out


print("Learning Rate:"+str(learning_rate)+", Batch Size:"+str(batch_size))
# Training start here

for epoch in range(0,n_epochs):
	np.random.shuffle(train_data)
	batches = [train_data[k:k + batch_size] for k in xrange(0, m, batch_size)]
	x=0
	for batch in batches:
		stime = time.time()
		# learning_rate =  learning_rate/(1+epoch/10.0)
		out = momentumGradDescent(batch, learning_rate, img_width, img_depth, mu, filt1, filt2, bias1, bias2, theta3, bias3, cost, acc)
		[filt1, filt2, bias1, bias2, theta3, bias3, cost, acc] = out
		epoch_acc = round(np.sum(acc[epoch*m/batch_size:])/(x+1),2)
		per = float(x+1)/len(batches)*100
		print("Epoch:"+str(per)+"% Of "+str(epoch+1)+"/"+str(n_epochs)+", Cost:"+str(cost[-1])+", B.Acc:"+str(acc[-1]*100)+", E.Acc:"+str(epoch_acc))
		ftime = time.time()
		deltime = ftime-stime
		remtime = (len(batches)-x-1)*deltime+deltime*len(batches)*(n_epochs-epoch-1)
		hrs = int(remtime)/3600
		mins = int((remtime/60-hrs*60))
		secs = int(remtime-mins*60-hrs*3600)
		print(str(int(deltime))+"secs/batch : ########  "+str(hrs)+"Hrs "+str(mins)+"Mins "+str(secs)+"Secs remaining  ########")
		x+=1

	
	
## saving the trained model parameters
with open('output.pickle', 'wb') as file:
	pickle.dump(out, file)

# Opening the saved model parameter
pickle_in = open('output.pickle', 'rb')
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
X = X.reshape(len(test_data), l, img_width, img_width)
y = test_data[:,-1]
corr = 0
for i in range(0,len(test_data)):
	image = X[i]
	label = np.zeros((theta3.shape[0],1))
	label[int(y[i]),0] = 1
	if predict(image, label, filt1, filt2, bias1, bias2, theta3, bias3)==y[i]:
		corr+=1
test_acc = float(corr)/len(test_data)*100
print("Test Set Accuracy:"+str(test_acc))