import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
import time

## Returns idexes of maximum value of the array
def nanargmax(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
        # In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx

## Activation function
def relu(x):
	return max(0, x)

def maxpool(X, f, s):
	(m, l, w, w) = X.shape
	pool = np.zeros((m, l, (w-f)/s+1,(w-f)/s+1))
	for ii in range(0,m):
		for jj in range(0,l):
			i=0
			while(i<w):
				j=0
				while(j<w):
					pool[ii,jj,i/2,j/2] = np.max(X[ii,jj,i:i+f,j:j+f])
					j+=s
				i+=s
	return pool

## Returns gradient for all the paramaters in each iteration
def getgrad(X, y, f1, f2, s1, s2, l1, l2, param1, param2, bias1, bias2, theta3, bias3, reg):
	#####################################################################################################################
	#######################################  Feed forward to get all the layers  ########################################
	#####################################################################################################################
	# print("param1[2]")
	# print(param1[2])

	## Calculating first Convolution layer
	(m, l, w, w) = X.shape
	conv1 = np.zeros((m,l1,w,w))
	for ii in range(0,m):
		image = X[ii,:,:,:]
		for jj in range(0,l1):
			for i in range(f1/2,w-f1/2):
				for j in range(f1/2, w-f1/2):
					conv1[ii,jj,i,j] = np.sum(param1[jj]*image[:,i-f1/2:i+f1/2+1, j-f1/2:j+f1/2+1]) + bias1[jj]
					conv1[ii,jj,i,j] = relu(conv1[ii,jj,i,j])
	# print("conv1[0,2,:,:]")
	# print(conv1[0,2,:,:])

	## Calculating second Convolution layer
	conv2 = np.zeros((m,l2,w,w))
	for ii in range(0,m):
		image = conv1[ii,:,:,:]
		for jj in range(0,l2):
			for i in range(f2/2,w-f2/2):
				for j in range(f2/2, w-f2/2):
					conv2[ii,jj,i,j] = np.sum(param2[jj]*image[:,i-f2/2:i+f2/2+1, j-f2/2:j+f2/2+1]) + bias2[jj]
					conv2[ii,jj,i,j] = relu(conv2[ii,jj,i,j])
	# print("conv2[0,2,:,:]")
	# print(conv2[0,2,:,:])

	## Pooled layer with 2*2 size and stride 2, 2
	pooled_layer = maxpool(conv2, 2, 2)
	# print("pooled_layer[0,2,:,:]")
	# print(pooled_layer[0,2,:,:])	
	fc1 = (pooled_layer.reshape((m,(w/2)*(w/2)*l2))).T
	# print("fc1[40:50,1]")
	# print(fc1[40:50,1])
	out = theta3.dot(fc1) + bias3	#10*m
	# print("out[:,0]")
	# print(out[:,0])
	# print("theta3[:,0]")
	# print(theta3[:,0])
	# print("bias3")
	# print(bias3)
	######################################################################################################################
	########################################  Using softmax function to get cost  ########################################
	######################################################################################################################
	eout = np.exp(out, dtype=np.float128)
	probs = eout/sum(eout)
	p = sum(y*probs)
	# print("probs[:,0]")
	# print(probs[:,0])
	cost = np.sum(-np.log(p))/m ## (Only data loss. No regularised loss)
	# print("Cost without reg:"+str(cost))
	reg_loss = 0.5*reg*(np.sum(theta3*theta3))
	for i in range(0,l1):
		reg_loss += 0.5*reg*(np.sum(param1[i]*param1[i]))
	for i in range(0,l2):
		reg_loss += 0.5*reg*(np.sum(param2[i]*param2[i]))
	cost += reg_loss
	# print("Cost with reg:"+str(cost))
	#######################################################################################################################
	##########################  Backpropagation to get gradient	using chain rule of differentiation  ######################
	#######################################################################################################################
	dout = probs - y	#	dL/dout
	# print(probs)
	# print(dout)
	dtheta3 = dout.dot(fc1.T) + reg*theta3 		##	dL/dtheta3
	# print(dtheta3[0,20:30])
	dbias3 = sum(dout.T).T.reshape((10,1))		##	dbias3	
	dfc1 = theta3.T.dot(dout)		##	dL/dfc1
	dpool = dfc1.T.reshape((m, l2, w/2, w/2))
	# print(dpool[0])
	dconv2 = np.zeros((m, l2, w, w))
	for ii in range(0,m):
		for jj in range(0,l):
			i=0
			while(i<w):
				j=0
				while(j<w):
					(a,b) = nanargmax(conv2[ii,jj,i:i+2,j:j+2]) ## Getting indexes of maximum value in the array
					dconv2[ii,jj,i+a,j+b] = dpool[ii,jj,i/2,j/2]
					# print(i+a,j+b)
					j+=2
				i+=2
	# print(dconv2[0,0,10:14,10:14])
	dparam2 = {}
	dbias2 = {}
	dconv1 = np.zeros((m,l1,w,w))
	for ii in range(0,m):
		image = conv1[ii,:,:,:]
		for jj in range(0,l2):
			dparam2[jj] = reg*param2[jj]	#np.zeros((l1,f2,f2))
			dbias2[jj] = 0
			for i in range(f2/2,w-f2/2):
				for j in range(f2/2, w-f2/2):
					if conv2[ii,jj,i,j]>0:
						dparam2[jj] += dconv2[ii,jj,i,j]*image[:,i-f2/2:i+f2/2+1, j-f2/2:j+f2/2+1]
						dbias2[jj] += dconv2[ii,jj,i,j]
						dconv1[ii,:,i-f2/2:i+f2/2+1, j-f2/2:j+f2/2+1] += dconv2[ii,jj,i,j]*param2[jj]
	dparam1 = {}
	dbias1 = {}
	for ii in range(0,m):
		image = X[ii,:,:,:]
		for jj in range(0,l1):
			dparam1[jj] = reg*param1[jj]	#np.zeros((3,f1,f1))
			dbias1[jj] = 0
			for i in range(f1/2,w-f1/2):
				for j in range(f1/2, w-f1/2):
					if conv1[ii,jj,i,j]>0:
							dparam1[jj] += dconv1[ii,jj,i,j]*image[:,i-f1/2:i+f1/2+1, j-f1/2:j+f1/2+1]
							dbias1[jj] += dconv1[ii,jj,i,j]
	# print("dparam1[2]")
	# print(dparam1[2])
	return [dparam1, dparam2, dbias1, dbias2, dtheta3, dbias3, cost]

		
def initialize_param(f, l):
	return 0.001*np.random.rand(l, f, f)

def initialize_theta(l_out, l_in):
	return 0.001*np.random.rand(l_out, l_in)

## Returns all the trained parameters
def cnn_fit(X, y, f1, f2, s1, s2, l1, l2, l_out, alpha, num_iter, reg, param1, param2, bias1, bias2, theta3, bias3, cost):
	#	Momentum Gradient Update
	# mu=0.5
	
	for i in range(0,num_iter):
		# alpha = alpha0/(1+i/2.0)
		# print(theta3[0,20:30])
		# print(param1[0])
		# print(bias1)

		## Fetching gradient for the current parameters
		[dparam1, dparam2, dbias1, dbias2, dtheta3, dbias3, curr_cost] = getgrad(X, y, f1, f2, s1, s2, l1, l2, param1, param2, bias1, bias2, theta3, bias3, reg)
		cost.append([curr_cost])
		# print(dtheta3[0,20:30])
		# print(dparam1[0])
		# print(dbias1)

		## Updating Parameters
		for j in range(0,l1):
			# v1[j] = mu*v1[j] -alpha*dparam1[j]
			# param1[j] += v1[j]
			param1[j] += -alpha*dparam1[j]
			bias1[j] += -alpha*dbias1[j]
		for j in range(0,l2):
			# v2[j] = mu*v2[j] -alpha*dparam2[j]
			# param2[j] += v2[j]
			param2[j] += -alpha*dparam2[j]
			bias2[j] += -alpha*dbias2[j]
		# v3 = mu*v3 - alpha*dtheta3
		# theta3 += v3
		theta3 += -alpha*dtheta3
		bias3 += -alpha*dbias3
		# printing the status
		# if (i+1)%(num_iter*0.1) == 0:

	return [param1, param2, bias1, bias2, theta3, bias3, cost]

## Predict class of each row of matrix X
def predict(X, param1, param2, bias1, bias2, theta3, bias3, f1, f2, s1, s2, l1, l2, l_out):	
	(m, l, w, w) = X.shape
	conv1 = np.zeros((m,l1,w,w))
	for ii in range(0,m):
		image = X[ii,:,:,:]
		for jj in range(0,l1):
			for i in range(f1/2,w-f1/2):
				for j in range(f1/2, w-f1/2):
					conv1[ii,jj,i,j] = np.sum(param1[jj]*image[:,i-f1/2:i+f1/2+1, j-f1/2:j+f1/2+1]) + bias1[jj]
					conv1[ii,jj,i,j] = relu(conv1[ii,jj,i,j])
	conv2 = np.zeros((m,l2,w,w))
	for ii in range(0,m):
		image = conv1[ii,:,:,:]
		for jj in range(0,l2):
			for i in range(f2/2,w-f2/2):
				for j in range(f2/2, w-f2/2):
					conv2[ii,jj,i,j] = np.sum(param2[jj]*image[:,i-f2/2:i+f2/2+1, j-f2/2:j+f2/2+1]) + bias2[jj]
					conv2[ii,jj,i,j] = relu(conv2[ii,jj,i,j])

	pooled_layer = maxpool(conv2, 2, 2)
	fc1 = (pooled_layer.reshape((m,(w/2)*(w/2)*l2))).T
	out = theta3.dot(fc1) + bias3	#10*m
	probs = np.exp(out)/sum(np.exp(out))
	h = np.empty((1,m))
	for i in range(0,m):
		h[0,i] = np.argmax(probs[:,i])
	return h

## Get the data from the file
def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

#####################################################################################################################################
################################################## ------ START HERE --------  ######################################################
##################################### ---------- CONVOLUTIONAL NEURAL NETWORK ---------------  ######################################
################ ----ARCHITECTURE PROPOSED : [INPUT - CONV1 - RELU - CONV2 - RELU- MAXPOOL - FC1 - OUT]---- #########################
#####################################################################################################################################


## Hyperparameters
l_out = 10
alpha = 0.0001	#learning rate
w = 32
f1 = 5
f2 = 5
s1 = 1
s2 = 1
l1 = 8
l2 = 8
num_iter = 15
reg = 10



## Data preprocessing
mat1 = unpickle('test_batch')

X_full = mat1['data']	# m * n
(m, n) = X_full.shape
idx = np.random.randint(m, size=1000)
X = X_full[idx,:]
(m, n) = X.shape
X = X.reshape(m, 3, w, w)

y_dash = np.array(mat1['labels'])	# m * 1
y_dash = y_dash[idx]
y_dash = y_dash.reshape(1, m)

print(X.shape)
print(y_dash.shape)
y = np.zeros((l_out,m))	# l2 * m
for i in range(0,m):
	y[y_dash[0,i], i]= 1


## Initializing all the parameters

batch = 50

param1 = {}
param2 = {}
bias1 = {}
bias2 = {}
for i in range(0,l1):
	param1[i] = initialize_param(f1,3)
	bias1[i] = 0
	# v1[i] = 0
for i in range(0,l2):
	param2[i] = initialize_param(f2,l1)
	bias2[i] = 0
	# v2[i] = 0
theta3 = initialize_theta(l_out, (w/2)*(w/2)*l2)
bias3 = np.zeros((l_out,1))
cost = []
epoch = m/batch

## Training start here
for x in range(0,epoch):
	out = cnn_fit(X[x*batch:(x+1)*batch], y[:,x*batch:(x+1)*batch], f1, f2, s1, s2, l1, l2, l_out, alpha, num_iter, reg, param1, param2, bias1, bias2, theta3, bias3, cost)
	[param1, param2, bias1, bias2, theta3, bias3, cost] = out
	per = float(x+1)/epoch*100
	print("Epoch:"+str(x+1)+"/"+str(epoch)+", "+str(per)+"% Completed, Cost:"+str(cost[num_iter*(x+1)-1]))

## saving the trained model parameters
with open('output.pickle', 'wb') as f:
	pickle.dump(out, f)

## Opening the saved model parameter
# pickle_in = open('output.pickle', 'rb')
# out = pickle.load(pickle_in)

[param1, param2, bias1, bias2, theta3, bias3, cost] = out
output = predict(X, param1, param2, bias1, bias2, theta3, bias3, f1, f2, s1, s2, l1, l2, l_out)


acc = np.mean((output==y_dash)*np.ones(output.shape))*100
print("With Learning Rate:"+str(alpha)+", Strides:"+str(s1)+", Filter Size:"+str(f1)+", L1:"+str(l1)+", L2:"+str(l2)+"Num_iter:"+str(num_iter))
print("Accuracy:"+str(acc))

plt.plot(cost)
plt.ylabel('cost')
plt.xlabel('iteration')
plt.show()
