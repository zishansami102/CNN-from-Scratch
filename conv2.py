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

def softmax_cost(out,y, theta3, param1, param2):
	eout = np.exp(out, dtype=np.float128)
	probs = eout/sum(eout)
	
	p = sum(y*probs)
	_, m = y.shape

	cost = np.sum(-np.log(p))/m ## (Only data loss. No regularised loss)
	reg_loss = 0.5*reg*(np.sum(theta3*theta3))
	for i in range(0,l1):
		reg_loss += 0.5*reg*(np.sum(param1[i]*param1[i]))
	for i in range(0,l2):
		reg_loss += 0.5*reg*(np.sum(param2[i]*param2[i]))
	cost += reg_loss
	return cost,probs

def rotate180(filt):
	(l,f,f) = filt.shape
	filt2 = np.zeros((l,f,f))
	for xx in range(0,l):
		filt2[xx]=(np.rot90(filt[xx],2))
	return filt2
## Returns gradient for all the paramaters in each iteration
def getgrad(X, Y, f, l1, l2, param1, param2, bias1, bias2, theta3, bias3, reg):
	#####################################################################################################################
	#######################################  Feed forward to get all the layers  ########################################
	#####################################################################################################################

	## Calculating first Convolution layer
	(m,l,w,w)=X.shape
	w1 = w-f+1
	conv1 = np.zeros((m,l1,w1,w1))
	for ii in range(0,m):
		im = X[ii]
		for jj in range(0,l1):
			filt = rotate180(param1[jj])
			for x in range(0,w1):
				for y in range(0,w1):
					conv1[ii,jj,x,y] = np.sum(im[:,x:x+f,y:y+f]*filt)+bias1[jj]
	conv1[conv1<=0] = 0 #relu activation

	## Calculating second Convolution layer
	w2 = w1-f+1
	conv2 = np.zeros((m,l2,w2,w2))
	for ii in range(0,m):
		im = conv1[ii]
		for jj in range(0,l2):
			filt = rotate180(param2[jj])
			for x in range(0,w2):
				for y in range(0,w2):
					conv2[ii,jj,x,y] = np.sum(im[:,x:x+f,y:y+f]*filt)+bias2[jj]
	conv2[conv2<=0] = 0 # relu activation

	## Pooled layer with 2*2 size and stride 2,2
	pooled_layer = maxpool(conv2, 2, 2)	
	fc1 = (pooled_layer.reshape((m,(w2/2)*(w2/2)*l2))).T
	out = theta3.dot(fc1) + bias3	#10*m

	######################################################################################################################
	########################################  Using softmax function to get cost  ########################################
	######################################################################################################################
	cost, probs = softmax_cost(out,Y, theta3, param1, param2)
	
	h = np.empty((1,m))
	y_dash = np.empty((1,m))
	for i in range(0,m):
		h[0,i] = np.argmax(probs[:,i])
		y_dash[0,i] = np.argmax(Y[:,i])

	acc = np.mean((h==y_dash)*np.ones(h.shape))
	#######################################################################################################################
	##########################  Backpropagation to get gradient	using chain rule of differentiation  ######################
	#######################################################################################################################
	dout = probs - Y	#	dL/dout
	
	dtheta3 = dout.dot(fc1.T) + reg*theta3 		##	dL/dtheta3

	dbias3 = sum(dout.T).T.reshape((10,1))		##	dbias3	

	dfc1 = theta3.T.dot(dout)		##	dL/dfc1

	dpool = dfc1.T.reshape((m, l2, w2/2, w2/2))

	dconv2 = np.zeros((m, l2, w2, w2))
	for ii in range(0,m):
		for jj in range(0,l2):
			i=0
			while(i<w2):
				j=0
				while(j<w2):
					(a,b) = nanargmax(conv2[ii,jj,i:i+2,j:j+2]) ## Getting indexes of maximum value in the array
					dconv2[ii,jj,i+a,j+b] = dpool[ii,jj,i/2,j/2]
					j+=2
				i+=2
	
	dconv2[conv2<=0]=0

	dconv1 = np.zeros((m, l2, w1, w1))
	dparam2 = {}
	dbias2 = {}
	for xx in range(0,l2):
		dparam2[xx] = np.zeros((l1,f,f))
		dbias2[xx] = 0
	for ii in range(0,m):
		for jj in range(0,l2):
			for zz in range(0,l1):
				filt = param2[jj][zz,:,:]
				for x in range(0,w1):
					for y in range(0,w1):
						if conv1[ii,zz,x,y]>0:
							# print(x,y)
							xs = x-f+1
							ys = y-f+1
							xe = x+1
							ye = y+1
							if xs<0:
								xs=0
							if ys<0:
								ys=0
							if xe>w2:
								xe=w2
							if ye>w2:
								ye=w2
							dim = dconv2[ii,jj,xs:xe,ys:ye]
							# print(filt.shape)
							# print(dim.shape)
							(a,b)=dim.shape
							if xe<=w2 and ye<=w2:
								filt2 = filt[-a:,-b:]
							if ye>w2 and xe<=w2:
								filt2 = filt[-a:,0:b]
							if xe>w2 and ye<=w2:
								filt2 = filt[0:a,-b:]
							if xe>w2 and ye>w2:
								filt2 = filt[0:a,0:b]
							# print(xs,xe)
							# print(ys,ye)
							# print(filt)
							dconv1[ii,zz,x,y]+=np.sum(dim*filt2)
	for jj in range(0,l2):
		for zz in range(0,l1):
			for x in range(0,f):
				for y in range(0,f):
					dparam2[jj][zz,x,y] = reg*param2[jj][zz,x,y]
					for ii in range(0,m):
						dparam2[jj][zz,x,y]+=np.sum(dconv2[ii,jj]*conv1[ii,zz,x:x+w2,y:y+w2])
		for ii in range(0,m):
			dbias2[jj] += np.sum(dconv2[ii,jj])


	dparam1 = {}
	dbias1 = {}
	for xx in range(0,l1):
		dparam1[xx] = np.zeros((3,f,f))
		dbias1[xx] = 0
	
	for jj in range(0,l1):
		for zz in range(0,3):
			for x in range(0,f):
				for y in range(0,f):
					dparam1[jj][zz,x,y] = reg*param1[jj][zz,x,y]
					for ii in range(0,m):
						dparam1[jj][zz,x,y]+=np.sum(dconv1[ii,jj]*X[ii,zz,x:x+w1,y:y+w1])
		for ii in range(0,m):				
			dbias1[jj] += np.sum(dconv1[ii,jj])
	
	return [dparam1, dparam2, dbias1, dbias2, dtheta3, dbias3, cost, acc]


def initialize_param(f, l):
	return 0.001*np.random.rand(l, f, f)

def initialize_theta(l_out, l_in):
	return 0.001*np.random.rand(l_out, l_in)

## Returns all the trained parameters
def cnn_fit(X, y, f, l1, l2, l_out, alpha, num_iter, reg, param1, param2, bias1, bias2, theta3, bias3, cost, acc):
	#	Momentum Gradient Update
	# mu=0.5
	
	for i in range(0,num_iter):
		# alpha = alpha0/(1+i/2.0)

		## Fetching gradient for the current parameters
		[dparam1, dparam2, dbias1, dbias2, dtheta3, dbias3, curr_cost, curr_acc] = getgrad(X, y, f, l1, l2, param1, param2, bias1, bias2, theta3, bias3, reg)
		cost.append([curr_cost])
		acc.append([curr_acc])
		# print(dtheta3[0,20:30])
		# print(dparam1[0])
		# print(dbias1)
		print(str(i+1)+" | Cost::"+str(curr_cost)+" | Accuracy::"+str(curr_acc))
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


	return [param1, param2, bias1, bias2, theta3, bias3, cost, acc]

## Predict class of each row of matrix X
def predict(X, param1, param2, bias1, bias2, theta3, bias3, f, l1, l2, l_out):	
	(m,l,w,w)=X.shape
	w1 = w-f+1
	conv1 = np.zeros((m,l1,w1,w1))
	for ii in range(0,m):
		im = X[ii]
		for jj in range(0,l1):
			filt = rotate180(param1[jj])
			for x in range(0,w1):
				for y in range(0,w1):
					conv1[ii,jj,x,y] = np.sum(im[:,x:x+f,y:y+f]*filt)+bias1[jj]
	conv1[conv1<=0] = 0 #relu activation

	## Calculating second Convolution layer
	w2 = w1-f+1
	conv2 = np.zeros((m,l2,w2,w2))
	for ii in range(0,m):
		im = conv1[ii]
		for jj in range(0,l2):
			filt = rotate180(param2[jj])
			for x in range(0,w2):
				for y in range(0,w2):
					conv2[ii,jj,x,y] = np.sum(im[:,x:x+f,y:y+f]*filt)+bias2[jj]
	conv2[conv2<=0] = 0 # relu activation

	pooled_layer = maxpool(conv2, 2, 2)
	fc1 = (pooled_layer.reshape((m,(w2/2)*(w2/2)*l2))).T
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
alpha = 0.00001	#learning rate
w = 32
f=5
l1 = 16
l2 = 16
num_iter = 4
reg = 0.01
batch = 4


## Data preprocessing
mat1 = unpickle('data_batch_1')

X_full = mat1['data']	# m * n
(m, n) = X_full.shape
idx = np.random.randint(m, size=10000)
X = X_full[idx,:]
(m, n) = X.shape
X = X.reshape(m, 3, w, w)

y_dash = np.array(mat1['labels'])	# m * 1
y_dash = y_dash[idx]
y_dash = y_dash.reshape(1, m)
print("Y mean:"+str(np.mean(y_dash)))
y = np.zeros((l_out,m))	# l2 * m
for i in range(0,m):
	y[y_dash[0,i], i]= 1


## Initializing all the parameters



param1 = {}
param2 = {}
bias1 = {}
bias2 = {}
for i in range(0,l1):
	param1[i] = initialize_param(f,3)
	bias1[i] = 0
	# v1[i] = 0
for i in range(0,l2):
	param2[i] = initialize_param(f,l1)
	bias2[i] = 0
	# v2[i] = 0
w1 = w-f+1
w2 = w1-f+1
theta3 = initialize_theta(l_out, (w2/2)*(w2/2)*l2)
bias3 = np.zeros((l_out,1))
cost = []
acc = []
epoch = m/batch

print("Alpha:"+str(alpha)+", Reg:"+str(reg)+", Batch Size:"+str(batch))
# Training start here
for x in range(0,epoch):
	out = cnn_fit(X[x*batch:(x+1)*batch], y[:,x*batch:(x+1)*batch], f, l1, l2, l_out, alpha, num_iter, reg, param1, param2, bias1, bias2, theta3, bias3, cost, acc)
	[param1, param2, bias1, bias2, theta3, bias3, cost, acc] = out
	per = float(x+1)/epoch*100
	print("Epoch:"+str(x+1)+"/"+str(epoch)+", "+str(per)+"% Completed, Cost:"+str(cost[num_iter*(x+1)-1]))

## saving the trained model parameters
with open('output.pickle', 'wb') as file:
	pickle.dump(out, file)

# Opening the saved model parameter
pickle_in = open('output.pickle', 'rb')
out = pickle.load(pickle_in)

[param1, param2, bias1, bias2, theta3, bias3, cost] = out


plt.plot(cost[2:])
plt.ylabel('cost')
plt.xlabel('iteration')
plt.show()

output = predict(X, param1, param2, bias1, bias2, theta3, bias3, f, l1, l2, l_out)


acc = np.mean((output==y_dash)*np.ones(output.shape))*100
print("With Learning Rate:"+str(alpha)+", Filter Size:"+str(f)+", L1:"+str(l1)+", L2:"+str(l2)+"Num_iter:"+str(num_iter))
print("Accuracy:"+str(acc))