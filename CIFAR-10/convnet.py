import numpy as np
import gzip

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
	(l, w, w) = X.shape
	pool = np.zeros((l, (w-f)/s+1,(w-f)/s+1))
	for jj in range(0,l):
		i=0
		while(i<w):
			j=0
			while(j<w):
				pool[jj,i/2,j/2] = np.max(X[jj,i:i+f,j:j+f])
				j+=s
			i+=s
	return pool

def softmax_cost(out,y):
	eout = np.exp(out, dtype=np.float128)
	probs = eout/sum(eout)
	
	p = sum(y*probs)
	cost = -np.log(p)	## (Only data loss. No regularised loss)
	return cost,probs	


## Returns gradient for all the paramaters in each iteration
def ConvNet(image, label, filt1, filt2, bias1, bias2, theta3, bias3):
	#####################################################################################################################
	#######################################  Feed forward to get all the layers  ########################################
	#####################################################################################################################

	## Calculating first Convolution layer
		
	## l - channel
	## w - size of square image
	## l1 - No. of filters in Conv1
	## l2 - No. of filters in Conv2
	## w1 - size of image after conv1
	## w2 - size of image after conv2
	(l, w, w) = image.shape		
	l1 = len(filt1)
	l2 = len(filt2)
	( _, f, f) = filt1[0].shape
	w1 = w-f+1
	w2 = w1-f+1
	
	conv1 = np.zeros((l1,w1,w1))
	conv2 = np.zeros((l2,w2,w2))

	for jj in range(0,l1):
		for x in range(0,w1):
			for y in range(0,w1):
				conv1[jj,x,y] = np.sum(image[:,x:x+f,y:y+f]*filt1[jj])+bias1[jj]
	conv1[conv1<=0] = 0 #relu activation

	## Calculating second Convolution layer
	for jj in range(0,l2):
		for x in range(0,w2):
			for y in range(0,w2):
				conv2[jj,x,y] = np.sum(conv1[:,x:x+f,y:y+f]*filt2[jj])+bias2[jj]
	conv2[conv2<=0] = 0 # relu activation

	## Pooled layer with 2*2 size and stride 2,2
	pooled_layer = maxpool(conv2, 2, 2)	

	fc1 = pooled_layer.reshape(((w2/2)*(w2/2)*l2,1))
	
	out = theta3.dot(fc1) + bias3	#10*1
	
	######################################################################################################################
	########################################  Using softmax function to get cost  ########################################
	######################################################################################################################
	cost, probs = softmax_cost(out, label)
	if np.argmax(out)==np.argmax(label):
		acc=1
	else:
		acc=0
	#######################################################################################################################
	##########################  Backpropagation to get gradient	using chain rule of differentiation  ######################
	#######################################################################################################################
	dout = probs - label	#	dL/dout
	
	dtheta3 = dout.dot(fc1.T) 		##	dL/dtheta3

	dbias3 = sum(dout.T).T.reshape((10,1))		##	dbias3	

	dfc1 = theta3.T.dot(dout)		##	dL/dfc1

	dpool = dfc1.T.reshape((l2, w2/2, w2/2))

	dconv2 = np.zeros((l2, w2, w2))
	
	for jj in range(0,l2):
		i=0
		while(i<w2):
			j=0
			while(j<w2):
				(a,b) = nanargmax(conv2[jj,i:i+2,j:j+2]) ## Getting indexes of maximum value in the array
				dconv2[jj,i+a,j+b] = dpool[jj,i/2,j/2]
				j+=2
			i+=2
	
	dconv2[conv2<=0]=0

	dconv1 = np.zeros((l1, w1, w1))
	dfilt2 = {}
	dbias2 = {}
	for xx in range(0,l2):
		dfilt2[xx] = np.zeros((l1,f,f))
		dbias2[xx] = 0

	dfilt1 = {}
	dbias1 = {}
	for xx in range(0,l1):
		dfilt1[xx] = np.zeros((l,f,f))
		dbias1[xx] = 0

	for jj in range(0,l2):
		for x in range(0,w2):
			for y in range(0,w2):
				dfilt2[jj]+=dconv2[jj,x,y]*conv1[:,x:x+f,y:y+f]
				dconv1[:,x:x+f,y:y+f]+=dconv2[jj,x,y]*filt2[jj]
		dbias2[jj] = np.sum(dconv2[jj])
	dconv1[conv1<=0]=0
	for jj in range(0,l1):
		for x in range(0,w1):
			for y in range(0,w1):
				dfilt1[jj]+=dconv1[jj,x,y]*image[:,x:x+f,y:y+f]

		dbias1[jj] = np.sum(dconv1[jj])

	
	return [dfilt1, dfilt2, dbias1, dbias2, dtheta3, dbias3, cost, acc]


def initialize_param(f, l):
	return 0.01*np.random.rand(l, f, f)

def initialize_theta(NUM_OUTPUT, l_in):
	return 0.01*np.random.rand(NUM_OUTPUT, l_in)

def initialise_param_lecun_normal(FILTER_SIZE, IMG_DEPTH, scale=1.0, distribution='normal'):
	
    if scale <= 0.:
            raise ValueError('`scale` must be a positive float. Got:', scale)

    distribution = distribution.lower()
    if distribution not in {'normal'}:
        raise ValueError('Invalid `distribution` argument: '
                             'expected one of {"normal", "uniform"} '
                             'but got', distribution)

    scale = scale
    distribution = distribution
    fan_in = FILTER_SIZE*FILTER_SIZE*IMG_DEPTH
    scale = scale
    stddev = scale * np.sqrt(1./fan_in)
    shape = (IMG_DEPTH,FILTER_SIZE,FILTER_SIZE)
    return np.random.normal(loc = 0,scale = stddev,size = shape)

## Returns all the trained parameters
def momentumGradDescent(batch, LEARNING_RATE, w, l, MU, filt1, filt2, bias1, bias2, theta3, bias3, cost, acc):
	#	Momentum Gradient Update
	# MU=0.5	
	X = batch[:,0:-1]
	X = X.reshape(len(batch), l, w, w)
	y = batch[:,-1]

	n_correct=0
	cost_ = 0
	batch_size = len(batch)
	dfilt2 = {}
	dfilt1 = {}
	dbias2 = {}
	dbias1 = {}
	v1 = {}
	v2 = {}
	bv1 = {}
	bv2 = {}
	for k in range(0,len(filt2)):
		dfilt2[k] = np.zeros(filt2[0].shape)
		dbias2[k] = 0
		v2[k] = np.zeros(filt2[0].shape)
		bv2[k] = 0
	for k in range(0,len(filt1)):
		dfilt1[k] = np.zeros(filt1[0].shape)
		dbias1[k] = 0
		v1[k] = np.zeros(filt1[0].shape)
		bv1[k] = 0
	dtheta3 = np.zeros(theta3.shape)
	dbias3 = np.zeros(bias3.shape)
	v3 = np.zeros(theta3.shape)
	bv3 = np.zeros(bias3.shape)



	for i in range(0,batch_size):
		
		image = X[i]

		label = np.zeros((theta3.shape[0],1))
		label[int(y[i]),0] = 1
		
		## Fetching gradient for the current parameters
		[dfilt1_, dfilt2_, dbias1_, dbias2_, dtheta3_, dbias3_, curr_cost, acc_] = ConvNet(image, label, filt1, filt2, bias1, bias2, theta3, bias3)
		for j in range(0,len(filt2)):
			dfilt2[j]+=dfilt2_[j]
			dbias2[j]+=dbias2_[j]
		for j in range(0,len(filt1)):
			dfilt1[j]+=dfilt1_[j]
			dbias1[j]+=dbias1_[j]
		dtheta3+=dtheta3_
		dbias3+=dbias3_

		cost_+=curr_cost
		n_correct+=acc_

	for j in range(0,len(filt1)):
		v1[j] = MU*v1[j] -LEARNING_RATE*dfilt1[j]/batch_size
		filt1[j] += v1[j]
		# filt1[j] -= LEARNING_RATE*dfilt1[j]/batch_size
		bv1[j] = MU*bv1[j] -LEARNING_RATE*dbias1[j]/batch_size
		bias1[j] += bv1[j]
	for j in range(0,len(filt2)):
		v2[j] = MU*v2[j] -LEARNING_RATE*dfilt2[j]/batch_size
		filt2[j] += v2[j]
		# filt2[j] += -LEARNING_RATE*dfilt2[j]/batch_size
		bv2[j] = MU*bv2[j] -LEARNING_RATE*dbias2[j]/batch_size
		bias2[j] += bv2[j]
	v3 = MU*v3 - LEARNING_RATE*dtheta3/batch_size
	theta3 += v3
	# theta3 += -LEARNING_RATE*dtheta3/batch_size
	bv3 = MU*bv3 -LEARNING_RATE*dbias3/batch_size
	bias3 += bv3

	cost_ = cost_/batch_size
	cost.append(cost_)
	accuracy = float(n_correct)/batch_size
	acc.append(accuracy)

	return [filt1, filt2, bias1, bias2, theta3, bias3, cost, acc]

## Predict class of each row of matrix X
def predict(image, label, filt1, filt2, bias1, bias2, theta3, bias3):
	(l,w,w)=image.shape
	(l1,f,f) = filt2[0].shape
	l2 = len(filt2)
	w1 = w-f+1
	w2 = w1-f+1
	conv1 = np.zeros((l1,w1,w1))
	conv2 = np.zeros((l2,w2,w2))
	for jj in range(0,l1):
		for x in range(0,w1):
			for y in range(0,w1):
				conv1[jj,x,y] = np.sum(image[:,x:x+f,y:y+f]*filt1[jj])+bias1[jj]
	conv1[conv1<=0] = 0 #relu activation
	## Calculating second Convolution layer
	for jj in range(0,l2):
		for x in range(0,w2):
			for y in range(0,w2):
				conv2[jj,x,y] = np.sum(conv1[:,x:x+f,y:y+f]*filt2[jj])+bias2[jj]
	conv2[conv2<=0] = 0 # relu activation
	## Pooled layer with 2*2 size and stride 2,2
	pooled_layer = maxpool(conv2, 2, 2)	
	fc1 = pooled_layer.reshape(((w2/2)*(w2/2)*l2,1))
	out = theta3.dot(fc1) + bias3	#10*1
	return np.argmax(out)

## Get the data from the file
def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict
