# -*- coding: utf-8 -*-
'''
Logistic and softmax regressions for facial expressions
using the CK+ dataset. 
'''

#%% paths and imports

main_dir = "./Dropbox/CK_exploration/"
data_dir = main_dir+"data"
fig_dir  = main_dir+"figures/"
code_dir = main_dir+"code/"

from os import listdir
import os, random, copy
from PIL import Image
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = "paper", style = "white", font = 'serif') 


#%% FUNCTIONS
# -------------------------------------------------------------------------- #
# data pre-processing 
'''
list of face expressions (contempt, neutral are excluded) are:
1. anger
2. disgust
3. fear
4. happiness
5. sadness
6. surprise
'''
def load_data(data_dir="./aligned/"):
	""" Load all PNG images stored in your data directory into a list of NumPy
	arrays. 

	Args:
		data_dir: The relative directory path to the CK+ image directory.
	Returns:
		images: A dictionary with keys as emotions and a list containing images associated with each key.
		cnt: A dictionary that stores the # of images in each emotion
	"""
	images = defaultdict(list)

	# Get the list of emotional directory:
	for e in listdir(data_dir):
		# excluding any non-directory files
		if not os.path.isdir(os.path.join(data_dir, e)):
			continue
		# Get the list of image file names
		all_files = listdir(os.path.join(data_dir, e))

		for file in all_files:
			# Load only image files as PIL images and convert to NumPy arrays
			if '.png' in file:
				img = Image.open(os.path.join(data_dir, e, file))
				images[e].append(np.array(img))

	print("Emotions: {} \n".format(list(images.keys())))

	cnt = defaultdict(int)
	for e in images.keys():
		print("{}: {} # of images".format(e, len(images[e])))
		cnt[e] = len(images[e])
	return images, cnt

def balanced_sampler(dataset, cnt, emotions):
	# this ensures everyone has the same balanced subset for model training, don't change this seed value
	random.seed(20)
	print("\nBalanced Set:")
	min_cnt = min([cnt[e] for e in emotions])
	balanced_subset = defaultdict(list)
	for e in emotions:
		balanced_subset[e] = copy.deepcopy(dataset[e])
		random.shuffle(balanced_subset[e])
		balanced_subset[e] = balanced_subset[e][:min_cnt]
		print('{}: {} # of images'.format(e, len(balanced_subset[e])))
	return balanced_subset

def display_face(img):
	""" Display the input image and optionally save as a PNG.

	Args:
		img: The NumPy array or image to display

	Returns: None
	"""
	# Convert img to PIL Image object (if it's an ndarray)
	if type(img) == np.ndarray:
		print("Converting from array to PIL Image")
		img = Image.fromarray(img)

	# Display the image
	img.show()

# -------------------------------------------------------------------------- #
# principal components analysis (Turk & Pentland 1991)
def PCA(train,k):
    '''
    Principal components analysis for image training data
    INPUT:   train     = training data: dict. with (keys,vals) = (emotions, images)
             k         = number of principal components you want to extract
    OUTPUT:  new_train = (kxM) training data projected onto principal components, 
                         ordered lexicographically first by keys then order
                         within keys
    '''
    # get emotions in training data
    emotions  = [i for i in train if train[i]!=train.default_factory()]
    img_shape = np.array(train[emotions[1]][1]).shape
     # number of arrays per emotions
    n = min([len(train[e]) for e in emotions])
    # total number of arrays
    M = n*len(emotions)
    # generate a d x M array: M images as columns with dimension d
    train = [train[e][i].flatten() for e in emotions for i in np.arange(0,n,1)]
    d = len(train[1]) 
    train = np.transpose(np.array(train))
    #print("your full data matrix has dimensions: " + str(train.shape))
    # average face
    Psi = (1/M)*np.sum(train,axis = 1) 
    Psi = np.reshape(Psi,(d,1))                 # Psi (dx1)
    # display average face
    avg_face = np.reshape(Psi,img_shape)
    #print("look at the average face... handsome right?")
    img = Image.fromarray(avg_face)
    # img.show()
    # centered images
    A = train - np.repeat(Psi,M,axis=1)        # A = [Phi_1,...Phi_M] (dxM)
    L = (1/M)*np.transpose(A).dot(A)                 # L = (MxM)
    # Find eigenvalues and vectors of L = A'A 
    [vals, vecs] = np.linalg.eig(L)
    # sort eigenvectors from greatest to smallest
    eigind = vals.argsort()[::-1]   
    vals = vals[eigind]
    vecs = vecs[:,eigind]
    U = A.dot(vecs)   # (dxM) vec of eigenvectors of C = AA'
    # Sanity check for PCA
    sanity = np.zeros((M,1))
    for ii in range(M):
        sanity[ii] = np.transpose(A[:,ii]).dot(U[:,0])/np.linalg.norm(U[:,0])
    #print("we want the mean of PC projections to be zero: mean = " 
    #      + str(round(np.mean(sanity))))
    #print("and the std. deviation to be: " + str(round(vals[0]**(1/2))) + 
    #      " and it is: " + str(round(np.std(sanity))) )
    # first k principal components
     #v∗ = Av1/(||Av1||λ0.5)
    components = np.zeros((d,k))
    components1 = np.zeros((d,k))
    for kk in range(k):
        components[:,kk] = U[:,kk]/(np.linalg.norm(U[:,kk])*(vals[kk]**(1/2)))
        components1[:,kk] = U[:,kk]
    new_train = np.transpose(np.transpose(A).dot(components)) # dxk
    return new_train, components, components1

# PCA for holdout and test sets    
def PCA_holdout(train,test,k):
    '''
    Principal components analysis for image training data
    INPUT:   train     = training data: dict. with (keys,vals) = (emotions, images)
             k         = number of principal components you want to extract
    OUTPUT:  new_train = (kxM) training data projected onto principal components, 
                         ordered lexicographically first by keys then order
                         within keys
    '''
    # get emotions in training data
    emotions  = [i for i in train if train[i]!=train.default_factory()]
    img_shape = np.array(train[emotions[1]][1]).shape
     # number of arrays per emotions
    n = min([len(train[e]) for e in emotions])
    nn = min([len(test[e]) for e in emotions])
    # total number of arrays
    M = n*len(emotions)
    # generate a d x M array: M images as columns with dimension d
    train = [train[e][i].flatten() for e in emotions for i in np.arange(0,n,1)]
    test  = [test[e][i].flatten() for e in emotions for i in np.arange(0,nn,1)]
    d = len(train[1]) 
    train = np.transpose(np.array(train))
    test = np.transpose(np.array(test))
    #print("your full data matrix has dimensions: " + str(train.shape))
    # average face
    Psi = (1/M)*np.sum(train,axis = 1) 
    Psi = np.reshape(Psi,(d,1))                 # Psi (dx1)
    # display average face
    avg_face = np.reshape(Psi,img_shape)
    #print("look at the average face... handsome right?")
    #img = Image.fromarray(avg_face)
    # img.show()
    # centered images
    A = train - np.repeat(Psi,M,axis=1)        # A = [Phi_1,...Phi_M] (dxM)
    L = (1/M)*np.transpose(A).dot(A)                 # L = (MxM)
    # Find eigenvalues and vectors of L = A'A 
    [vals, vecs] = np.linalg.eig(L)
    # sort eigenvectors from greatest to smallest
    eigind = vals.argsort()[::-1]   
    vals = vals[eigind]
    vecs = vecs[:,eigind]
    U = A.dot(vecs)   # (dxM) vec of eigenvectors of C = AA'
    # Sanity check for PCA
    sanity = np.zeros((M,1))
    for ii in range(M):
        sanity[ii] = np.transpose(A[:,ii]).dot(U[:,0])/np.linalg.norm(U[:,0])
    #print("we want the mean of PC projections to be zero: mean = " 
    #      + str(round(np.mean(sanity))))
    #print("and the std. deviation to be: " + str(round(vals[0]**(1/2))) + 
    #      " and it is: " + str(round(np.std(sanity))) )
    # first k principal components
     #v∗ = Av1/(||Av1||λ0.5)
    components = np.zeros((d,k))
    for kk in range(k):
        components[:,kk] = U[:,kk]/(np.linalg.norm(U[:,kk])*(vals[kk]**(1/2)))
    test = test - np.repeat(Psi, test.shape[1],axis=1)
    new_test = np.transpose(np.transpose(test).dot(components)) # dxk
    return new_test

# -------------------------------------------------------------------------- #
# Cross validation and testing
# train, cross, test (80/10/10) splitter
def splitter1(data): 
    '''
    function to split data into 80/10/10 train-validation-test sets
    '''
    random.seed(20)
    emotions  = [i for i in data if data[i]!=data.default_factory()] 
    n = min([len(data[e]) for e in emotions])
    train = defaultdict(list)
    test  = defaultdict(list)
    cv    = defaultdict(list)
    for e in emotions: 
        seq = np.arange(0,n,1)
        random.shuffle(seq) #shuffle without replacement
        permutation = seq
        ind = np.array([round(.8*n),round(.1*n),round(.1*n)])
        if sum(ind) != n:
            ind[2] = ind[2] + (n-sum(ind))
            tr_ind  = permutation[np.arange(0,ind[0],1)]
            cv_ind  = permutation[np.arange(ind[0],ind[1]+ind[0],1)]
            test_ind = permutation[np.arange(ind[1]+ind[0],ind[2]+ind[1]+ind[0],1)]
            for i in tr_ind:
                train[e].append(data[e][i])
            for i in cv_ind:
                cv[e].append(data[e][i])   
            for i in test_ind:
                test[e].append(data[e][i])
        if e == emotions[1]: 
            print("size of test set: " + str(len(test_ind)) + 
                  ", size of training: " + str(len(tr_ind)) + 
                  ", size of cv set: " + str(len(cv_ind)))
    return [train,cv,test]

# 10 fold cross validation, fr fr
def kfoldcross(k,data,i,emotions,npc):
    '''
    function for 10 fold cross validation
    '''
    def generate_targets(data):
        emo  = [i for i in data if data[i]!=data.default_factory()]
        # for logistic regression
        if len(emo) == 2:
            n0 = len(data[emo[0]])
            n1 = len(data[emo[1]])
            y = np.reshape(np.repeat(0,n0 + n1,axis=0),(n0+n1,1))
            y[n0:n1+n0] = np.reshape(np.repeat(1,n1,axis=0),(n1,1))
        # for softmax
        # if len(emo) > 2: 
        return y 
    folds1 = np.array_split(data[emotions[0]],k)
    folds2 = np.array_split(data[emotions[1]],k)
    val = defaultdict(list)
    test = defaultdict(list)
    train = defaultdict(list)
    idx = list(set(range(0,k)) - set([i,(i+1)%k]))
    val[emotions[0]] = folds1[i]
    test[emotions[0]] = folds1[(i+1)%k]
    train_temp = [folds1[i] for i in idx]
    train[emotions[0]] = [i for sub in train_temp for i in sub]
    val[emotions[1]] = folds2[i]
    test[emotions[1]] = folds2[(i+1)%k]
    train_temp = [folds2[i] for i in idx]
    train[emotions[1]] = [i for sub in train_temp for i in sub]
    [new_train,components,components1] = PCA(train,npc)
    new_val = PCA_holdout(train, val,npc)
    new_test = PCA_holdout(train, test,npc)   
    y_train = generate_targets(train)
    y_cv    = generate_targets(val)
    y_test  = generate_targets(test)
    return new_train,new_val,new_test,y_train,y_cv,y_test

# -------------------------------------------------------------------------- #
# additional functions

#  Batch gradient descent
def batch_descent(tr,tar,M,alpha,cv,test,tar_cv,tar_test):
    '''
    INPUT: tr    = (k x n) training data, 
           tar   = (n x 1) target vector,
           M     =  number of epochs,
           alpha = learning rate
    OUTPUT: w    = trained weights after M epochs
            E    = (M x 1) vector of error after each iteration
    '''
    def logistic(z):
        ''' 
        logistic sigmoid function
        '''
        return 1 / (1+np.exp(-z))
    def grad(tar,w,x):
        # gradient of cross entropy loss
        z = np.reshape(np.transpose(w).dot(x),tar.shape)
        y = logistic(z)
        m = x.shape[1]
        return  -(1/m)*(x.dot(tar-y))
        #for j in range(len(w)):
        #    dE[j] = -x[j,:].dot(tar-y)
        #return dE        
    def cost_function(tar,w,x):
        # Computes the cost function for all the training samples
        z = np.reshape(np.transpose(w).dot(x),tar.shape)
        y = logistic(z)
        cost = -np.sum(np.multiply(tar,np.log(y)) + np.multiply(1-tar,np.log(1-y)))
        return cost
    k = tr.shape[0]
    w = np.reshape(np.repeat(0,k,axis=0),(k,1))
    E_train = []# np.reshape(np.repeat(0,M,axis=0),(M,1))
    E_test  = []# np.reshape(np.repeat(0,M,axis=0),(M,1))
    E_cv    = []# np.reshape(np.repeat(0,M,axis=0),(M,1))
    test_acc =[]
    w_best = []
    E_best = []
    acc_best=[]
    for t in range(M): 
        # calculate error for each epoch
        E_train.append(cost_function(tar,w,tr)/len(tar))
        E_test.append(cost_function(tar_test,w,test)/len(tar_test))
        E_cv.append(cost_function(tar_cv,w,cv)/len(tar_cv))
        yhat = logistic(np.reshape(np.transpose(w).dot(test),tar_test.shape))
        yhat[yhat< .5] = 0
        yhat[yhat>=.5] = 1
        test_acc.append(1- np.sum((np.abs(yhat - tar_test))/len(tar_test)))
        if E_cv[t] > E_cv[t-1] and (t > 1) :
            w_best = w
            E_best = E_test.append(cost_function(tar_test,w,test)) 
            ypred  = logistic(np.reshape(np.transpose(w).dot(test),tar_test.shape))
            ypred[ypred<.5] = 0
            ypred[ypred>=.5] = 1
            acc_best = np.sum(np.abs(ypred - tar_test))
            acc_best = 1 - acc_best / len(tar_test)  # percent accuracy on test
        w = w - alpha*grad(tar,w,tr)
    return w, E_train, E_test, E_cv,test_acc,w_best,E_best,acc_best

# -------------------------------------------------------------------------- #
# additional functions
def generate_targets(data):
    emo  = [i for i in data if data[i]!=data.default_factory()]
    if len(emo) == 2:
        n0 = len(data[emo[0]])
        n1 = len(data[emo[1]])
        y = np.reshape(np.repeat(0,n0 + n1,axis=0),(n0+n1,1))
        y[n0:n1+n0] = np.reshape(np.repeat(1,n1,axis=0),(n1,1))
    return y 

# -------------------------------------------------------------------------- #
    def batch_gradient(tr,val,test,M,alpha):
    
    def softmax(x):
        softmax_output = np.exp(x)/np.sum(np.exp(x))
        return softmax_output
    
    k = tr.shape[0]
    num_images_train = tr.shape[1]/6
    num_images_val = val.shape[1]/6
    num_images_test = test.shape[1]/6
    w = np.zeros((6, k))
    E_train = []
    E_val = []
    E_test = []
    acc_train = []
    acc_val = []
    acc_test = []

    for t in range(0,M):
        counter = -1
        acc_temp = 0
        E_temp_train = 0
        E_temp_val = 0
        E_temp_test = 0
        dE = np.zeros((6, k))
        conf_matrix = np.zeros((6,6))
        
        for image in range(0, tr.shape[1]):
            
            if image%num_images_train == 0:
                counter += 1
                
            target = np.zeros((6,1))
            target[counter] = 1
            target = np.transpose(target)[0]

            a_train = np.matmul(w,tr[:,image])
            y_output_train = softmax(a_train)
            
            if counter == np.argmax(y_output_train):
                acc_temp += 1
            
            for i in range(0,len(y_output_train)):
                E_temp_train += -target[i]*np.log(y_output_train[i])
                
            error_diff_temp = -np.outer((target - y_output_train),tr[:,image])
            dE += error_diff_temp
        
        dE = dE/tr.shape[1]
        acc_train_temp = acc_temp/tr.shape[1]
        E_temp_train = E_temp_train/tr.shape[1]
        E_temp_train = E_temp_train/6
        
        acc_temp = 0
        counter = -1
        for image in range(0, val.shape[1]):
            
            if image%num_images_val == 0:
                counter += 1
                
            target = np.zeros((6,1))
            target[counter] = 1
            target = np.transpose(target)[0]
            
            a_val = np.matmul(w,val[:,image])
            y_output_val = softmax(a_val)
            
            if counter == np.argmax(y_output_val):
                acc_temp += 1
            
            for i in range(0,len(y_output_val)):
                E_temp_val += -target[i]*np.log(y_output_val[i])
        
        acc_val_temp = acc_temp/val.shape[1]
        E_temp_val = E_temp_val/val.shape[1]
        E_temp_val = E_temp_val/6
        
        counter = -1
        acc_temp = 0
        for image in range(0, test.shape[1]):
            
            if image%num_images_test == 0:
                counter += 1
                
            target = np.zeros((6,1))
            target[counter] = 1
            target = np.transpose(target)[0]
            
            a_test = np.matmul(w,test[:,image])
            y_output_test = softmax(a_test)
            
            conf_matrix[counter,np.argmax(y_output_test)] += 1/(test.shape[1]/6)
            if counter == np.argmax(y_output_test):
                acc_temp += 1
            
            for i in range(0,len(y_output_test)):
                E_temp_test += -target[i]*np.log(y_output_test[i])
        
        acc_test_temp = acc_temp/test.shape[1]
        E_temp_test = E_temp_test/test.shape[1]
        E_temp_test = E_temp_test/6
        
        w = w - alpha * dE
        E_train.append(E_temp_train)
        E_val.append(E_temp_val)
        E_test.append(E_temp_test)
        
        acc_train.append(acc_train_temp)
        acc_val.append(acc_val_temp)
        acc_test.append(acc_test_temp)
        
    return w, conf_matrix, E_train, E_val, E_test, acc_train, acc_val, acc_test

def batch_gradient_EC(tr,val,test,M,alpha,count,count2):
    
    def softmax(x):
        softmax_output = np.exp(x)/np.sum(np.exp(x))
        return softmax_output
    
    k = tr.shape[0]
    w = np.zeros((6, k))
    E_train = []
    E_val = []
    E_test = []
    acc_train = []
    acc_val = []
    acc_test = []

    for t in range(0,M):
        counter = -1
        acc_temp = 0
        E_temp_train = 0
        E_temp_val = 0
        E_temp_test = 0
        dE = np.zeros((6, k))
        conf_matrix = np.zeros((6,6))
        
        for image in range(0, tr.shape[1]):
            
            if image >= 0 and image < count[0]:
                counter = 0
                buffer = (1/6)/(count[0]/(np.sum(count)))
            elif image >= count[0] and image < count[0] + count[1]:
                counter = 1
                buffer = (1/6)/(count[1]/(np.sum(count)))
            elif image >= count[0] + count[1] and image < count[0] + count[1] + count[2]:
                counter = 2
                buffer = (1/6)/(count[2]/(np.sum(count)))
            elif image >= count[0] + count[1] + count[2] and image < count[0] + count[1] + count[2] + count[3]:
                counter = 3
                buffer = (1/6)/(count[3]/(np.sum(count)))
            elif image >= count[0] + count[1] + count[2] + count[3] and image < count[0] + count[1] + count[2] + count[3] + count[4]:
                counter = 4
                buffer = (1/6)/(count[4]/(np.sum(count)))
            elif image >= count[0] + count[1] + count[2] + count[3] + count[4] and image < count[0] + count[1] + count[2] + count[3] + count[4] + count[5]:
                counter = 5
                buffer = (1/6)/(count[5]/(np.sum(count)))
                    
            target = np.zeros((6,1))
            target[counter] = 1
            target = np.transpose(target)[0]

            a_train = np.matmul(w,tr[:,image])
            y_output_train = softmax(a_train)
            
            if counter == np.argmax(y_output_train):
                acc_temp += 1
            
            for i in range(0,len(y_output_train)):
                E_temp_train += -target[i]*np.log(y_output_train[i])
                
            error_diff_temp = -np.outer((target - y_output_train),tr[:,image]) * buffer
            dE += error_diff_temp
        
        dE = dE/tr.shape[1]
        acc_train_temp = acc_temp/tr.shape[1]
        E_temp_train = E_temp_train/tr.shape[1]
        E_temp_train = E_temp_train/6
        
        acc_temp = 0
        counter = -1
        for image in range(0, val.shape[1]):
            
            if image >= 0 and image < count2[0]:
                counter = 0
                buffer = count2[0]/(np.sum(count2))
            elif image >= count2[0] and image < count2[0] + count2[1]:
                counter = 1
                buffer = count2[1]/(np.sum(count2))
            elif image >= count2[0] + count2[1] and image < count2[0] + count2[1] + count2[2]:
                counter = 2
                buffer = count2[2]/(np.sum(count2))
            elif image >= count2[0] + count2[1] + count2[2] and image < count2[0] + count2[1] + count2[2] + count2[3]:
                counter = 3
                buffer = count2[3]/(np.sum(count2))
            elif image >= count2[0] + count2[1] + count2[2] + count2[3] and image < count2[0] + count2[1] + count2[2] + count2[3] + count2[4]:
                counter = 4
                buffer = count2[4]/(np.sum(count2))
            elif image >= count2[0] + count2[1] + count2[2] + count2[3] + count2[4] and image < count2[0] + count2[1] + count2[2] + count2[3] + count2[4] + count2[5]:
                counter = 5
                buffer = count2[5]/(np.sum(count2))
                
            target = np.zeros((6,1))
            target[counter] = 1
            target = np.transpose(target)[0]
            
            a_val = np.matmul(w,val[:,image])
            y_output_val = softmax(a_val)
            
            if counter == np.argmax(y_output_val):
                acc_temp += 1
            
            for i in range(0,len(y_output_val)):
                E_temp_val += -target[i]*np.log(y_output_val[i])
        
        acc_val_temp = acc_temp/val.shape[1]
        E_temp_val = E_temp_val/val.shape[1]
        E_temp_val = E_temp_val/6
        
#        counter = -1
#        acc_temp = 0
#        for image in range(0, test.shape[1]):
#            
#            if image%num_images_test == 0:
#                counter += 1
#                
#            target = np.zeros((6,1))
#            target[counter] = 1
#            target = np.transpose(target)[0]
#            
#            a_test = np.matmul(w,test[:,image])
#            y_output_test = softmax(a_test)
#            
#            conf_matrix[counter,np.argmax(y_output_test)] += 1/(test.shape[1]/6)
#            if counter == np.argmax(y_output_test):
#                acc_temp += 1
#            
#            for i in range(0,len(y_output_test)):
#                E_temp_test += -target[i]*np.log(y_output_test[i])
        
        acc_test_temp = acc_temp/test.shape[1]
        E_temp_test = E_temp_test/test.shape[1]
        E_temp_test = E_temp_test/6
        
        w = w - alpha * dE
        E_train.append(E_temp_train)
        E_val.append(E_temp_val)
        E_test.append(E_temp_test)
        
        acc_train.append(acc_train_temp)
        acc_val.append(acc_val_temp)
        acc_test.append(acc_test_temp)
        
    return w, conf_matrix, E_train, E_val, E_test, acc_train, acc_val, acc_test

def stochastic_gradient(tr,val,test,M,alpha):
    
    def softmax(x):
        softmax_output = np.exp(x)/np.sum(np.exp(x))
        return softmax_output
    
    alpha = alpha/tr.shape[1]
    k = tr.shape[0]
    num_images_val = val.shape[1]/6
    num_images_test = test.shape[1]/6
    w = np.zeros((6, k))
    E_train = []
    E_val = []
    E_test = []
    acc_train = []
    acc_val = []
    acc_test = []

    for t in range(0,M):
        counter = -1
        acc_temp = 0
        E_temp_train = 0
        E_temp_val = 0
        E_temp_test = 0
        conf_matrix = np.zeros((6,6))
        w_new = w
        
        for image in range(0, val.shape[1]):
            
            if image%num_images_val == 0:
                counter += 1
                
            target = np.zeros((6,1))
            target[counter] = 1
            target = np.transpose(target)[0]
            
            a_val = np.matmul(w,val[:,image])
            y_output_val = softmax(a_val)
            
            if counter == np.argmax(y_output_val):
                acc_temp += 1
            
            for i in range(0,len(y_output_val)):
                E_temp_val += -target[i]*np.log(y_output_val[i])
        
        acc_val_temp = acc_temp/val.shape[1]
        E_temp_val = E_temp_val/val.shape[1]
        E_temp_val = E_temp_val/6
        
        counter = -1
        acc_temp = 0
        for image in range(0, test.shape[1]):
            
            if image%num_images_test == 0:
                counter += 1
                
            target = np.zeros((6,1))
            target[counter] = 1
            target = np.transpose(target)[0]
            
            a_test = np.matmul(w,test[:,image])
            y_output_test = softmax(a_test)
            
            conf_matrix[counter,np.argmax(y_output_test)] += 1/(test.shape[1]/6)
            if counter == np.argmax(y_output_test):
                acc_temp += 1
            
            for i in range(0,len(y_output_test)):
                E_temp_test += -target[i]*np.log(y_output_test[i])
                
        acc_temp = 0
        counter = -1
        
        acc_test_temp = acc_temp/test.shape[1]
        E_temp_test = E_temp_test/test.shape[1]
        E_temp_test = E_temp_test/6
        
        for image in range(0, tr.shape[1]):
            
            P = [i for i in range(0, tr.shape[1])]
            random.shuffle(P)
            
            if P[image] >= 0 and P[image] < tr.shape[1]/6:
                counter = 0
            elif P[image] >= tr.shape[1]/6 and P[image] < 2*tr.shape[1]/6: 
                counter = 1
            elif P[image] >= 2*tr.shape[1]/6 and P[image] < 3*tr.shape[1]/6: 
                counter = 2
            elif P[image] >= 3*tr.shape[1]/6 and P[image] < 4*tr.shape[1]/6: 
                counter = 3
            elif P[image] >= 4*tr.shape[1]/6 and P[image] < 5*tr.shape[1]/6: 
                counter = 4
            elif P[image] >= 5*tr.shape[1]/6 and P[image] < tr.shape[1]: 
                counter = 5
                
            target = np.zeros((6,1))
            target[counter] = 1
            target = np.transpose(target)[0]

            a_train = np.matmul(w,tr[:,P[image]])
            y_output_train = softmax(a_train)
            
            if counter == np.argmax(y_output_train):
                acc_temp += 1
            
            for i in range(0,len(y_output_train)):
                E_temp_train += -target[i]*np.log(y_output_train[i])
                
            error_diff_temp = -np.outer((target - y_output_train),tr[:,P[image]])
            w_new = w_new - alpha * error_diff_temp
            
        w = w_new
        
        acc_train_temp = acc_temp/tr.shape[1]
        E_temp_train = E_temp_train/tr.shape[1]
        E_temp_train = E_temp_train/6
        
        E_train.append(E_temp_train)
        E_val.append(E_temp_val)
        E_test.append(E_temp_test)
        
        acc_train.append(acc_train_temp)
        acc_val.append(acc_val_temp)
        acc_test.append(acc_test_temp)
        
    return w, conf_matrix, E_train, E_val, E_test, acc_train, acc_val, acc_test

def generate_targets(data):
    emo  = [i for i in data if data[i]!=data.default_factory()]
    if len(emo) == 2:
        n0 = len(data[emo[0]])
        n1 = len(data[emo[1]])
        y = np.reshape(np.repeat(0,n0 + n1,axis=0),(n0+n1,1))
        y[n0:n1+n0] = np.reshape(np.repeat(1,n1,axis=0),(n1,1))
    return y 


if __name__ == '__main__':
    # -------------------------------------------------------------------------- #
    # additional functions
    
    
        
#%% LOGISTIC REGRESSION EXPERIMENTS
# -------------------------------------------------------------------------- #
# build datasets
aligned_dir = main_dir + "./aligned"  # aligned images directory
resized_dir = main_dir + "./resized"  # resized images directory
[dataset,cnt]       = load_data(aligned_dir)
[re_dataset,re_cnt] = load_data(resized_dir)

# experiment 1 - logistic regression (happy vs. angry) with resized images
LR_emotions = ['happiness','anger']
LR_dataset = balanced_sampler(re_dataset, re_cnt, LR_emotions)
[train,cv,test] = splitter1(LR_dataset) # split data into train,cv,test
# generate target vectors
y_train = generate_targets(train)
y_cv    = generate_targets(cv)
y_test  = generate_targets(test)
# project all data onto principal components
npc = 9
new_test                = PCA_holdout(train,test,npc)
new_cv                  = PCA_holdout(train,cv,npc)
[new_train, components,components1] = PCA(train,npc)
# train using batch gradient descent
[w, E_train, E_test, E_cv,test_acc,w_best,E_best,acc_best] = batch_descent(tr=new_train,
tar=y_train,M = 50, alpha=1,cv=new_cv,test=new_test,tar_cv=y_cv,tar_test=y_test)

# plot results
xx = range(50)
fig = plt.figure()
p = fig.add_subplot(111)
p.set_xlabel('epoch',fontsize = 16)
p.set_ylabel('loss',fontsize = 16)
p.plot(xx,E_train, 'r-', linewidth = 3, label = 'training')
p.plot(xx,E_cv, 'b-',linewidth = 3, label = 'validation')
p.legend()
p.set_title('Resized: Training and validation loss for epochs=50, lr = 1, npc = 9')
plt.savefig(fig_dir + '6.png',dpi=100)
plt.show()

# show first 4 principal components as images
img_shape = LR_dataset['anger'][0].shape
face = np.reshape(components1[:,0],img_shape)
img = Image.fromarray(face)
img.show()

#---------------------------------------------------------------------------#

# experiment 2/3 - logistic regression (happy vs. angry) with aligned images
#                    and (fear vs. surprise)
emotions = ['fear','surprise']
data = balanced_sampler(dataset,cnt,emotions)
# loop through folds
ERR_train = []
ERR_cv    = []
ERR_test  = []
ERR_best  = []
ACC_best  = []
ACC_test  = []
for kk in range(10): 
    [new_train,new_cv,new_test,y_train,y_cv,y_test] = kfoldcross(k=10, data = data,
    i = kk, emotions = emotions,npc = 15)
    [w, E_train, E_test, E_cv,test_acc,w_best,E_best,acc_best] = batch_descent(tr=new_train,
    tar=y_train,M = 50, alpha=.9,cv=new_cv,test=new_test,tar_cv=y_cv,tar_test=y_test)
    if not E_best: 
        E_best = E_test[-1]
    if not acc_best: 
        acc_best = test_acc[-1]
    ERR_train.append(E_train)
    ERR_cv.append(E_cv)
    ERR_test.append(E_test)
    ERR_best.append(E_best)
    ACC_best.append(acc_best)
    ACC_test.append(test_acc)
ERR_train  = np.array(ERR_train)
ERR_cv     = np.array(ERR_cv)
ERR_test   = np.array(ERR_test)
ERR_best   = np.array(ERR_best)
ACC_best   = np.array(ACC_best)
ACC_test   = np.array(ACC_test)


# plot average training loss
xx = range(50)
avg_train_loss = np.mean(ERR_train,axis = 0)
std_train_loss = np.std(ERR_train,axis = 0)
avg_val_loss   = np.mean(ERR_cv,axis = 0)
std_val_loss   = np.std(ERR_cv,axis=0)
ind = np.arange(9,50,10)
fig = plt.figure()
p = fig.add_subplot(111)
p.set_xlabel('epoch',fontsize = 16)
p.set_ylabel('loss',fontsize = 16)
p.plot(xx,avg_train_loss, 'r-', linewidth = 3, label = 'training')
p.errorbar(ind,avg_train_loss[ind],yerr = std_train_loss[ind],ecolor = 'r',fmt = 'none') #,elinewidth = 2, capsize = 5)
p.plot(xx,avg_val_loss, 'b-',linewidth = 3, label = 'validation')
p.errorbar(ind,avg_val_loss[ind],yerr = std_val_loss[ind],ecolor = 'b',fmt = 'none') #,elinewidth = 2, capsize = 5)
p.legend()
p.set_title('Training and validation loss for epochs=50, lr = .9, npc = 15')
plt.savefig(fig_dir + 'image1.png',dpi=100)
plt.show()
print("your average accuracy is: " + str(np.mean(ACC_best)))
print("your average accuracy std dev is : " + str(np.std(ACC_best)))
print("your avg test loss is: " + str(np.mean(ERR_best,axis=0)))
print("your test loss std dev is: " + str(np.std(ERR_best,axis=0)))


#%% SOFTMAX REGRESSION EXPERIMENTS
# -------------------------------------------------------------------------- #
   # softmax multi-class regression
    SF_emotions = ['happiness', 'anger', 'disgust', 'fear', 'sadness', 'surprise']
    #SF_dataset = balanced_sampler(dataset, cnt, SF_emotions)
    SF_dataset = dataset
    
    k = 10
    emo_counter = 0
    M = 50
    
    folds1 = np.array_split(SF_dataset[SF_emotions[0]], k)
    folds2 = np.array_split(SF_dataset[SF_emotions[1]], k)
    folds3 = np.array_split(SF_dataset[SF_emotions[2]], k)
    folds4 = np.array_split(SF_dataset[SF_emotions[3]], k)
    folds5 = np.array_split(SF_dataset[SF_emotions[4]], k)
    folds6 = np.array_split(SF_dataset[SF_emotions[5]], k)
    
    val = defaultdict(list)
    test = defaultdict(list)
    train = defaultdict(list)
    
    E_train_cum = np.zeros((M,k))
    E_val_cum = np.zeros((M,k))
    E_test_cum = np.zeros((M,k))
    acc_train_cum = np.zeros((M,k))
    acc_val_cum = np.zeros((M,k))
    acc_test_cum = np.zeros((M,k))
    
    E_train_cum_sgd = np.zeros((M,k))
    E_val_cum_sgd = np.zeros((M,k))
    E_test_cum_sgd = np.zeros((M,k))
    acc_train_cum_sgd = np.zeros((M,k))
    acc_val_cum_sgd = np.zeros((M,k))
    acc_test_cum_sgd = np.zeros((M,k))

    for i in range(0,k):
        
        idx = list(set(range(0,k)) - set([i,(i+1)%k]))
        
        val[SF_emotions[0]] = folds1[i]
        test[SF_emotions[0]] = folds1[(i+1)%k]
        train_temp = [folds1[i] for i in idx]
        train[SF_emotions[0]] = [i for sub in train_temp for i in sub]
        val[SF_emotions[1]] = folds2[i]
        test[SF_emotions[1]] = folds2[(i+1)%k]
        train_temp = [folds2[i] for i in idx]
        train[SF_emotions[1]] = [i for sub in train_temp for i in sub]
        val[SF_emotions[2]] = folds3[i]
        test[SF_emotions[2]] = folds3[(i+1)%k]
        train_temp = [folds3[i] for i in idx]
        train[SF_emotions[2]] = [i for sub in train_temp for i in sub]
        val[SF_emotions[3]] = folds4[i]
        test[SF_emotions[3]] = folds4[(i+1)%k]
        train_temp = [folds4[i] for i in idx]
        train[SF_emotions[3]] = [i for sub in train_temp for i in sub]
        val[SF_emotions[4]] = folds5[i]
        test[SF_emotions[4]] = folds5[(i+1)%k]
        train_temp = [folds5[i] for i in idx]
        train[SF_emotions[4]] = [i for sub in train_temp for i in sub]
        val[SF_emotions[5]] = folds6[i]
        test[SF_emotions[5]] = folds6[(i+1)%k]
        train_temp = [folds6[i] for i in idx]
        train[SF_emotions[5]] = [i for sub in train_temp for i in sub]
        
        count = [len(train[SF_emotions[0]]),len(train[SF_emotions[1]]),len(train[SF_emotions[2]]),len(train[SF_emotions[3]]),len(train[SF_emotions[4]]),len(train[SF_emotions[5]])]
        count2 = [len(val[SF_emotions[0]]),len(val[SF_emotions[1]]),len(val[SF_emotions[2]]),len(val[SF_emotions[3]]),len(val[SF_emotions[4]]),len(val[SF_emotions[5]])]
        
        [new_train,components] = PCA(train,40)
        new_val = PCA_holdout(train, val, 40)
        new_test = PCA_holdout(train, test, 40)
    
        [weights, conf_matrix, E_train, E_val, E_test, acc_train, acc_val, acc_test] = batch_gradient_EC(new_train,new_val,new_test,M,1.5,count,count2)
        [w_sgd, conf_matrix_sgd, E_train_sgd, E_val_sgd, E_test_sgd, acc_train_sgd, acc_val_sgd, acc_test_sgd] = stochastic_gradient(new_train,new_val,new_test,M,1.5)
        E_train_cum[:,i] = E_train
        E_val_cum[:,i] = E_val
        E_test_cum[:,i] = E_test
        
        E_train_cum_sgd[:,i] = E_train_sgd
        E_val_cum_sgd[:,i] = E_val_sgd
        E_test_cum_sgd[:,i] = E_test_sgd
        
        acc_train_cum[:,i] = acc_train
        acc_val_cum[:,i] = acc_val
        acc_test_cum[:,i] = acc_test
        
        acc_train_cum_sgd[:,i] = acc_train_sgd
        acc_val_cum_sgd[:,i] = acc_val_sgd
        acc_test_cum_sgd[:,i] = acc_test_sgd

    avg_train_loss = []
    avg_val_loss = []
    avg_test_loss = []
    
    std_train_loss = []
    std_val_loss = []
    std_test_loss = []
    
    avg_train_acc = []
    avg_val_acc = []
    avg_test_acc = []
    
    std_train_acc = []
    std_val_acc = []
    std_test_acc = []
    
    avg_train_loss_sgd = []
    avg_val_loss_sgd = []
    avg_test_loss_sgd = []
    
    std_train_loss_sgd = []
    std_val_loss_sgd = []
    std_test_loss_sgd = []
    
    avg_train_acc_sgd = []
    avg_val_acc_sgd = []
    avg_test_acc_sgd = []
    
    std_train_acc_sgd = []
    std_val_acc_sgd = []
    std_test_acc_sgd = []
    
    for i in range(len(E_train_cum)):
        avg_train_loss.append(np.mean(E_train_cum[i,:]))
        avg_val_loss.append(np.mean(E_val_cum[i,:]))
        avg_test_loss.append(np.mean(E_test_cum[i,:]))
        
        if i%10 == 0:
            
            std_train_loss.append(np.std(E_train_cum[i,:]))
            std_val_loss.append(np.std(E_val_cum[i,:]))
            std_test_loss.append(np.std(E_test_cum[i,:]))
        else:
            std_train_loss.append(0)
            std_val_loss.append(0)
            std_test_loss.append(0)
        
        avg_train_acc.append(np.mean(acc_train_cum[i,:]))
        avg_val_acc.append(np.mean(acc_val_cum[i,:]))
        avg_test_acc.append(np.mean(acc_test_cum[i,:]))
        
        if i%10 == 0:
            
            std_train_acc.append(np.std(acc_train_cum[i,:]))
            std_val_acc.append(np.std(acc_val_cum[i,:]))
            std_test_acc.append(np.std(acc_test_cum[i,:]))
            
        else:
            std_train_acc.append(0)
            std_val_acc.append(0)
            std_test_acc.append(0)
            
        avg_train_loss_sgd.append(np.mean(E_train_cum_sgd[i,:]))
        avg_val_loss_sgd.append(np.mean(E_val_cum_sgd[i,:]))
        avg_test_loss_sgd.append(np.mean(E_test_cum_sgd[i,:]))
        
        if i%10 == 0:
            
            std_train_loss_sgd.append(np.std(E_train_cum_sgd[i,:]))
            std_val_loss_sgd.append(np.std(E_val_cum_sgd[i,:]))
            std_test_loss_sgd.append(np.std(E_test_cum_sgd[i,:]))
        else:
            std_train_loss_sgd.append(0)
            std_val_loss_sgd.append(0)
            std_test_loss_sgd.append(0)
        
        avg_train_acc_sgd.append(np.mean(acc_train_cum_sgd[i,:]))
        avg_val_acc_sgd.append(np.mean(acc_val_cum_sgd[i,:]))
        avg_test_acc_sgd.append(np.mean(acc_test_cum_sgd[i,:]))
        
        if i%10 == 0:
            
            std_train_acc_sgd.append(np.std(acc_train_cum_sgd[i,:]))
            std_val_acc_sgd.append(np.std(acc_val_cum_sgd[i,:]))
            std_test_acc_sgd.append(np.std(acc_test_cum_sgd[i,:]))
            
        else:
            std_train_acc_sgd.append(0)
            std_val_acc_sgd.append(0)
            std_test_acc_sgd.append(0)
        
    xx = range(M)
    
    fig, ax = plt.subplots()
    ax.errorbar(xx,avg_train_loss, yerr = std_train_loss, label = 'train')
    ax.errorbar(xx,avg_val_loss, yerr = std_val_loss, label = 'val')
    ax.set_title('Normalized Loss w/ 10 components')
    ax.set_xlabel('# of Epochs')
    ax.set_ylabel('Loss')
    plt.legend()
    
    fig, ax = plt.subplots()
    ax.errorbar(xx,avg_train_acc, yerr = std_train_acc, label = 'train')
    ax.errorbar(xx,avg_val_acc, yerr = std_val_acc, label = 'val')
    ax.set_title('Accuracy w/ 10 components')
    ax.set_xlabel('# of Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    
    fig, ax = plt.subplots()
    ax.errorbar(xx,avg_train_loss, yerr = std_train_loss_sgd, label = 'Batch')
    ax.errorbar(xx,avg_train_loss_sgd, yerr = std_val_loss_sgd, label = 'Stochastic')
    ax.set_title('Batch vs Gradient')
    ax.set_xlabel('# of Epochs')
    ax.set_ylabel('Loss')
    plt.legend()
    1
    
    weights_to_image = np.matmul(weights,np.transpose(components))
    
    for i in range(len(weights_to_image)):
        weights_temp = weights_to_image[i,:]
        
        weights_to_image[i,:] = 255*(weights_temp-np.min(weights_temp))/(np.max(weights_temp) - np.min(weights_temp))
        
        images_display = np.reshape(weights_to_image[i,:],SF_dataset['happiness'][0].shape)
        #display_face(images_display)

