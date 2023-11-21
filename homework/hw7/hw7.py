# hw7.py
import jax.numpy as jnp 
from jax import grad 
import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt 
datapath = "./"

np.random.seed(0)

#################### Task 1 ###################

###### Helper functions for task 1 ########
# multi-class linear classification model 
def model(x, w): 
	"""
	input: 
	- x: shape (N, P)  
	- W: shape (N+1, C) 

	output: 
	- prediction: shape (C, P) 
	"""
	# option 1: stack 1 
	f = x   
	# print("before stack 1, x.shape: ", f.shape)

	# tack a 1 onto the top of each input point all at once
	o = jnp.ones((1, np.shape(f)[1]))
	f = jnp.vstack((o,f))

	# print("after stack 1, the X.shape:", f.shape)

	# compute linear combination and return
	a = jnp.dot(f.T,w)

	# option 2: 
	# a = w[0, :] + jnp.dot(x.T, w[1:, :])
	return a.T


# multi-class softmax cost function 
def multiclass_softmax(w, x_p, y_p):     
	"""
	Args:
	 	- w: parameters. shape (N+1, C), C= the number of classes
	 	- x_p: input. shape (N, P) 
		- y_p: label. shape (1, P)
	Return: 
		- softmax cost: shape (1,)
	"""
    
	# pre-compute predictions on all points
	all_evals = model(x_p,w)
	# print(f"all_evals[:, 0:5].T={all_evals[:, 0:5].T}")

	# logsumexp trick
	maxes = jnp.max(all_evals, axis=0)
	a = maxes + jnp.log(jnp.sum(jnp.exp(all_evals - maxes), axis=0))

	# compute cost in compact form using numpy broadcasting
	b = all_evals[y_p.astype(int).flatten(), jnp.arange(np.size(y_p))]
	cost = jnp.sum(a - b)

	# return average
	return cost/float(np.size(y_p))

def gradient_descent(cost_func, x, y, alpha=1e-2, iterations=500, weights_scaling_factor=1):
    gradient = grad(cost_func, argnums=0)

    # w = np.array([3.,3.])
    w_dim = (x.shape[0]+1, np.unique(y).size)
    w = np.random.rand(w_dim[0], w_dim[1])*weights_scaling_factor
    cost = cost_func(w, x, y)
    w_history = [w, ]
    cost_history = [cost, ]
    grad_history = [gradient(w, x, y), ]
    for i in np.arange(1, iterations+1, 1):
        # get gradient
        w_grad = gradient(w, x, y)
        grad_history.append(w_grad)

        w = w - alpha * w_grad
        w_history.append(w)

        # cost
        cost = cost_func(w, x, y)
        cost_history.append(cost)

    return w_history, cost_history

def get_accuracy(w, x, y):
	x_append_transpose = jnp.vstack([jnp.ones(x.shape[1]), x]).T
	pred_values = jnp.dot(x_append_transpose, w)
	pred_y = pred_values.argmax(axis=1)
	# print(pred_y)
	miscalssfications = jnp.count_nonzero(pred_y - y)
	# print("miscalssfications", miscalssfications)
	acc = (y.size - miscalssfications) / y.size
	# print("accuracy: %.1f %%" % (acc*100))
	return acc

def get_optimal_gamma(cost_func, x, y, gamma_arr=np.arange(-2, 7, 1), iterations=10, weights_scaling_factor=1):
    results = []
    for a in gamma_arr:
        w_history, cost_history = gradient_descent(cost_func, x, y, alpha=np.float_power(10, -1*a),\
                                                    iterations=iterations, weights_scaling_factor=weights_scaling_factor)
        # print("cost_history", cost_history)
        acc_tmp = []
        for w in w_history:
            acc = get_accuracy(w, x, y)
            acc_tmp.append(acc)
        results.append((a, cost_history, acc_tmp))
        
    min_idx = int(jnp.argmin(jnp.array([i[1][-1] for i in results])))
    return results[min_idx]

def standard_normalization(x):
    """
    x: input N; (dimensions) * P (samples)
    return: standardized_x; N (dimensions) * P (samples)
    """
    standardized_x = ( ( x.T - x.mean(axis=1) ) / x.std(axis=1) ).T
    standardized_x = jnp.nan_to_num(standardized_x, nan=0)
    return standardized_x

def center(x):
    standardized_x = (x.T - x.mean(axis=1)).T
    return standardized_x

def compute_pcs(x, lam):
    p = float(x.shape[1])
    cov = 1/p * np.dot(x, x.T) + lam*np.eye(x.shape[0]) # adding lam to prevent numerical stability
    D, V = np.linalg.eigh(cov)
    return D, V

def pca_transform_data(x, **kwargs):
    num_components = x.shape[0]
    if 'num_components' in kwargs:
        num_components = kwargs["num_components"]
    lam = 1e-7
    if "lam" in kwargs:
        lam = kwargs["lam"]
    
    D, V = compute_pcs(x, lam)
    V = V[:, -num_components:]
    D = D[-num_components:]
    
    # compute transformed data for PC space
    w = np.dot(V.T, x)
    return w, V

def run_task1(): 

	# import MNIST
	x, y = fetch_openml('mnist_784', version=1, return_X_y=True)

	# re-shape input/output data
	x = x.T
	y = np.array([int(v) for v in y])[np.newaxis,:]

	print(np.shape(x)) # (784, 70000)
	print(np.shape(y)) # (1, 70000)

	# TODO: fill in your code
	x_training = jnp.array(x.iloc[:,:50000])
	y_training = jnp.array(y[:,:50000])

	# original dataset
	results_original = get_optimal_gamma(multiclass_softmax, x_training, y_training,\
									   gamma_arr=np.arange(-2, 7, 1), iterations=10, weights_scaling_factor=0.01)
	print(f"gamma for traing using original data: {results_original[0]}")
    # standard normalization
	x_standardized = standard_normalization(x_training)
	results_standard = get_optimal_gamma(multiclass_softmax, x_standardized, y_training,\
                                       gamma_arr=np.arange(-2, 7, 1), iterations=10, weights_scaling_factor=1)
	print(f"gamma for traing using standardized data: {results_standard[0]}")
     
	# PCA Sphering
	x_center = center(x_training)
	kwargs = {'lam':1e-7}
	x_pca, V = pca_transform_data(x_center, **kwargs)
	x_pac_scale = ( x_pca.T  / x_pca.std(axis=1) ).T
	x_pac_scale = jnp.nan_to_num(x_pac_scale, nan=0)
	results_pca = get_optimal_gamma(multiclass_softmax, x_pac_scale, y_training,\
										gamma_arr=np.arange(-2, 7, 1), iterations=10, weights_scaling_factor=1)
	print(f"gamma for traing using PCA sphered data: {results_pca[0]}")
	
	# plot cost history individually & together
	fig, ax = plt.subplots(2, 2, figsize=(12,8))
	ax[0][0].plot(results_original[1], "b-", linewidth=2, label="original")
	ax[0][1].plot(results_standard[1], "r-", linewidth=2, label="standard normalization")
	ax[1][0].plot(results_pca[1], "g-", linewidth=2, label="PCA sphering")
    
	ax[1][1].plot(results_original[1], "b-", linewidth=2, label="original")
	ax[1][1].plot(results_standard[1], "r-", linewidth=2, label="standard normalization")
	ax[1][1].plot(results_pca[1], "g-", linewidth=2, label="PCA sphering")
	for axi in ax.flatten():
		axi.set_xlabel("Iterations")
		axi.set_ylabel("Cost")
		axi.legend()
		axi.grid()
	plt.tight_layout()
	plt.savefig("task1-cost.png")
	plt.show()
     
	# plot accuracy together
	fig, ax = plt.subplots(1, 1, figsize=(8,6))
	ax.plot(results_original[-1], "b-", linewidth=2, label="original")
	ax.plot(results_standard[-1], "r-", linewidth=2, label="standard normalization")
	ax.plot(results_pca[-1], "g-", linewidth=2, label="PCA sphering")
	plt.xlabel("Iterations")
	plt.ylabel("Accuracy")
	plt.tight_layout()
	plt.legend()
	plt.grid()
	plt.savefig("task1-accuracy.png")
	plt.show()

################## TASK2
def model2(x, w): 
    """
    input: 
    - x: shape (N, P)  
    - W: shape (N+1, 1) 

    output: 
    - prediction: shape (1, P) 
    """
    # option 1: stack 1 
    f = x   
    # print("before stack 1, x.shape: ", f.shape)

    # tack a 1 onto the top of each input point all at once
    o = jnp.ones((1, np.shape(f)[1]))
    f = jnp.vstack((o,f))

    # print("after stack 1, the X.shape:", f.shape)

    # compute linear combination and return
    a = jnp.dot(f.T,w)

    # option 2: 
    # a = w[0, :] + jnp.dot(x.T, w[1:, :])
    return a.T

def gradient_descent_linear(cost_func, x, y, lamb, alpha=1e-2, iterations=500):
    gradient = grad(cost_func, argnums=0)

    # w = np.array([3.,3.])
    w_dim = (x.shape[0]+1, 1)
    w = np.random.rand(w_dim[0], w_dim[1])
    cost = cost_func(w, x, y, lamb)
    w_history = [w, ]
    cost_history = [cost, ]
    # grad_history = [gradient(w, x, y, lamb), ]
    beta = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    grad_history = [gradient(w, x, y, lamb), ]
    momentum_history = [np.zeros(w_dim), ]
    secondary_list = [np.zeros(w_dim), ]
    
    for i in np.arange(1, iterations, 1):
        # get gradient
        w_grad = gradient(w, x, y, lamb)
        grad_history.append(w_grad)

        # get momentum
        momentum = (1 - beta) * w_grad + beta * momentum_history[-1]
        momentum_history.append(momentum)

        # second derivative estimate
        w_grad_square = w_grad ** 2
        secondary = beta_2 * secondary_list[-1] + (1 - beta_2) * w_grad_square
        secondary_list.append(secondary)

        # momentum with normalization <-- adam
        w = w - ((alpha * momentum / (1 - jnp.power(beta, i))) / (
                jnp.sqrt(secondary / (1 - jnp.power(beta_2, i))) + epsilon))
        w_history.append(w)

        # cost
        cost = cost_func(w, x, y, lamb)
        cost_history.append(cost)

    return w_history, cost_history

def linear_regression_cost(w, x_p, y_p, lamb):     
    """
    Args:
        - w: parameters. shape (N+1, 1)
        - x_p: input. shape (N, P) 
        - y_p: label. shape (1, P)
    Return: 
        - linear regression cost: shape (1,)
    """

    # pre-compute predictions on all points
    # w: (N+1, 1)
    all_evals = model2(x_p, w)    # this would have a shape of 1 * P

    w_feature_touching = w[1:]
    # mean squre lost plus l1 norm
    cost = jnp.sum((all_evals - y_p)**2) + lamb*jnp.sum(jnp.abs(w_feature_touching))

    # return average
    return cost/float(np.size(y_p))


def run_task2(): 
	# load in data
	csvname =  datapath + 'boston_housing.csv'
	data = np.loadtxt(csvname, delimiter = ',')
	x = data[:-1,:]
	y = data[-1:,:] 

	print(np.shape(x))
	print(np.shape(y))
	# input shape: (13, 506)
	# output shape: (1, 506)

	# TODO: fill in your code 
	x_standard = standard_normalization(x)
	result_arr = []
	for lamb in [0, 50, 100, 150]:
		w_history, cost_history = gradient_descent_linear(linear_regression_cost, x_standard,\
														y, lamb=lamb, alpha=1e-1, iterations=400)
		result_arr.append((lamb, w_history[-1], cost_history))
		print(f"Final cost for lambda={lamb}: {cost_history[-1]}")
    
	# report hyperparameters
	print("alpha=0.1")
	print("iterations=400")
	# plot cost history
	fig, ax = plt.subplots(2, 2, figsize=(10,6))
	for axi, result in zip(ax.flatten(), result_arr):
		axi.plot(result[-1], '-', linewidth=2, label=f"lambda={result[0]}")
		axi.set_xlabel("Iterations")
		axi.set_ylabel("Cost")
		axi.legend()
		axi.grid()
	plt.tight_layout()
	plt.savefig("task2-costs.png")
	plt.show()
      
	# plot weights
	fig, ax = plt.subplots(2, 2, figsize=(10,6))
	for axi, w in zip(ax.flatten(), result_arr):
		axi.bar(np.arange(1,14,1), np.array(w[1][1:]).flatten(),  label=f"lambda={w[0]}")
		axi.set_title(f"feature touching weights, lambda={w[0]}")
		axi.set_ylim([-4, 3])
	plt.tight_layout()
	plt.savefig("task2-weights.png")
	plt.show()

if __name__ == '__main__':
	run_task1()
	run_task2() 