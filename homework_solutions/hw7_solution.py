# hw7.py
import jax.numpy as jnp 
from jax import grad 
import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt 
from matplotlib import gridspec
datapath = "./"


#################### Task 1 ###################

#### compare cost function histories ####
def plot_cost_histories(cost_histories,labels, start=0, figname="cost_histories.png"):
	"""
	args: 
		- cost_histories: list of list 
		- labels: the legend used for labeling 
		- start: the start position to plot each line  
		- figname: the name used for the figure 
	return: 
		- None 
	
	Will save the figure using the figname 
	"""

	# initialize figure
	# plt.figure(figsize = (10,3))
	plt.figure()

	# run through input histories, plotting each beginning at 'start' iteration
	for c in range(len(cost_histories)):
		history = cost_histories[c]
		label = labels[c]
		    
		# plot cost function history
		plt.plot(np.arange(start,len(history),1),
			history[start:],
			linewidth = 3*(0.8)**(c),
			label = label) 

	# clean up panel / axes labels
	xlabel = 'step $k$'
	ylabel = r'$g\left(\mathbf{w}^k\right)$'
	plt.xlabel(xlabel,fontsize = 14)
	plt.ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
	title = 'cost history'
	plt.title(title,fontsize = 18)

	# plot legend
	anchor = (1,1)
	plt.legend(loc='upper right', bbox_to_anchor=anchor)
	plt.xlim([start - 0.5,len(history) - 0.5]) 
	# plt.show()
	plt.savefig(figname)


def plot_acc_histories(acc_histories,labels, start=0, figname="acc_histories.png"):
	"""
	args: 
		- acc_histories: list of list 
		- labels: the legend used for labeling 
		- start: the start position to plot each line  
		- figname: the name used for the figure 
	return: 
		- None 
	
	Will save the figure using the figname 
	"""

	# initialize figure
	# plt.figure(figsize = (10,3))
	plt.figure()

	# run through input histories, plotting each beginning at 'start' iteration
	for c in range(len(acc_histories)):
		history = acc_histories[c]
		label = labels[c]
		    
		# plot cost function history
		plt.plot(np.arange(start,len(history),1),
			history[start:],
			linewidth = 3*(0.8)**(c),
			label = label) 

	# clean up panel / axes labels
	xlabel = 'step $k$'
	ylabel = 'accuracy'
	plt.xlabel(xlabel,fontsize = 14)
	plt.ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
	title = 'accuracy history'
	plt.title(title,fontsize = 18)

	# plot legend
	anchor = (1,1)
	plt.legend(loc='upper right', bbox_to_anchor=anchor)
	plt.xlim([start - 0.5,len(history) - 0.5]) 
	# plt.show()
	plt.savefig(figname)


# model multi-class linear classification model 
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
		- 
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

#  gradient descent with a fixed learning rate 
def gradient_descent(g, X, Y, w, alpha, max_its):  
	"""
	Args: 
		- g: cost function 
		- X: input
		- Y: label 
		- w: initial parameters 
		- alpha (float): learning rate 
		- max_its (integer): maximum iteations 
	"""
	grad_func = grad(g)

	# record history
	w_hist = [w]
	# item()  will convert the single-value DeviceArray to a Python float.
	cost_hist = [g(w, X, Y).item()]
	acc_hist = [evaluate_classifier_accuracy(Y, model(X, w))]

	# print(f"initial weight: w[:5]={w[:5]}")
	# print(f"initial cost: {cost_hist[-1]}")
	# print(f"initial accuracy: {acc_hist[-1]}")

	for k in range(max_its):   
		grad_eval = grad_func(w, X, Y)
		# print(f"k={k}, grad_eval[:5]={grad_eval[:5]}")

		# take descent step
		w = w - alpha*grad_eval

		# record weight update
		w_hist.append(w)
		cost_hist.append(g(w, X, Y).item())
		acc_hist.append(evaluate_classifier_accuracy(Y, model(X, w)))

	return w_hist,cost_hist, acc_hist 

def to_classlabel(z):
	"""
	input: 
		- z: shape (C, P)
	output: 
		- class label prediction: (P,)
	"""
	return z.argmax(axis=0)

def evaluate_classifier_accuracy(y, y_hat):
	"""
	input: 
		- y: shape (1, P) 
		- y_hat: shape (P,)

	"""
	pred_labels = to_classlabel(y_hat)

	# calculate classification accuracy

	correct_preds = sum(pred_labels == y.astype(int).flatten())
	classifier_accuracy = correct_preds / np.size(y)
	# print(f"correct_preds:{correct_preds}, classifier_accuracy:{classifier_accuracy}")

	return classifier_accuracy.item() 


def standard_normalizer(x):
	# compute the mean and standard deviation of the input
	x_means = np.mean(x, axis=1)[:, np.newaxis]
	x_stds = np.std(x, axis=1)[:, np.newaxis]

	# check to make sure thta x_stds > small threshold, for those not
	# divide by 1 instead of original standard deviation
	ind = np.argwhere(x_stds < 10 ** (-2))
	if len(ind) > 0:
		ind = [v[0] for v in ind]
		adjust = np.zeros((x_stds.shape))
		adjust[ind] = 1.0
		x_stds += adjust

	# create standard normalizer function
	normalizer = lambda data: (data - x_means) / x_stds

	# create inverse standard normalizer
	inverse_normalizer = lambda data: data * x_stds + x_means

	# return normalizer
	return normalizer, inverse_normalizer


def PCA(x, **kwargs):
	# regularization parameter for numerical stability
	lam = 10 ** (-7)
	if "lam" in kwargs:
		lam = kwargs["lam"]

	# create the correlation matrix
	P = float(x.shape[1])

	Cov = 1 / P * np.dot(x, x.T) + lam * np.eye(x.shape[0])

	# use numpy function to compute eigenvalues / vectors of correlation matrix
	# JNP gives a different (wrong) value here
	d, V = np.linalg.eigh(Cov)
	return d, V


def PCA_sphereing(x, **kwargs):
	# Step 1: mean-center the data
	x_means = np.mean(x, axis=1)[:, np.newaxis]
	x_centered = x - x_means

	# Step 2: compute pca transform on mean-centered data
	d, V = PCA(x_centered, **kwargs)

	# Step 3: divide off standard deviation of each (transformed) input,
	# which are equal to the returned eigenvalues in 'd'.
	stds = (d[:, np.newaxis]) ** (0.5)

	# check to make sure thta x_stds > small threshold, for those not
	# divide by 1 instead of original standard deviation
	ind = np.argwhere(stds < 10 ** (-2))

	# check to make sure thta x_stds > small threshold, for those not
	# divide by 1 instead of original standard deviation
	ind = np.argwhere(stds < 10**(-2))
	if len(ind) > 0:
		ind = [v[0] for v in ind]
		adjust = np.zeros((stds.shape))
		adjust[ind] = 1.0
		stds += adjust

	normalizer = lambda data: np.dot(V.T, data - x_means) / stds

	# create inverse normalizer
	inverse_normalizer = lambda data: np.dot(V, data * stds) + x_means

	# return normalizer
	return normalizer, inverse_normalizer


def fit_model(X, Y, gammas=range(-4, 3), figid=""):
	"""
	input: 
		- X: data of shape (N, P)
		- Y: label of shape (1, C) 
		- gammas: list of integers for learning rate tuning   

	output: 
		- weight_hist: the history of weights 
		- cost_hist: the history of costs 
		- acc_hist: the history of classification accuracy 
	"""
	### original initilize weight w 
	np.random.seed(0)
	w = np.random.randn(785, 10) 

	# gradient descent to optimize cost function 
	max_its = 10 

	# tune the learning rate 
	min_cost = -1 
	best_alpha = 0 
	best_w_hist = None 
	best_cost_hist = None 
	best_acc_hist = None 


	# used to identify the best alpha 
	cost_histories = []
	labels = [] 

	print(f"The list of gamma to try: {gammas}")
	for gamma in gammas:
		alpha = 10**gamma
		print("\n alpha = ", alpha) 
		w_hist, cost_hist, acc_hist = gradient_descent(multiclass_softmax, X, Y, w, 
		alpha=alpha, max_its=max_its)
		
		if min_cost == -1: 
			min_cost = cost_hist[-1] 
			best_alpha = alpha
			best_w_hist = w_hist 
			best_cost_hist = cost_hist 
			best_acc_hist = acc_hist 

		else:
			# if the current alpha cannot reach the so-far-best cost, we can early stop it. 
			if min_cost < cost_hist[-1]:
				# uncomment to show the diverged cost history 
				# cost_histories.append(cost_hist)
				# labels.append(f"alpha={alpha}")
				# print("cost_hist", cost_hist)
				break
			else: 
				best_alpha = alpha
				min_cost = cost_hist[-1]
				best_w_hist = w_hist 
				best_cost_hist = cost_hist 
				best_acc_hist = acc_hist 


		cost_histories.append(cost_hist)
		labels.append(f"alpha={alpha}")
		print("cost_hist", cost_hist)
		print("acc_hist", acc_hist)

	if len(gammas) > 1: 
		print("Find the best alpha: ", best_alpha)

	# optional: compare cost histories 
	plot_cost_histories(cost_histories,labels, start=0, 
		figname=f"task1-cost-histories-lr-tuning-{figid}.png")

	return best_w_hist, best_cost_hist, best_acc_hist 


def run_task1(): 
	print("Task 1 start ....")

	# import MNIST
	x, y = fetch_openml('mnist_784', version=1, return_X_y=True)

	# re-shape input/output data
	x = x.T
	y = np.array([int(v) for v in y])[np.newaxis,:]

	print(np.shape(x)) # (784, 70000)
	print(np.shape(y)) # (1, 70000)

	# TODO: fill in your code 
	# Get first 50000 elements
	X_task1 = x.iloc[:, 0:50000].to_numpy()
	Y_task1 = y[:, 0:50000]
	print(np.shape(X_task1))  # (784, 70000)
	print(np.shape(Y_task1))  # (1, 70000)

	# print(f"Y[0, 1:10]={Y_task1[0, 1:10]}")

	# without normlaization 
	print("\n\n--------- Fit model on the original data --------- ")
	# the best alpha should be 0.01
	# gammas = [-2] 
	gammas = range(-5, 4)
	_, cost_hist_original, acc_hist_original = fit_model(X_task1, Y_task1, 
		gammas = gammas, 
		figid="-org")

	# with standard normalization 
	print("\n\n--------- Fit model on the standard normalized data --------- ")
	normalizer, _  = standard_normalizer(X_task1)
	X_task1_norm = normalizer(X_task1)
	# comment this line to do learning rate tuning: 
	# Find the best alpha:  10
	# gammas = [1] 
	gammas = range(-5, 4)
	_, cost_hist_norm, acc_hist_norm = fit_model(X_task1_norm, Y_task1, 
		gammas=gammas, 
		figid="-std")

	# with PCA normalized 
	print("\n\n--------- Fit model on the PCA normalized data --------- ")
	normalizer, _ = PCA_sphereing(X_task1)
	X_task1_pca = normalizer(X_task1)
	# comment this line to do learning rate tuning. The best alpha should be  100
	# gammas = [2] 
	gammas = range(-5, 4)
	_, cost_hist_pca, acc_hist_pca = fit_model(X_task1_pca, Y_task1, 
		gammas=gammas , 
		figid="-pca")




	# compare cost histories 
	cost_histories = [cost_hist_original, cost_hist_norm, cost_hist_pca]
	labels = ["original", "standard normalized", "PCA normalized"]
	start = 0 
	figname="task1-cost-histories.png"
	print("final costs:", list(zip(labels, [item[-1] for item in cost_histories])))
	plot_cost_histories(cost_histories,labels, start, figname)

	for label, history in zip(labels, cost_histories): 
		figname=f"task1-cost-history-{label}.png"
		plot_cost_histories([history], [label], start, figname) 


	# compare accuracy histories 
	acc_histories = [acc_hist_original, acc_hist_norm, acc_hist_pca]
	figname = "task1-acc-histories.png"
	print("final accuracies:", list(zip(labels, [item[-1] for item in acc_histories])))
	# final accuracies: 
	# [('original', 0.608240008354187), 
	# ('standard normalized', 0.8695999979972839), 
	# ('PCA normalized', 0.9229599833488464)]
	plot_acc_histories(acc_histories, labels, start, figname)

	for label, history in zip(labels, cost_histories): 
		figname=f"task1-acc-history-{label}.png"
		plot_acc_histories([history], [label], start, figname)

	print("task 1 finished \n")
		

############################
# Helper functions for task 2.
############################

def model_task2(x, w): 
	"""
	input: 
	- x: shape (N, P) 
	- w: shape (N+1, )
	returns: 
	- regression result: shape (1, P)
	"""
	# print(f"w.shape: {w.shape}")
	# print(f"x.shape: {x.shape}")
	a = w[0] + jnp.dot(x.T,w[1:])
	return a.T 


# regression cost function 
def least_squares(w, x, y, lam):
	"""
	args: 
	- w: parameters of shape (N, ). 
	- x: the input of shape (N, P)
	- y: the label of shape (1, P)
	- lam (float): the penalty of the l1 regularizer 
	returns: 
	- the cost 
	"""
	# get batch of points
	x_p = x
	y_p = y

	# compute cost
	cost = jnp.sum((model_task2(x_p,w) - y_p)**2)

	# add l1 regularizer 
	cost += lam*jnp.sum(jnp.abs(w[1:]))

	# return average
	return cost/float(np.size(y_p))


def MSE(w, x, y): 
	return least_squares(w, x, y, lam=0)

#  gradient descent with a fixed learning rate 
def gradient_descent_task2(g, X, Y, w, alpha, max_its, lam):  
	"""
	Args: 
		- g: cost function 
		- X: input
		- Y: label 
		- w: initial parameters 
		- alpha (float): learning rate 
		- max_its (integer): maximum iteations 
		- lam: the penalty of l1 regularizer 
	"""
	grad_func = grad(g)

	# record history
	w_hist = [w]
	
	# item()  will convert the single-value DeviceArray to a Python float.
	cost_hist = [g(w, X, Y, lam).item()]
	mse_hist = [MSE(w, X, Y).item()]

	for k in range(max_its):   
		grad_eval = grad_func(w, X, Y, lam)
		# print(f"k={k}, grad_eval[:5]={grad_eval[:5]}")

		# take descent step
		w = w - alpha*grad_eval

		# record weight update
		w_hist.append(w)
		cost_hist.append(g(w, X, Y, lam).item())
		mse_hist.append(MSE(w, X, Y).item())

		# early stop
		# if the mse didn't improve for the last 10 iterations. early stop 
		# if abs(mse_hist[-1] - np.mean(mse_hist[-10:])) < 10e-8: 
		# 	print(f"early stop at k={k}, final mse={mse_hist[-1]}")
		# 	break 

	return w_hist, cost_hist, mse_hist 


def plot_mse_histories(mse_histories,labels, start=0, figname="mse_histories.png"):
	"""
	args: 
		- mse_histories: list of list 
		- labels: the legend used for labeling 
		- start: the start position to plot each line  
		- figname: the name used for the figure 
	return: 
		- None 
	
	Will save the figure using the figname 
	"""

	# initialize figure
	# plt.figure(figsize = (10,3))
	plt.figure()

	# run through input histories, plotting each beginning at 'start' iteration
	for c in range(len(mse_histories)):
		history = mse_histories[c]
		label = labels[c]
		    
		# plot cost function history
		plt.plot(np.arange(start,len(history),1),
			history[start:],
			linewidth = 3*(0.8)**(c),
			label = label) 

	# clean up panel / axes labels
	xlabel = 'step $k$'
	ylabel = 'mse'
	plt.xlabel(xlabel,fontsize = 14)
	plt.ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
	title = 'MSE history'
	plt.title(title,fontsize = 18)

	# plot legend
	anchor = (1,1)
	plt.legend(loc='upper right', bbox_to_anchor=anchor)
	plt.xlim([start - 0.5,len(history) - 0.5]) 
	# plt.show()
	plt.savefig(figname)


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
	print(f"y: {y[0, :10]}")

	# TODO: fill in your code 

	# normalize the data 
	normalizer, _  = standard_normalizer(x)
	# normalizer, _ = PCA_sphereing(x)
	x = normalizer(x)

	# initialize the parameters 
	N, P = x.shape
	np.random.seed(0)
	w = np.random.randn(N + 1)
	print(f"w.shape:{w.shape}")

	lambdas = [0, 50, 100, 150]
	max_its = 1000 
	alpha = 0.1 

	# compare cost and mse of different lambdas 
	cost_histories = [] 
	mse_histories = [] 
	labels = [] 

	for lam in lambdas: 
		print(f"\n\n------- Lambda = {lam} --------")
		# run gradient descent 
		w_hist, cost_hist, mse_hist = gradient_descent_task2(
			least_squares, x, y, w, alpha, max_its, lam)

		print(f"final cost: {cost_hist[-1]}, final mse: {mse_hist[-1]}")
		cost_histories.append(cost_hist)
		mse_histories.append(mse_hist)
		labels.append(f"lambda={lam}")

		# plot final weights 
		plt.figure()
		final_w = w_hist[-1][1:]
		plt.bar(range(1, len(final_w)+1), final_w)
		plt.ylabel('weight')
		plt.title(f"feature touching weights, lambda={lam}")
		plt.ylim([-4, 3])
		plt.savefig(f"task2-weights-lambda-{lam}.png")


	plot_cost_histories(cost_histories,labels, 
		start=20, figname="task2-cost_histories.png")
	plot_mse_histories(mse_histories, labels, 
		start=20, figname="task2-mse-histories.png")




if __name__ == '__main__':
	run_task1()
	run_task2() 




	