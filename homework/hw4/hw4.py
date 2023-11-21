import numpy as np
from jax import grad
import jax.numpy as jnp 
import matplotlib.pyplot as plt 

datapath = "./"

#################### Task 1 ###################

"""
Implementing the linear classification with Softmax cost; 
verify the implementation is correct by achiving zero misclassification. 
"""


def softmax_cost(w, x, y):
	# x_append_transpose = jnp.vstack([jnp.ones(len(x.shape)), x.flatten()]).T
	# tmp = -y.flatten()*jnp.dot(x_append_transpose, w)

	x_append_transpose = jnp.vstack([jnp.ones(x.shape[1]), x]).T
	tmp = -y * jnp.dot(x_append_transpose, w)
	cost = jnp.sum(jnp.log(1 + jnp.exp(tmp))) / y.flatten().size
	return cost

def perceptron(w, x, y):
	x_append_transpose = jnp.vstack([jnp.ones(x.shape[1]), x]).T
	tmp = -y * jnp.dot(x_append_transpose, w)
	cost = jnp.sum(jnp.where(tmp > 0, tmp, 0)) / y.flatten().size
	# cost = jnp.sum(jnp.log(1+jnp.exp(tmp)))/y.flatten().size
	return cost

def gradient_descent_adam(cost_func, x, y, w_initial=None, alpha=1e-2, iterations=500):
	"""
	:param w_initial:
	:param cost_func:
	:param x: a array of N_features*N_samples; rows are each feature; each column is a sample
	:param y: a array of the same size of N_samples
	:param alpha:
	:param iterations:
	:return:
	"""
	gradient = grad(cost_func, argnums=0)

	# w = np.array([3.,3.])
	w_dim = x.shape[0] + 1
	if w_initial is not None:
		w = w_initial
		assert len(w) == w_dim
	else:
		w = np.random.rand(w_dim)
	cost = cost_func(w, x, y)
	w_history = [w, ]
	cost_history = [cost, ]
	alpha_initial = np.empty(0)
	alpha_initial.fill(alpha)  # learning rate for w0 and w1
	beta = 0.9
	beta_2 = 0.999
	epsilon = 1e-8
	grad_history = [gradient(w, x, y), ]
	momentum_history = [np.zeros(w_dim), ]
	secondary_list = [np.zeros(w_dim), ]

	for i in np.arange(1, iterations, 1):
		# get gradient
		w_grad = gradient(w, x, y)
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
		cost = cost_func(w, x, y)
		cost_history.append(cost)

	return w_history, cost_history

def confusion_matrix(w, x, y):
	#															predicted
	#		   ----------------------------------------------------------------------------------------------------|
	#   	   |				|	class 1	(1)				|	class 2 (-1)			 |		Total			   |
	#		   |	class1 (1)	| class_positive1_correct	|  class_negative1_correct	 |	class_positive1_actual |
	#	actual |---------------------------------------------------------------------------------------------------|
	#		   |	class2 (-1) | class_positive1_incorrect |  class_negative1_incorrect |	class_negative1_actual |
	#		   |---------------------------------------------------------------------------------------------------|
	#		   |		Total	|  class_positive1_predict	|	class_negative1_predict  |	   Total_samples	   |
	#		   |---------------------------------------------------------------------------------------------------|
	#

	confusion_mat = {}
	x_append_transpose = jnp.vstack([jnp.ones(x.shape[1]), x]).T
	tmp = jnp.dot(x_append_transpose, w)
	predicted_classes = jnp.sign(tmp)
	class_positive1_actual = jnp.count_nonzero(y==1)
	class_negative1_actual = jnp.count_nonzero(y==-1)
	class_positive1_predict = jnp.count_nonzero(predicted_classes==1)
	class_negative1_predict = jnp.count_nonzero(predicted_classes==-1)

	class_positive1_incorrect = jnp.count_nonzero((predicted_classes-y)==2)
	class_positive1_correct = class_positive1_predict - class_positive1_incorrect
	class_negative1_incorrect = jnp.count_nonzero((predicted_classes-y)==-2)
	class_negative1_correct = class_negative1_predict - class_negative1_incorrect
	Total_samples = y.size
	# print("class_positive1_actual: ", class_positive1_actual)
	confusion_mat["class_positive1_actual"] = class_positive1_actual
	# print("class_negative1_actual: ", class_negative1_actual)
	confusion_mat["class_negative1_actual"] = class_negative1_actual
	# print("class_positive1_predict: ", class_positive1_predict)
	confusion_mat["class_positive1_predict"] = class_positive1_predict
	# print("class_negative1_predict: ", class_negative1_predict)
	confusion_mat["class_negative1_predict"] = class_negative1_predict
	# print("class_positive1_incorrect: ", class_positive1_incorrect)
	confusion_mat["class_positive1_incorrect"] = class_positive1_incorrect
	# print("class_positive1_correct: ", class_positive1_correct)
	confusion_mat["class_positive1_correct"] = class_positive1_correct
	# print("class_negative1_incorrect: ", class_negative1_incorrect)
	confusion_mat["class_negative1_incorrect"] = class_negative1_incorrect
	# print("class_negative1_correct: ", class_negative1_correct)
	confusion_mat["class_negative1_correct"] = class_negative1_correct
	# print("Total_samples: ", Total_samples)
	confusion_mat["Total_samples"] = Total_samples
	return confusion_mat

def run_task1():
	print("\n------------running task1------------")
	# load in data
	csvname = datapath + '2d_classification_data_v1.csv'
	data = np.loadtxt(csvname,delimiter = ',')

	# take input/output pairs from data
	x = data[:-1, :]
	y = data[-1:, :] 

	print(np.shape(x)) # (1, 11)
	print(np.shape(y)) # (1, 11)

	# TODO: fill in the rest of the code 
	w_history, cost_history = gradient_descent_adam(softmax_cost, x, y, w_initial=jnp.array([3.,3.]),\
													alpha=1e-1, iterations=2000)

	# report accuracy and misclassifications
	# the boundary
	x0 = -w_history[-1][0] / w_history[-1][1]
	print(f"The boundary: x = {x0}")

	confusion_mat = confusion_matrix(w_history[-1], x, y)
	misclassified = confusion_mat["class_positive1_incorrect"] + confusion_mat["class_negative1_incorrect"]
	print(f"Misclassified: {misclassified}")
	accuracy = (y.size - misclassified) / y.size
	print(f"Accuracy:{accuracy}")

	# plot cost history
	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	ax.plot(cost_history, "go-", label="Cost")
	ax.set_xlabel("Iterations", fontsize=16)
	ax.set_ylabel("Cost", fontsize=16)
	plt.grid()
	plt.legend(frameon=False)
	plt.tight_layout()
	plt.savefig("task1-cost.png")
	plt.show()

	# plot original data and the fitted tanh curve
	x_fitted = jnp.linspace(-1, 5, 200)
	y_fitted = jnp.tanh(w_history[-1][0] + w_history[-1][1] * x_fitted)
	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	ax.plot(x.flatten(), y.flatten(), "ko", label="original")
	ax.plot(x_fitted, y_fitted, "r-", label="fitted tanh() function")
	ax.set_xlabel("X", fontsize=16)
	ax.set_ylabel("Y", fontsize=16)
	plt.grid()
	plt.tight_layout()
	plt.legend(frameon=False)
	plt.savefig("task1-fitting.png")
	plt.show()


#################### Task 2 ###################

"""
Compare the efficacy of the Softmax and 
the Perceptron cost functions in terms of the 
minimal number of misclassifications each can 
achieve by proper minimization via gradient descent 
on a breast cancer dataset. 
"""

def run_task2():
	print("\n------------running task2------------")
	# data input
	csvname = datapath + 'breast_cancer_data.csv'
	data = np.loadtxt(csvname,delimiter = ',')

	# get input and output of dataset
	x = data[:-1, :]
	y = data[-1:, :] 

	print(np.shape(x)) # (8, 699)
	print(np.shape(y)) # (1, 699)
	
	# TODO: fill in the rest of the code
	w_history_percep, cost_history_percep = gradient_descent_adam(perceptron, x, y, w_initial=jnp.ones(x.shape[0]+1), alpha=1e-1, iterations=500)
	w_history_softmax, cost_history_softmax = gradient_descent_adam(softmax_cost, x, y, alpha=1e-1, iterations=500)

	# plot cost
	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	ax.plot(cost_history_softmax, "ro-", label="softmax")
	ax.plot(cost_history_percep, "bo-", label="perceptron")
	ax.set_xlabel("Iterations", fontsize=16)
	ax.set_ylabel("Cost", fontsize=16)
	plt.grid()
	plt.legend(frameon=False)
	plt.tight_layout()
	plt.savefig("task2.png")
	plt.show()

	print("\n### Summary of Perceptron ###")
	confusion_mat_percep = confusion_matrix(w_history_percep[-1], x, y)
	print(confusion_mat_percep)
	misclassified_percep = confusion_mat_percep["class_positive1_incorrect"] + confusion_mat_percep["class_negative1_incorrect"]
	print(f"Misclassified: {misclassified_percep}")
	accuracy_percep = (y.size - misclassified_percep) / y.size
	print(f"Accuracy:{accuracy_percep}")

	print("\n### Summary of Softmax ###")
	confusion_mat_softmax = confusion_matrix(w_history_softmax[-1], x, y)
	print(confusion_mat_softmax)
	misclassified_softmax = confusion_mat_softmax["class_positive1_incorrect"] + confusion_mat_softmax["class_negative1_incorrect"]
	print(f"Misclassified: {misclassified_softmax}")
	accuracy_softmax = (y.size - misclassified_softmax) / y.size
	print(f"Accuracy:{accuracy_softmax}")

	# plot accuracy over iterations
	acc_array_softmax = []
	for i in w_history_softmax:
		confusion_mat_tmp = confusion_matrix(i, x, y)
		misclassified_tmp = confusion_mat_tmp["class_positive1_incorrect"] + confusion_mat_tmp["class_negative1_incorrect"]
		acc_tmp = (y.size - misclassified_tmp) / y.size
		acc_array_softmax.append(acc_tmp)

	acc_array_percep = []
	for i in w_history_percep:
		confusion_mat_tmp = confusion_matrix(i, x, y)
		misclassified_tmp = confusion_mat_tmp["class_positive1_incorrect"] + confusion_mat_tmp["class_negative1_incorrect"]
		acc_tmp = (y.size - misclassified_tmp) / y.size
		acc_array_percep.append(acc_tmp)

	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	ax.plot(acc_array_softmax, "ro-", label="softmax")
	ax.plot(acc_array_percep, "bo-", label="perceptron")
	ax.set_xlabel("Iterations", fontsize=16)
	ax.set_ylabel("Accuracy", fontsize=16)
	ax.set_ylim([0, 1.2])
	plt.grid()
	plt.legend(frameon=False)
	plt.tight_layout()
	plt.savefig("task2-accuracy.png")
	plt.show()


if __name__ == '__main__':
	run_task1()
	run_task2()



