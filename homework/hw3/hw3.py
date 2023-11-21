# hw3.py 


import jax.numpy as jnp 
from jax import grad 
# intro of jax library: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html 
import numpy as np 
import matplotlib.pyplot as plt 

datapath="./"

#################### Task 1 ###################

"""
Fit a linear regression model to the student debt data 
All parts marked "TO DO" are for you to construct.
"""

def model(x, w):
    a = w[0] + jnp.dot(x.T, w[1:])
    return a.T

def least_squares(w, x, y):
    cost = jnp.sum((model(x, w) -y)**2)
    return cost/float(y.size)

def training_plot(x, y, cost_history, w_history, save_path=None):
	x_flatten = x.flatten()

	# print(x_flatten)
	# transform w_history
	x_mean = np.mean(x_flatten)
	x_std = np.std(x_flatten)
	y_mean = np.mean(y)
	y_std = np.std(y)

	w0, w1 = w_history[-1]
	w0_original = (w0 - w1 * (x_mean / x_std)) * y_std + y_mean
	w1_original = w1 * y_std / x_std

	print(f"Training results: w0={w0_original}, w1={w1_original}")


	fig, axs = plt.subplots(1, 2, figsize=(12, 6))
	axs[0].plot(cost_history, "ro-", label="cost")
	axs[0].title.set_text('Cost')
	axs[0].legend(loc="upper right")

	axs[1].plot(x_flatten, y, "ko", label="original data")
	# for idx, wi in enumerate(w_history[-1:]):
	axs[1].plot(x_flatten, model(x, np.array([w0_original, w1_original])), "go-", label="predicted")
	axs[1].title.set_text('Model prediction')
	axs[1].legend(loc="upper right")

	# fig, axs = plt.subplots(1, 1, figsize=(8, 6))
	# axs.plot(x_flatten, y, "ko", label="original data")
	# # for idx, wi in enumerate(w_history[-1:]):
	# axs.plot(x_flatten, model(x, np.array([w0_original, w1_original])), "go-", label="predicted")
	# axs.set_title('Linear regression', size=16)
	# axs.tick_params(axis='both', which='major', labelsize=16)
	# axs.set_xlabel("Year", fontsize=16)
	# axs.set_ylabel("Debt", fontsize=16)
	# axs.legend(loc="upper left", frameon=False, fontsize=16)

	if save_path:
		plt.savefig(save_path)
	plt.show()
	return w0_original, w1_original

def get_closed_form(x, y):
	A = np.vstack([x, np.ones(len(x))]).T
	w1, w0 = np.linalg.lstsq(A, y, rcond=None)[0]
	return w0, w1

def gradient_descent_adam(cost_func, x, y, alpha=1e-2, iterations=200):
	gradient = grad(cost_func, argnums=0)

	w = np.array([1.,1.])	#np.random.rand(2)
	cost = cost_func(w, x, y)
	w_history = [w, ]
	cost_history = [cost, ]
	alpha = np.array([alpha, alpha])  # learning rate for w0 and w1
	beta = 0.9
	beta_2 = 0.999
	epsilon = 1e-8
	grad_history = [gradient(w, x, y), ]
	momentum_history = [np.zeros(2), ]
	secondary_list = [np.zeros(2), ]

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


def run_task1(): 
	# import the dataset
	csvname = datapath + 'student_debt_data.csv'
	data = np.loadtxt(csvname, delimiter=',')

	# extract input - for this dataset, these are times
	x = data[:,0]

	# extract output - for this dataset, these are total student debt
	y = data[:,1]

	print(np.shape(x))
	print(np.shape(y))

	# TODO: fit a linear regression model to the data
	print("-----------------------------------")
	print("Running task1")

	# feature scaling
	x_p = x.reshape(1, x.size)
	x_center = (x - np.mean(x)) / np.std(x)
	x_center_p = x_center.reshape(1, x.size)	# convert it to 2D array
	y_center = (y - np.mean(y)) / np.std(y)

	# gradient descent
	w_history, cost_history = gradient_descent_adam(least_squares, x_center_p, y_center, alpha=1e-2, iterations=200)

	# plot results
	w0_final, w1_final = training_plot(x_p, y, cost_history, w_history, save_path="task1.png")

	# get the closed-form solution
	w0_closed, w1_closed = get_closed_form(x_p.flatten(), y.flatten())
	print(f"The closed-form solution: w0={w0_closed}, w1={w1_closed}")

	# what is the debt in 2023
	pred_2030 = model(np.array([2030]), np.array([w0_final, w1_final]))
	# print(pred_2030)
	print("Predicted student loan in 2023: %.3f" % (float(pred_2030)) )


#################### Task 2 ###################

"""
Compare the least squares and the least absolute deviation costs 
All parts marked "TO DO" are for you to construct.
"""

def least_absoute(w, x, y):

	cost = jnp.sum(jnp.abs((model(x, w) -y)))

	return cost/float(y.size)

def run_task2():
	# load in dataset
	data = np.loadtxt(datapath + 'regression_outliers.csv',delimiter = ',')
	x = data[:-1,:]
	y = data[-1:,:] 

	print(np.shape(x))
	print(np.shape(y))

	# TODO: fit two linear models to the data
	x_p = x.reshape(1, x.size)
	print("-----------------------------------")
	print("Running task2")

	# Least square
	w_history_lsq, cost_history_lsq = gradient_descent_adam(least_squares, x_p, y, alpha=1e-2, iterations=400)
	print(f"weights from Least Squares training: w0 = {w_history_lsq[-1][0]}; w1={w_history_lsq[-1][1]}")

	# Least absolute
	w_history_abs, cost_history_abs = gradient_descent_adam(least_absoute, x_p, y, alpha=1e-2, iterations=400)
	print(f"weights from Least Absolutes training: w0 = {w_history_abs[-1][0]}; w1={w_history_abs[-1][1]}")

	# plot
	# fig, ax = plt.subplots(figsize=(8,6))
	# ax.plot(x.flatten(), y.flatten(), "ko", label="original data")
	# ax.plot(x.flatten(), model(x, np.array(w_history_lsq[-1]) ), "g-", label="Least squares")
	# ax.plot(x.flatten(), model(x, np.array(w_history_abs[-1]) ), "m-", label="Least absoutes")
	# ax.set_title('Least Squares Vs. Least Absolutes', size=16)
	# ax.tick_params(axis='both', which='major', labelsize=16)
	# ax.set_xlabel("x", fontsize=16)
	# ax.set_ylabel("y", fontsize=16)
	# ax.legend(loc="upper left", frameon=False, fontsize=16)
	# plt.savefig("task2.png")
	# plt.show()


	fig, axs = plt.subplots(1, 2, figsize=(12, 6))
	axs[0].plot(cost_history_lsq, "go-", label="Least squares")
	axs[0].plot(cost_history_abs, "mo-", label="Least absolutes")
	axs[0].set_title('Cost during training', size=16)
	axs[0].tick_params(axis='both', which='major', labelsize=16)
	axs[0].set_xlabel("Iterations", fontsize=16)
	axs[0].set_ylabel("Cost", fontsize=16)
	axs[0].legend(loc="upper left", frameon=False, fontsize=10)

	axs[1].plot(x.flatten(), y.flatten(), "ko", label="original data")
	axs[1].plot(x.flatten(), model(x, np.array(w_history_lsq[-1]) ), "g-", label="Least squares")
	axs[1].plot(x.flatten(), model(x, np.array(w_history_abs[-1]) ), "m-", label="Least absolutes")
	axs[1].set_title('Least Squares Vs. Least Absolutes', size=16)
	axs[1].tick_params(axis='both', which='major', labelsize=16)
	axs[1].set_xlabel("x", fontsize=16)
	axs[1].set_ylabel("y", fontsize=16)
	axs[1].legend(loc="upper left", frameon=False, fontsize=10)
	plt.savefig("task2-2.png")
	plt.show()




if __name__ == '__main__':
	run_task1()
	run_task2()


