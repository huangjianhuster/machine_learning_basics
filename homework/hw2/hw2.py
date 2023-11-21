

import jax.numpy as jnp
import numpy as np  
from jax import grad
import jax.numpy as jnp
# intro of jax library: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html 


import matplotlib.pyplot as plt 

#################### Task 1 ###################

"""
In this exercise you will implement gradient descent using the hand-computed derivative.
All parts marked "TO DO" are for you to construct.
"""
def cost_func(w):
	"""
	Params: 
	- w (weight)

	Returns: 
	- cost (value of cost function)
	"""
	## TODO: calculate the cost given w
	g = (w**4 + w**2 + 10*w) / 50
	return g

def gradient_func(w):
	"""
	Params: 
	- w (weight)

	Returns: 
	- grad (gradient of the cost function)
	"""
	## TODO: calculate the gradient given w
	grad_g = (4*w**3 + 2*w + 10) / 50
	return grad_g

def gradient_descent(g: object, gradient: object, alpha: object, max_its: object, w: object) -> object:
	"""
	Params: 
	- g (input function), 
	- gradient (gradient function that computes the gradients of the variable)
	- alpha (steplength parameter), 
	- max_its (maximum number of iterations), 
	- w (initialization)

	Returns: 
	- cost_history 
	"""

	# run the gradient descent loop
	cost_history = [g(w)]        # container for corresponding cost function history
	weights = [w]

	print("Cost of w0=2:", g(w))
	print("Gradient of w0=2:", gradient(w))

	for k in range(1,max_its+1):       
		# TODO: evaluate the gradient, store current weights and cost function value

		# evaluate the current gradient
		current_grad = gradient(w)

		# update weights
		w = w - alpha*current_grad	# gradient descent
		weights.append(w)

		# collect final weights
		cost_history.append(g(w))  
	return cost_history

# def cost_plt(cost_history, save_path=None):
# 	fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# 	ax.plot(np.arange(1, len(cost_history)+1, 1), cost_history, "r-", linewidth=2)
# 	ax.grid()
# 	ax.set_xlabel("Iterations")
# 	ax.set_ylabel("Cost")
#
# 	plt.tight_layout()
# 	if save_path:
# 		plt.savefig(save_path)
# 	plt.show()
#
# 	return None

def run_task1(): 
	print("run task 1 ...")
	# TODO: Three seperate runs using different steplength
	cost_hist1 = gradient_descent(g=cost_func, gradient=gradient_func, alpha=1, max_its=1000, w=2)
	cost_hist2 = gradient_descent(g=cost_func, gradient=gradient_func, alpha=0.1, max_its=1000, w=2)
	cost_hist3 = gradient_descent(g=cost_func, gradient=gradient_func, alpha=0.01, max_its=1000, w=2)

	# plot
	fig, ax = plt.subplots(1, 1, figsize=(10, 6))
	ax.plot(np.arange(1, len(cost_hist1)+1, 1), cost_hist1, "ro-", linewidth=2, label=r"${\alpha}$=1")
	ax.plot(np.arange(1, len(cost_hist2)+1, 1), cost_hist2, "bo-", linewidth=2, label=r"${\alpha}$=0.1")
	ax.plot(np.arange(1, len(cost_hist3)+1, 1), cost_hist3, "yo-", linewidth=2, label=r"${\alpha}$=0.01")
	ax.grid()
	ax.tick_params(axis='both', which='major', labelsize=16)
	ax.set_xlabel("Iterations", fontsize=16)
	ax.set_ylabel("Cost", fontsize=16)

	plt.tight_layout()
	plt.legend(frameon=False, fontsize=16)
	plt.savefig("task1.png")
	plt.show()

	print("task 1 finished")

#################### Task 2 ###################

"""
In this exercise you will implement gradient descent 
using the automatically computed derivative.
All parts marked "TO DO" are for you to construct.
"""



def gradient_descent_auto(g, alpha, max_its, w, diminishing_alpha=False):
	"""
	
	gradient descent function using automatic differentiator 
	Params: 
	- g (input function), 
	- alpha (steplength parameter), 
	- max_its (maximum number of iterations), 
	- w (initialization)
	
	Returns: 
	- weight_history
	- cost_history

	"""
	# TODO: compute gradient module using jax
	

	# run the gradient descent loop
	weight_history = [w]           # container for weight history
	cost_history = [g(w)]          # container for corresponding cost function history

	for k in range(1, max_its+1):
		# TODO: evaluate the gradient, store current weights and cost function value

		# evaluate current gradient using JAX
		gradient_function = grad(g)
		current_gradient = gradient_function(w)

		# update weights
		if diminishing_alpha:
			w = w - (alpha * current_gradient / k)	# diminishing step length
		else:
			w = w - alpha * current_gradient  # fixed step length

		# record weight and cost
		weight_history.append(w)
		cost_history.append(g(w))
	return weight_history, cost_history

def run_task2(): 
	print("run task 2 ...")
	# TODO: implement task 2

	# def func(x):
	# 	if x >= 0:
	# 		return x
	# 	else:
	# 		return x*-1

	# fixed
	func = lambda x:jnp.abs(x)
	weight_history_fixed, cost_history_fixed = gradient_descent_auto(g=func, alpha=0.5, max_its=20, w=2.0, diminishing_alpha=False)
	weight_history_diminishing, cost_history_diminishing = gradient_descent_auto(g=func, alpha=1, max_its=20, w=2.0, diminishing_alpha=True)

	# plot
	fig, ax = plt.subplots(1, 1, figsize=(10, 6))
	ax.plot(np.arange(1, len(cost_history_fixed)+1, 1), cost_history_fixed, "ro-", linewidth=2, label="fixed")
	ax.plot(np.arange(1, len(cost_history_diminishing)+1, 1), cost_history_diminishing, "bo-", linewidth=2, label=r"diminishing")
	ax.grid()
	ax.set_xticks(np.arange(1, len(cost_history_fixed)+1, 1))
	print(np.arange(1, len(cost_history_fixed)+1, 1))
	ax.tick_params(axis='both', which='major', labelsize=16)
	ax.set_xlabel("Iterations", fontsize=16)
	ax.set_ylabel("Cost", fontsize=16)

	plt.tight_layout()
	plt.legend(frameon=False, fontsize=16)
	plt.savefig("task2.png")
	plt.show()

	print("task 2 finished")


if __name__ == '__main__':
	run_task1()
	run_task2() 



