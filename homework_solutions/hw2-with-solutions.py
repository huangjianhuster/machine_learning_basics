

import jax.numpy as jnp
import numpy as np  
from jax import grad 
# intro of jax library: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html 


import matplotlib.pyplot as plt 

#################### Task 1 ###################

"""
In this exercise you will implement gradient descent using the hand-computed derivative.
All parts marked "TO DO" are for you to construct.
"""
def cost_func(w):
	cost = 1/50*(w**4 + w**2 + 10*w)
	return cost

def gradient_func(w):
	g = 1/50*(4*w**3 + 2*w + 10)
	return g

def gradient_descent(g,grad,alpha,max_its,w):
	"""
	Params: 
	- g (cost function),
	- grad (gradient function) 
	- alpha (steplength parameter), 
	- max_its (maximum number of iterations), 
	- w (initialization)

	Returns: 
	- cost_history 
	"""

	# run the gradient descent loop
	cost_history = [g(w)]        # container for corresponding cost function history
	for k in range(1,max_its+1):       
		# evaluate the gradient, store current weights and cost function value
		## TO DO
		gradients = grad(w) 
		w = w - alpha * gradients 

		# take gradient descent step
		## TO DO
		    
		# collect final weights
		cost_history.append(g(w))  
	return cost_history



def run_task1(): 
	print("run task 1 ...")
	# initial point
	w = 2.0
	max_its = 1000
	
	print("Cost function:", cost_func(w))
	print("Gradient function:", gradient_func(w))
	
	# produce gradient descent runs
	alpha = 10**(0)
	cost_history_1 = gradient_descent(cost_func,gradient_func,alpha,max_its,w)

	alpha = 10**(-1)
	cost_history_2 = gradient_descent(cost_func,gradient_func,alpha,max_its,w)

	alpha = 10**(-2)
	cost_history_3 = gradient_descent(cost_func,gradient_func,alpha,max_its,w)

	# plot cost function histories
	## TO DO
	plt.figure() 
	plt.plot(np.arange(1, len(cost_history_1)+1), cost_history_1, "k-", label="alpha=1")
	plt.plot(np.arange(1, len(cost_history_2)+1), cost_history_2, "r-", label="alpha=0.1")
	plt.plot(np.arange(1, len(cost_history_2)+1), cost_history_3, "b-", label="alpha=0.01")
	plt.legend() 
	plt.xlabel("k")
	plt.ylabel('cost')
	plt.savefig("task1.png") 
	print("task 1 finished")



#################### Task 2 ###################

"""
In this exercise you will implement gradient descent 
using the automatically computed derivative.
All parts marked "TO DO" are for you to construct.
"""



def gradient_descent_auto(g,alpha,max_its,w, diminishing_alpha=False):
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
	# compute gradient module using autograd
	gradient = grad(g)

	# run the gradient descent loop
	weight_history = [w]           # container for weight history
	cost_history = [g(w)]          # container for corresponding cost function history
	for k in range(1, max_its+1):
		# evaluate the gradient, store current weights and cost function value
		## TO DO
		if diminishing_alpha: 
			alpha = 1.0 / k 

		d = gradient(w) 
		w = w - alpha * d
		# print("gradient is: {x}".format(x=d))
		# take gradient descent step
		## TO DO

		# record weight and cost
		weight_history.append(w)
		cost_history.append(g(w))
	return weight_history,cost_history

def run_task2(): 
	print("run task 2 ...")
	g = lambda w: jnp.absolute(w)
	# initial point
	w = 2.0
	max_its = 20

	# produce gradient descent runs
	alpha = 0.5
	_, cost_history_1 = gradient_descent_auto(g,alpha,max_its,w, diminishing_alpha=False)

	_, cost_history_2 = gradient_descent_auto(g, alpha, max_its, w, diminishing_alpha=True)

	plt.figure() 
	plt.plot(np.arange(1, len(cost_history_1)+1), cost_history_1, "k-", marker='o', label="alpha=0.5")
	plt.plot(np.arange(1, len(cost_history_2)+1), cost_history_2, "r-", marker='o', label="alpha=1/k")
	plt.legend() 
	plt.xlabel("k")
	plt.ylabel('cost')
	plt.savefig("task2.png") 
	print("task 2 finished")


if __name__ == '__main__':
	run_task1()
	run_task2() 



