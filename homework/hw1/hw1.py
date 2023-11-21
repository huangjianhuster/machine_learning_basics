
import numpy as np 
import matplotlib.pyplot as plt 
import os 

# set random seed to make experiment reproducible 
np.random.seed(1) 


#################### Task 1 ###################
"""
Implement the random search function
Below we have a Python wrapper providing a skeleton for 
your production of of the random local search algorithm. 
All parts marked "TO DO" are for you to construct. 
Notice that the history of function evaluations returned is called cost_history. 
This is because - in the context of machine learning / deep learning - 
mathematical functions are often referred to as cost or loss functions.
"""

def random_search(g, w, \
    alpha = 1, \
    max_its = 10, \
    num_samples = 1000, \
    diminishing_steplength=False):
    """ 
    params: 
    - g: the function to optimize 
    - w: a numpy array, the parameters of g 
    - alpha: the learning rate 
    - max_its: the maximum number of iterations 
    - num_samples: the number of samples sampled in each iteration 
    - diminishing_steplength: if True, use diminishing the learning rate 

    returns: 
    - weight_history: a list of the weight values 
    - cost_history: a list of cost values evaluated with g 
    """
    

    # init 
    weight_history = [w]         # container for weight history
    cost_history = [g(w)]           # container for corresponding cost function history

    # random search with max_its number of iterations 
    for k in range(1,max_its+1):

        # TODO: put your code here. 

        # 1. for each iteration, from the given starting point w, need to randomize num_samples for the next move
        # 1.1 assume w is a vector with dimension of n: [a1, a2, ..., an]
        # 1.2 generate num_samples of randomized matrix (direction matrix) with a dimension of num_samples * n
        # 1.3 normalize the direction matrix
        directions  = np.random.randn(num_samples, np.size(w))
        # calculate the unit directions
        norms = np.linalg.norm(directions, axis=1)
        normalized_directions = directions / norms[:,None]
        # print(normalized_directions)

        # 2. apply the generated normalized matrix on the wi of the current iteration
        if not diminishing_steplength:
            new_ws = w + normalized_directions
        else:
            new_ws = w + normalized_directions * alpha / k
        # print(new_ws)

        # 3. apply functions to all rows of new_ws
        evaluations = np.apply_along_axis(g, 1, new_ws)
        # print(evaluations.shape)
        # 4. only go to the descending direction
        if np.min(evaluations) < cost_history[-1]:
            w = new_ws[np.argmin(evaluations)]
            print(f"updating w: {w}; and current value is {np.min(evaluations)}")

        # record weights and cost evaluation
        weight_history.append(w)
        cost_history.append(g(w))
        # print(weight_history, cost_history)

    print("Random search finished with K={K} iterations".format(K=max_its))
    return weight_history,cost_history


def run_task1():
    print("Run task 1 ....")
    # This is a test function. 
    # The random search should easily find the global optium x^*=0 
    g = lambda x: x**2
    w = np.array([-2])
    weight_history, cost_history = random_search(g, w, num_samples=5, max_its=5)

    print("weight_history", weight_history)
    print("cost_history", cost_history)
    
    # Uncomment the two lines below to check if your result is correct. 
    assert np.isclose(cost_history[-1], 0)==True, "The minimum cost should be zero"
    assert np.isclose(weight_history[-1], 0)==True, "The minimum x value should be zero"
    print("Task 1 finished.")


#################### Task 2 ###################
"""
In task 2, you should apply the implemented random search function to optimize 
the function described in the homework assignment. 
All parts marked "TODO" are for you to construct.
"""

# plot function 
def plot_cost_history(cost_history): 
    plt.figure() 
    plt.plot(np.arange(1, len(cost_history)+1), cost_history, marker='o')
    plt.xlabel("k")
    plt.ylabel('cost')
    plt.savefig("cost_history.png", bbox_inches='tight')
    plt.show()

def con_func(w): 
    """
    The function to be minimized.

    Params: 
    - w: the parameters of the function 
    Returns: 
    - cost: the value of the function
    """
    cost = None 

    # TODO: fill in the function definition
    cost = 100 * (w[1] - w[0]**2)**2 +(w[0]-1)**2

    return cost 

def run_task2(): 
    ### Apply the random search to optimize the task2_function  
    print("Run task 2 ....")
    # TODO: fill in the code; 
    #  you could use the plot_cost_history to produce the figure. 
    w = np.array([-2, -2])
    weight_history, cost_history = random_search(con_func, w, num_samples=1000, max_its=50)
    # print(cost_history)
    plot_cost_history(cost_history)

    print("Task 2 finished.")


#################### Task 3 ###################
"""
In task 3, you should improve the random search function to 
allow dinimishing learning rate. 
Then apply the random search function to the function again 
All parts marked "TODO" are for you to construct.

After you get the cost_history, you should plot the cost_history with a fixed learning rate
 and the dinimishing learning rate in the same figure to compare them.
"""

def compare_cost_history(costs_fixed, costs_diminished, out_png = None):
    plt.figure()
    plt.plot(np.arange(1, len(costs_fixed) + 1), costs_fixed, 'k-', marker="o", label="with fixed steplength")
    plt.plot(np.arange(1, len(costs_diminished) + 1), costs_diminished, "r-", marker='o', label="with diminishing steplength")
    plt.xlabel("k")
    plt.ylabel("cost")
    plt.legend()
    if out_png:
        plt.savefig(out_png, bbox_inches='tight')
        plt.show()
    else:
        plt.show()


def compare_contour(weights_fixed, weights_dinimished): 
    delta = 0.001
    x = np.arange(-5.0, 5.0, delta)
    y = np.arange(-5.0, 5.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = 100 * (Y - X * X)**2 + (X - 1)**2 
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Contour') 

    # plot weights on top of it 
    w1 = [x[0] for x in weights_fixed]
    w2 = [x[1] for x in weights_fixed]
    ax.plot(w1, w2, 'k-', marker='o', alpha=0.5)

    w1 = [x[0] for x in weights_dinimished]
    w2 = [x[1] for x in weights_dinimished]
    ax.plot(w1, w2, 'r-', marker='o', alpha=0.5)

    fig.savefig("contour.png", bbox_inches='tight')


def run_task3():
    print("Run task 3 ...") 

    # TODO: fill in the code 
    # You coudl use "compare_cost_history" and "compare_contour" to 
    # produce figures used for the report.
    w = np.array([-2, -2])
    weights_fixed, costs_fixed = random_search(con_func, w, num_samples=1000, max_its=50, diminishing_steplength=False)
    weights_diminished, costs_diminished = random_search(con_func, w, num_samples=1000, max_its=50, diminishing_steplength=True)
    # print(cost_history)
    compare_cost_history(costs_fixed, costs_diminished, out_png="comparison_whole.png")
    print("final weights for fixing steplength: ", weights_fixed[-1])
    print("final cost for fixing steplength: ", costs_fixed[-1])
    compare_cost_history(costs_fixed[10:], costs_diminished[10:], out_png="comparison_starts10.png")
    print("final weights for diminishing steplength: ", weights_diminished[-1])
    print("final cost for diminishing steplength: ", costs_diminished[-1])
    compare_contour(weights_fixed, weights_diminished)

    print("Task 3 finished.") 





if __name__ == '__main__':
    run_task1()
    run_task2()
    run_task3()




