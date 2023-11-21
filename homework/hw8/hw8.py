import numpy as np
import pandas as pd
import jax
from jax import grad
import jax.numpy as jnp
from sklearn.datasets import fetch_openml
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd

datapath = "./"


def train_test_split(*arrays, test_size=0.2, shuffle=True, rand_seed=0):
    # set the random state if provided
    np.random.seed(rand_seed)

    # initialize the split index
    array_len = len(arrays[0].T)
    split_idx = int(array_len * (1 - test_size))

    # initialize indices to the default order
    indices = np.arange(array_len)

    # shuffle the arrays if shuffle is True
    if shuffle:
        np.random.shuffle(indices)

    # Split the arrays
    result = []
    for array in arrays:
        if shuffle:
            array = array[:, indices]
        train = array[:, :split_idx]
        test = array[:, split_idx:]
        result.extend([train, test])

    return result

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

def model(x, w): 
	"""
	input: 
	- x: shape (N, P)  
	- W: shape (N+1, 1) 

	output: 
	- prediction: shape (1, P) 
	"""
	# tack a 1 onto the top of each input point all at once
	o = jnp.ones((1, np.shape(x)[1]))
	f = jnp.vstack((o, x))
	# compute linear combination and return
	a = jnp.dot(f.T,w)
	return a

def softmax_cost(w, x, y, lamb):
    evals = model(x, w)
    # log-sum-exp trick
    max_eval = jnp.max(-y*evals)
    # cost = max_eval + jnp.log(1+jnp.exp(-y*evals-max_eval) )
    cost = jnp.sum(jnp.log(1 + jnp.exp(-y*evals))) 
    # cost = jnp.sum(  max_eval + jnp.log( jnp.exp(-max_eval) + jnp.exp(-y*evals - max_eval))  )
    # using l1 norm to regularize feature-touching weights
    w_feature_touching = w[1:]
    cost_regularizer = lamb*jnp.sum(jnp.abs(w_feature_touching))
    cost_total = cost + cost_regularizer
    return cost_total / y.flatten().size


def gradient_descent_adam(cost_func, x, y, w_initial=None, alpha=1e-2, lamb=1e-3, iterations=500):
	"""
	:param w_initial:
	:param cost_func:
	:param x: a array of N_features*N_samples; rows are each feature; each column is a sample
	:param y: a array of the same size of N_samples
	:param alpha: learning rate
    :param lamb: l1 norm regularization factor
	:param iterations: optimization iterastion
	:return: w_history, cost_history (both are in the list type)
	"""
	gradient = grad(cost_func, argnums=0)

	# w = np.array([3.,3.])
	w_dim = x.shape[0] + 1
	if w_initial is not None:
		w = w_initial
		assert len(w) == w_dim
	else:
		w = np.random.rand(w_dim)*0.005
	cost = cost_func(w, x, y, lamb)
	w_history = [w, ]
	cost_history = [cost, ]
	alpha_initial = np.empty(0)
	alpha_initial.fill(alpha)  # learning rate for w0 and w1
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

def eval_model_error(w, x, y):
    """
    - x: shape (N, P)  
    - w: shape (N+1, 1) 
    - y: shape (P, )
    """
    # print("w.shape: ", w.shape)
    # print("x.shape: ", x.shape)
    # print("y.shape: ", y.shape)
    evals = model(x, w)
    pred_y = jnp.sign(evals)
    miss = jnp.count_nonzero(pred_y-y) # nonzeror means the labels are different <-- missclassifications
    return miss / y.flatten().size

def k_fold_cross_validation(X, y, cost_function, hyperparams, k=5, rand_seed=0):
    np.random.seed(rand_seed)
    # this is for searching hyperparams: learning rate and lambda
    alpha, lamb = hyperparams
    
    # Shuffle the data
    indices = np.random.permutation(len(X[0]))
    X_shuffled = X[:, indices]
    y_shuffled = y[:, indices]
    
    # Split the data into k folds
    fold_size = len(X[0]) // k
    folds_X = [X_shuffled[:, i * fold_size:(i + 1) * fold_size] for i in range(k)]
    folds_y = [y_shuffled[:, i * fold_size:(i + 1) * fold_size] for i in range(k)]
    
    validation_error_list = []
    cost_kfold = []
    for i in range(k):
        # Select the current fold as the validation set
        X_val, y_val = folds_X[i], folds_y[i]

        # Use the remaining folds for training
        X_train = np.concatenate([fold for j, fold in enumerate(folds_X) if j != i], axis=1)
        y_train = np.concatenate([fold for j, fold in enumerate(folds_y) if j != i], axis=1)

        # Train your model (replace this with your actual model training code)
        # model.fit(X_train, y_train)
        w_history, cost_history = gradient_descent_adam(cost_function, X_train, y_train, w_initial=None,\
                                                alpha=alpha, lamb=lamb, iterations=100)
        # print(cost_history[-1])
        if cost_history[-1] > 5.0:
            print(f"searching alpha={alpha}, lambda={lamb} in fold {i}: the cost function is not fully converged {cost_history[-1]}!")
        cost_kfold.append(cost_history)

        # Evaluate the model on the validation set
        validation_error = eval_model_error(w_history[-1], X_val, y_val)
        validation_error_list.append(validation_error)
    
    # plot
    # for idx,j in enumerate(cost_kfold):
    #     plt.plot(j, label=str(idx))
    # plt.legend()
    # plt.show()
    return np.mean(validation_error_list)

def run_task1():
    csvname = datapath + 'new_gene_data.csv'
    data = np.loadtxt(csvname, delimiter=',')
    x = data[:-1, :]
    y = data[-1:, :]

    print(np.shape(x))  # (7128, 72)
    print(np.shape(y))  # (1, 72)

    np.random.seed(0)  # fix randomness
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, rand_seed=0)

    # TODO
    x_train_standardized = standard_normalization(x_train)
    x_train_mean = np.mean(x_train, axis=1)
    x_train_std = np.std(x_train, axis=1)


    # search learning rate
    alpha_list = [0.1, 0.05, 0.01, 0.001, 0.0001]
    # searching regularizer
    lamb_list = [5., 2., 1., 0.1, 0.05, 0.01, 0.001]
    # create search combinations
    hyperparameter_combination_list = list(product(alpha_list, lamb_list))

    """
    results = []
    for hyperparams in hyperparameter_combination_list:
        averaged_kfold_valid_error = k_fold_cross_validation(x_train_standardized, y_train, softmax_cost,\
                                                              hyperparams=hyperparams, k=4, rand_seed=0)
        results.append((hyperparams, averaged_kfold_valid_error))
        print(f"searching alpha={hyperparams[0]}, lambda={hyperparams[1]}, averaged validation error: {averaged_kfold_valid_error}")

    # format searching results
    df = pd.DataFrame(index=lamb_list, columns=alpha_list)
    for i in results:
        alpha, lamb = i[0][0], i[0][1]
        df.loc[lamb][alpha] = i[1]
    print(df)
    df.to_csv("hyperparams_searching.csv")
    """

    # train on the whole training set
    w_history, cost_history = gradient_descent_adam(softmax_cost, x_train_standardized, y_train, w_initial=None,\
                                        alpha=0.01, lamb=5.0, iterations=200)
    # plot cost history data
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(cost_history, "go-")
    ax.set_xlabel("Iterations", fontsize=16)
    ax.set_ylabel("Cost", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title("Cost history plot", fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig("Cost_history_on_training_set.png")
    plt.show()

    # report the performance of the final model on the splitted testing data
    x_test_standardized = (x_test - x_train_mean.reshape(-1,1)) / x_train_std.reshape(-1,1) 
    error = eval_model_error(w_history[-1], x_test_standardized, y_test)
    accuracy = 1 - error
    print(f"Accuracy of the final model on the testing set: {accuracy}")

    # the 5 most important features
    final_feature_touching_weights = w_history[-1][1:]
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    ax.plot(np.arange(1, len(final_feature_touching_weights)+1, 1), final_feature_touching_weights, "o", markersize=4)

    indices = np.argsort(-np.abs(final_feature_touching_weights))
    for i in range(5):
        ax.plot(indices[i], final_feature_touching_weights[indices[i]], "o", color="r", markersize=8)
        ax.text(indices[i]-800, final_feature_touching_weights[indices[i]]-0.02, f"Top {i+1}, index={indices[i]+1}",\
                fontsize=10, color='red', ha='left', va='bottom')
    ax.set_xlabel("Feature index (gene)", fontsize=16)
    ax.set_xlim([1, 7128])
    ax.set_xticks([1, 1000, 2000, 3000, 4000, 5000, 6000, 7128])
    ax.set_ylabel("Weight value", fontsize=16)
    ax.set_title("The top 5 important features (Bias was removed)")
    plt.tight_layout()
    plt.savefig("top-5-important features.png")
    plt.show()

if __name__ == '__main__':
    run_task1()
