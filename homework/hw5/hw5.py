import jax.numpy as jnp
import numpy as np
from jax import grad
import matplotlib.pyplot as plt

datapath = "./"

#################### Task 3 ###################

"""
Implementing the multi-class classification with Softmax cost; 
verify the implementation is correct by achiving small misclassification rate. 
"""


# A helper function to plot the original data
def show_dataset(x, y):
    y = y.flatten()
    num_classes = np.size(np.unique(y.flatten()))
    accessible_color_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    # initialize figure
    plt.figure()

    # color current class
    for a in range(0, num_classes):
        t = np.argwhere(y == a)
        t = t[:, 0]
        plt.scatter(
            x[0, t],
            x[1, t],
            s=50,
            color=accessible_color_cycle[a],
            edgecolor='k',
            linewidth=1.5,
            label="class:" + str(a))
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend(bbox_to_anchor=(1.1, 1.05))

    plt.savefig("data.png")
    plt.close()


def show_dataset_labels(x, y, modelf, n_axis_pts=120):
    y = y.flatten()
    num_classes = np.size(np.unique(y.flatten()))
    accessible_color_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    # initialize figure
    plt.figure()

    # fill in label regions using scatter points
    # get (x1, x2) for plot region
    anyax = np.linspace(0.05, 0.95, num=n_axis_pts)
    xx = np.meshgrid(anyax, anyax)
    xx_vars = np.reshape(xx, (2, n_axis_pts ** 2))
    # get class weights from classifier model
    z = modelf(xx_vars)
    # get class label from model output
    y_hat = z.argmax(axis=1)

    for a in range(0, num_classes):
        t = np.argwhere(y_hat == a)
        t = t[:, 0]
        plt.scatter(
            xx_vars[0, t],
            xx_vars[1, t],
            s=5,
            color=accessible_color_cycle[a],
            linewidth=1.5,
            label="class:" + str(a))

    # color current class
    for a in range(0, num_classes):
        t = np.argwhere(y == a)
        t = t[:, 0]
        plt.scatter(
            x[0, t],
            x[1, t],
            s=50,
            color=accessible_color_cycle[a],
            edgecolor='k',
            linewidth=1.5,
            label="class:" + str(a))
        plt.xlabel("x1")
        plt.ylabel("x2")
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.savefig("classifier_label_regions.png")
    plt.show()


def gradient_descent_adam(cost_func, x, y, alpha=1e-2, iterations=500):
    """
	:param cost_func:
	:param x: a array of N_features*N_samples; rows are each feature; each column is a sample
	:param y: a array of the same size of N_samples
	:param alpha: learning rate
	:param iterations: maximal iteration number
	:return: weights_history, cost_history
	"""
    gradient = grad(cost_func, argnums=0)

    # w = np.array([3.,3.])
    w_dim = (3, 4)
    w = np.random.rand(w_dim[0], w_dim[1])
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


def multiclass_softmax_cost(w, x, y, lamb=1e-1):
    x_append_transpose = jnp.vstack([jnp.ones(x.shape[1]), x]).T

    tmp = jnp.log(jnp.sum(jnp.exp(jnp.dot(x_append_transpose, w)), axis=1))
    w_yp_idx = y.astype('int').flatten()
    tmp_w_yp = jnp.sum(x_append_transpose*w[:, w_yp_idx].T, axis=1)

    w_touch_feature = w[1:, :]
    cost = jnp.sum(tmp - tmp_w_yp)  + lamb * jnp.sum(w_touch_feature**2)
    return cost / y.flatten().size


def run_task3():
    # load in dataset
    data = np.loadtxt(datapath + '4class_data.csv', delimiter=',')

    # get input/output pairs
    x = data[:-1, :]
    y = data[-1:, :]

    print(np.shape(x))
    print(np.shape(y))

    # show_dataset(x, y)

    # show data classified with dummy multiclass model
    # def dummy_classifier_model(xs):
    #     y_hats = np.zeros((np.shape(xs)[1], 4))
    #     ys = ((1 - xs[0, :]) > xs[1, :]).astype(int)
    #     ys[np.where(xs[0, :] > xs[1, :])] = ys[np.where(xs[0, :] > xs[1, :])] + 2
    #     for i, e in enumerate(ys):
    #         y_hats[i, e] = 1
    #     return y_hats
    #
    # show_dataset_labels(x, y, dummy_classifier_model)

    # TODO: fill in your code
    w_history, cost_history = gradient_descent_adam(multiclass_softmax_cost, x, y, alpha=1e-1, iterations=200)

    # plot cost history
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(cost_history, "go-", label="Cost")
    ax.set_xlabel("Iterations", fontsize=16)
    ax.set_ylabel("Cost", fontsize=16)
    plt.grid()
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("task3-cost.png")
    plt.show()

    # report final cost
    print(f"The final cost of my model: {cost_history[-1]}")
    print("final learnt wegihts: \n", w_history[-1])

    # Assess the accuracy of the final model
    x_append_transpose = jnp.vstack([jnp.ones(x.shape[1]), x]).T
    pred_values = jnp.dot(x_append_transpose, w_history[-1])
    pred_y = pred_values.argmax(axis=1)
    print(pred_y)
    miscalssfications = jnp.count_nonzero(pred_values.argmax(axis=1) - y)
    print("miscalssfications", miscalssfications)
    acc = (y.size - miscalssfications) / y.size
    print("accuracy: %.1f %%" % (acc*100))

    # A plot showing the original data and my classification areas
    # weights = w_history[-1]

    def ml_classifier_model(xs):
        y_hats = np.zeros((np.shape(xs)[1], 4))

        x_append_transpose = jnp.vstack([jnp.ones(xs.shape[1]), xs]).T
        pred_values = jnp.dot(x_append_transpose, w_history[-1])
        pred_y = pred_values.argmax(axis=1)

        # assign lables
        for i, e in enumerate(pred_y):
            y_hats[i, e] = 1
        return y_hats

    show_dataset_labels(x, y, ml_classifier_model, n_axis_pts=200)

if __name__ == '__main__':
    run_task3()
